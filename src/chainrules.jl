_zero_field(value) = value isa AbstractArray ? zero(value) : zero(value)

# Pseudopotential paths, artifact resolution and all other discrete run
# configuration are outside the differentiable chart.  Shield Julia artifact
# lookup itself so AD frontends do not attempt to differentiate registry/TOML
# internals while evaluating an otherwise constant pseudopotential dictionary.
function ChainRulesCore.rrule(::typeof(Artifacts.artifact_meta), args...; kwargs...)
    value = Artifacts.artifact_meta(args...; kwargs...)
    function artifact_meta_pullback(_)
        (NoTangent(), ntuple(_ -> NoTangent(), length(args))...)
    end
    value, artifact_meta_pullback
end

function ChainRulesCore.rrule(::Type{VersionNumber}, value::AbstractString)
    parsed = VersionNumber(value)
    parsed, _ -> (NoTangent(), NoTangent())
end

function _cotangent_field(tangent, name::Symbol, default)
    tangent = unthunk(tangent)
    tangent isa AbstractZero && return default
    hasproperty(tangent, name) || return default
    value = unthunk(getproperty(tangent, name))
    value isa AbstractZero ? default : value
end

function ChainRulesCore.rrule(::Type{Crystal}, lattice::AbstractMatrix,
                              species::AbstractVector{Symbol},
                              positions::AbstractMatrix; masses::AbstractVector,
                              positions_are_fractional::Bool=true)
    crystal = Crystal(lattice, species, positions; masses, positions_are_fractional)
    function crystal_pullback(delta_crystal)
        lattice_bar = Matrix{Float64}(_cotangent_field(
            delta_crystal, :_lattice, zeros(3, 3)))
        fractional_bar = Matrix{Float64}(_cotangent_field(
            delta_crystal, :_positions, zeros(size(positions))))
        if positions_are_fractional
            positions_bar = fractional_bar
        else
            h = getfield(crystal, :_lattice)
            fractional = getfield(crystal, :_positions)
            positions_bar = h' \ fractional_bar
            lattice_bar .-= positions_bar * fractional'
        end
        (NoTangent(), lattice_bar, NoTangent(), positions_bar)
    end
    crystal, crystal_pullback
end

function ChainRulesCore.rrule(::Type{KSModel}, crystal::Crystal;
                              pseudopotentials::AbstractDict, xc::Symbol=:pbe,
                              charge::Real=0, spin::Symbol=:unpolarized)
    model = KSModel(crystal; pseudopotentials, xc, charge, spin)
    function model_pullback(delta_model)
        crystal_bar = _cotangent_field(delta_model, :_crystal, ZeroTangent())
        (NoTangent(), crystal_bar)
    end
    model, model_pullback
end

function ChainRulesCore.rrule(::Type{PlaneWaveBasis}, model::KSModel;
                              Ecut::Real,
                              kgrid::NTuple{3,<:Integer},
                              fft_size::Union{Nothing,NTuple{3,<:Integer}}=nothing)
    basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size)
    function basis_pullback(delta_basis)
        model_bar = _cotangent_field(delta_basis, :_model, ZeroTangent())
        (NoTangent(), model_bar)
    end
    basis, basis_pullback
end

function ChainRulesCore.rrule(::typeof(energy), gs::GroundState)
    value = energy(gs)
    value, delta -> (NoTangent(), Tangent{GroundState}(;
        total_energy=unthunk(delta)))
end

function ChainRulesCore.rrule(::typeof(forces), gs::GroundState)
    value = forces(gs)
    value, delta -> (NoTangent(), Tangent{GroundState}(;
        force_values=unthunk(delta)))
end

function ChainRulesCore.rrule(::typeof(density), gs::GroundState)
    value = density(gs)
    function density_pullback(delta)
        values_bar = _cotangent_field(delta, :values, zeros(size(value.values)))
        (NoTangent(), Tangent{GroundState}(; density_values=values_bar))
    end
    value, density_pullback
end

function _ground_state_geometry_pullback(gs::GroundState, delta_state)
    crystal = getfield(getfield(gs.basis, :_model), :_crystal)
    lattice = getfield(crystal, :_lattice)
    fractional = getfield(crystal, :_positions)
    masses = getfield(crystal, :_masses)
    nat = length(masses)
    cartesian_bar = zeros(3, nat)
    lattice_bar = zeros(3, 3)

    energy_bar = Float64(_cotangent_field(delta_state, :total_energy, 0.0))
    if !iszero(energy_bar)
        cartesian_bar .-= energy_bar .* forces(gs)
        if getfield(crystal, :_positions_are_fractional)
            volume = abs(det(lattice))
            lattice_bar .-= energy_bar .* volume .* stress(gs) * inv(lattice)'
        end
    end

    density_bar = _cotangent_field(
        delta_state, :density_values, zeros(size(gs.density_values)))
    if !iszero(norm(density_bar))
        for atom in 1:nat, direction in 1:3
            result = response(
                gs, AtomicDisplacement(atom, direction, (0.0, 0.0, 0.0)))
            cartesian_bar[direction, atom] +=
                real(sum(conj.(density_bar) .* result.delta_density))
        end
    end

    force_bar = _cotangent_field(
        delta_state, :force_values, zeros(3, nat))
    if !iszero(norm(force_bar))
        dynamical = dynamical_matrix(gs, (0.0, 0.0, 0.0))
        for atom_i in 1:nat, atom_j in 1:nat, a in 1:3, b in 1:3
            row = 3(atom_i - 1) + a
            column = 3(atom_j - 1) + b
            cartesian_bar[b, atom_j] -= force_bar[a, atom_i] *
                sqrt(masses[atom_i] * masses[atom_j]) * real(dynamical[row, column])
        end
    end

    fractional_bar = lattice' * cartesian_bar
    crystal_bar = Tangent{Crystal}(;
        _lattice=lattice_bar,
        _positions=fractional_bar,
    )
    Tangent{PlaneWaveBasis}(;
        _model=Tangent{KSModel}(; _crystal=crystal_bar))
end

function ChainRulesCore.rrule(::typeof(ground_state), basis::PlaneWaveBasis;
                              options::SCFOptions=SCFOptions())
    state = ground_state(basis; options)
    function ground_state_pullback(delta_state)
        basis_bar = _ground_state_geometry_pullback(state, unthunk(delta_state))
        (NoTangent(), basis_bar)
    end
    state, ground_state_pullback
end
