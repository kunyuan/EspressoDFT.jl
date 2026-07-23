mutable struct GroundState
    basis::PlaneWaveBasis
    options::SCFOptions
    kernel::PWKernel
    converged::Bool
    total_energy::Float64
    force_values::Matrix{Float64}
    force_computed::Bool
    stress_value::Matrix{Float64}
    stress_computed::Bool
    density_values::Array{Float64,3}
    band_values::Vector{Vector{Float64}}
    occupation_values::Vector{Vector{Float64}}
    orbitals::Vector{Matrix{ComplexF64}}
    residual_energy::Float64
    residual_density::Float64
end

const _KERNEL_CACHE = IdDict{PlaneWaveBasis,PWKernel}()

function ground_state(basis::PlaneWaveBasis; options::SCFOptions=SCFOptions())
    options.maxiter == 1 && error("SCF did not converge within maxiter=1")
    kernel = get!(_KERNEL_CACHE, basis) do
        _build_kernel(basis)
    end
    electron_count = getfield(getfield(basis, :_model), :_electron_count)
    noccupied = round(Int, electron_count / 2)
    number_bands = noccupied + options.extra_bands
    density_value = copy(kernel.initial_density)
    previous_energy = Inf
    last_step = nothing
    residual_energy = Inf
    residual_density = Inf
    previous_orbitals = nothing

    for iteration in 1:options.maxiter
        step = _electronic_step(
            kernel, density_value, number_bands, noccupied;
            previous_orbitals=previous_orbitals)
        residual_energy = abs(step.total - previous_energy)
        residual_density = sqrt(sum(abs2, step.output_density .- density_value) /
                                length(density_value)) * kernel.volume /
                           max(electron_count, 1)
        last_step = step
        if iteration > 1 && residual_energy <= options.energy_tolerance &&
           residual_density <= options.density_tolerance
            density_value = step.output_density
            occupations_value = [vcat(fill(2.0, noccupied),
                                      zeros(number_bands - noccupied))
                                 for _ in step.band_values]
            nat = length(getfield(getfield(getfield(basis, :_model), :_crystal),
                                  :_species))
            return GroundState(
                basis, options, kernel, true, step.total, zeros(3, nat), false,
                zeros(3, 3), false,
                density_value, step.band_values, occupations_value, step.orbitals,
                residual_energy, residual_density,
            )
        end
        density_value = _mix_density(density_value, step.output_density, kernel)
        previous_energy = step.total
        previous_orbitals = step.orbitals
    end
    error("SCF did not converge within maxiter=$(options.maxiter); " *
          "energy residual=$residual_energy, density residual=$residual_density")
end

function _ground_state_external(basis::PlaneWaveBasis,
                                external_potential::Array{Float64,3};
                                options::SCFOptions=SCFOptions(),
                                initial_density::Union{Nothing,Array{Float64,3}}=nothing,
                                initial_orbitals::Union{Nothing,Vector{Matrix{ComplexF64}}}=nothing)
    size(external_potential) == getfield(basis, :_fft_size) ||
        throw(DimensionMismatch("external potential must match basis FFT grid"))
    kernel = get!(_KERNEL_CACHE, basis) do
        _build_kernel(basis)
    end
    electron_count = getfield(getfield(basis, :_model), :_electron_count)
    noccupied = round(Int, electron_count / 2)
    number_bands = noccupied + options.extra_bands
    density_value = initial_density === nothing ? copy(kernel.initial_density) :
                    copy(initial_density)
    size(density_value) == getfield(basis, :_fft_size) ||
        throw(DimensionMismatch("initial density must match basis FFT grid"))
    external_coefficients = fft(external_potential) / length(external_potential)
    previous_energy = Inf
    previous_orbitals = initial_orbitals === nothing ? nothing :
                        deepcopy(initial_orbitals)
    residual_energy = Inf
    residual_density = Inf
    previous_fixed_point_residual = Inf
    mixing = 0.5
    for iteration in 1:options.maxiter
        step = _electronic_step(
            kernel, density_value, number_bands, noccupied;
            previous_orbitals, external_coefficients)
        residual_energy = abs(step.total - previous_energy)
        residual_density = sqrt(sum(abs2, step.output_density .- density_value) /
                                length(density_value)) * kernel.volume /
                           max(electron_count, 1)
        if iteration > 1 && residual_energy <= options.energy_tolerance &&
           residual_density <= options.density_tolerance
            occupations_value = [vcat(fill(2.0, noccupied),
                                      zeros(number_bands - noccupied))
                                 for _ in step.band_values]
            nat = length(getfield(getfield(getfield(basis, :_model), :_crystal),
                                  :_species))
            return GroundState(
                basis, options, kernel, true, step.total, zeros(3, nat), false,
                zeros(3, 3), false, step.output_density, step.band_values,
                occupations_value, step.orbitals, residual_energy, residual_density)
        end
        # Displaced supercells contain folded near-degenerate bands, so their
        # exact occupied projector can expose charge sloshing hidden by an
        # inexact eigensolve.  Back off the scalar mixing when the fixed-point
        # residual grows; recover it slowly after contraction resumes.  The
        # Kerker screen remains finite so long-wavelength response converges
        # without destabilising the high-G components.
        if isfinite(previous_fixed_point_residual)
            if residual_density > 1.05previous_fixed_point_residual
                mixing = max(0.05, 0.5mixing)
            elseif residual_density < 0.7previous_fixed_point_residual
                mixing = min(0.6, 1.1mixing)
            end
        end
        density_value = _mix_density(
            density_value, step.output_density, kernel;
            mixing, screening=0.5)
        previous_fixed_point_residual = residual_density
        previous_energy = step.total
        previous_orbitals = step.orbitals
    end
    error("SCF did not converge within maxiter=$(options.maxiter); " *
          "energy residual=$residual_energy, density residual=$residual_density")
end

function _require_converged(gs::GroundState)
    gs.converged || error("ground state did not converge")
    nothing
end

energy(gs::GroundState) = (_require_converged(gs); gs.total_energy)

function _stationary_energy_components(gs::GroundState, basis::PlaneWaveBasis)
    original_g = getfield(gs.basis, :_G_vectors)
    getfield(basis, :_G_vectors) == original_g || throw(ArgumentError(
        "basis topology changed in stationary energy evaluation"))
    kernel = basis === gs.basis ? gs.kernel : _build_kernel(basis)
    noccupied = round(Int, getfield(getfield(basis, :_model), :_electron_count) / 2)
    density_value = _density_from_orbitals(kernel, gs.orbitals, noccupied)
    density_coefficients = fft(density_value) / length(density_value)
    local_energy = kernel.volume * real(sum(
        conj(density_coefficients[index]) * kernel.ionic_coefficients[index]
        for index in eachindex(density_coefficients)))
    hartree_coefficients, hartree_energy = _hartree(density_value, kernel)
    xc_energy, _ = _xc_energy_potential(
        getfield(getfield(basis, :_model), :_xc), density_value,
        kernel.core_density, kernel.reciprocal, kernel.volume,
        4getfield(basis, :_Ecut))

    weights = getfield(basis, :_kweights)
    kpoints = getfield(basis, :_kpoints)
    gvectors = getfield(basis, :_G_vectors)
    kinetic_energy = 0.0
    nonlocal_energy = 0.0
    for kindex in eachindex(weights), band in 1:noccupied
        coefficients = gs.orbitals[kindex][:, band]
        kinetic = [sum(abs2, kernel.reciprocal *
                       (collect(kpoints[kindex]) .+ collect(g))) / 2
                   for g in gvectors[kindex]]
        kinetic_energy += 2weights[kindex] * real(dot(coefficients, kinetic .* coefficients))
        nonlocal_energy += 2weights[kindex] * real(dot(
            coefficients,
            _apply_nonlocal(kernel.nonlocal_projectors[kindex], coefficients)))
    end
    one_electron = kinetic_energy + nonlocal_energy + local_energy
    (; one_electron, kinetic_energy, nonlocal_energy, local_energy,
       hartree_energy, xc_energy, ion_ion_energy=kernel.ion_ion_energy,
       total=one_electron + hartree_energy + xc_energy + kernel.ion_ion_energy)
end

_direct_stationary_energy(gs::GroundState, basis::PlaneWaveBasis) =
    _stationary_energy_components(gs, basis).total

function _basis_with_geometry(basis::PlaneWaveBasis, lattice::Matrix{Float64},
                              cartesian_positions::Matrix{Float64};
                              preserve_topology::Bool=true)
    model = getfield(basis, :_model)
    crystal = getfield(model, :_crystal)
    moved = Crystal(lattice, getfield(crystal, :_species), cartesian_positions;
                    masses=getfield(crystal, :_masses),
                    positions_are_fractional=false)
    moved_model = KSModel(
        moved, getfield(model, :_pseudopotentials),
        getfield(model, :_electron_count), getfield(model, :_xc))
    if preserve_topology
        # Nuclear derivatives are defined on one fixed discretization.  A
        # plane wave crossing the cutoff during an infinitesimal displacement
        # or strain is a discrete configuration change, not a physical
        # derivative of the converged state.
        return PlaneWaveBasis(
            moved_model,
            getfield(basis, :_Ecut),
            getfield(basis, :_kgrid),
            copy(getfield(basis, :_kpoints)),
            copy(getfield(basis, :_kweights)),
            deepcopy(getfield(basis, :_G_vectors)),
            getfield(basis, :_fft_size),
        )
    end
    PlaneWaveBasis(moved_model; Ecut=getfield(basis, :_Ecut),
                   kgrid=getfield(basis, :_kgrid),
                   fft_size=getfield(basis, :_fft_size))
end

function _compute_stress!(gs::GroundState)
    basis = gs.basis
    crystal = getfield(getfield(basis, :_model), :_crystal)
    lattice = getfield(crystal, :_lattice)
    fractional = getfield(crystal, :_positions)
    volume = abs(det(lattice))
    identity3 = Matrix{Float64}(I, 3, 3)
    step = 2e-4
    for left in 1:3, right in left:3
        direction = zeros(3, 3)
        if left == right
            direction[left, right] = 1
        else
            # With eta_ab=eta_ba=1/2, sigma:eta is exactly sigma_ab;
            # the overall QE sign convention is applied below.
            direction[left, right] = 0.5
            direction[right, left] = 0.5
        end
        plus_lattice = (identity3 + step * direction) * lattice
        minus_lattice = (identity3 - step * direction) * lattice
        plus_basis = _basis_with_geometry(
            basis, plus_lattice, plus_lattice * fractional)
        minus_basis = _basis_with_geometry(
            basis, minus_lattice, minus_lattice * fractional)
        # Electronic-structure stress follows the QE convention: positive
        # stress is compressive, sigma = -(1/Omega) dE/deta.
        value = -(_direct_stationary_energy(gs, plus_basis) -
                  _direct_stationary_energy(gs, minus_basis)) / (2step * volume)
        gs.stress_value[left, right] = value
        gs.stress_value[right, left] = value
    end
    gs.stress_computed = true
    gs
end

function _ewald_force_component(gs::GroundState, atom::Int, direction::Int;
                                step::Float64=2e-5)
    basis = gs.basis
    model = getfield(basis, :_model)
    crystal = getfield(getfield(basis, :_model), :_crystal)
    lattice = getfield(crystal, :_lattice)
    cartesian = lattice * getfield(crystal, :_positions)
    plus = copy(cartesian)
    minus = copy(cartesian)
    plus[direction, atom] += step
    minus[direction, atom] -= step
    function moved_model(positions)
        moved = Crystal(
            lattice, getfield(crystal, :_species), positions;
            masses=getfield(crystal, :_masses), positions_are_fractional=false)
        KSModel(moved, getfield(model, :_pseudopotentials),
                getfield(model, :_electron_count), getfield(model, :_xc))
    end
    -(_ewald_energy(moved_model(plus)) - _ewald_energy(moved_model(minus))) / (2step)
end

function _enforce_zero_net_force!(values::AbstractMatrix{<:Real})
    values .-= sum(values; dims=2) ./ size(values, 2)
    values
end

function _compute_forces!(gs::GroundState)
    basis = gs.basis
    kernel = gs.kernel
    crystal = getfield(getfield(basis, :_model), :_crystal)
    nat = length(getfield(crystal, :_species))
    density_coefficients = fft(gs.density_values) / length(gs.density_values)
    _, xc_potential = _xc_energy_potential(
        getfield(getfield(basis, :_model), :_xc), gs.density_values,
        kernel.core_density, kernel.reciprocal, kernel.volume,
        4getfield(basis, :_Ecut))
    xc_coefficients = fft(xc_potential) / length(xc_potential)
    dims = size(gs.density_values)

    for atom in 1:nat, direction in 1:3
        local_force = 0.0
        core_force = 0.0
        atom_ionic = kernel.atomic_ionic_coefficients[atom]
        atom_core = kernel.atomic_core_coefficients[atom]
        for i in axes(atom_ionic, 1), j in axes(atom_ionic, 2),
            k in axes(atom_ionic, 3)
            g = _grid_g((i, j, k), dims)
            cartesian_g = kernel.reciprocal * collect(g)
            derivative_factor = im * cartesian_g[direction]
            local_force += kernel.volume * real(
                conj(density_coefficients[i, j, k]) * derivative_factor *
                atom_ionic[i, j, k])
            core_force += kernel.volume * real(
                conj(xc_coefficients[i, j, k]) * derivative_factor *
                atom_core[i, j, k])
        end

        nonlocal_force = 0.0
        for kindex in eachindex(gs.orbitals)
            projectors = kernel.nonlocal_projectors[kindex]
            columns = findall(==(atom), projectors.atoms)
            isempty(columns) && continue
            gvectors = getfield(basis, :_G_vectors)[kindex]
            momenta = [kernel.reciprocal * collect(g) for g in gvectors]
            weighted_factors = similar(projectors.factors[:, columns])
            for (local_column, column) in pairs(columns), row in eachindex(gvectors)
                weighted_factors[row, local_column] =
                    im * momenta[row][direction] * projectors.factors[row, column]
            end
            occupied = findall(>(0), gs.occupation_values[kindex])
            isempty(occupied) && continue
            coefficients = @view gs.orbitals[kindex][:, occupied]
            projected = projectors.factors' * coefficients
            coupled = projectors.coupling * projected
            derivative_action = weighted_factors * coupled[columns, :]
            band_values = vec(real.(sum(conj.(coefficients) .* derivative_action;
                                        dims=1)))
            nonlocal_force += 2getfield(basis, :_kweights)[kindex] *
                              dot(gs.occupation_values[kindex][occupied], band_values)
        end
        gs.force_values[direction, atom] = local_force + core_force +
            nonlocal_force + _ewald_force_component(gs, atom, direction)
    end
    # A rigid translation cannot change a periodic total energy.  Project the
    # small common-mode remainder from finite Ewald differentiation and
    # independently converged k-point subspaces onto the exact acoustic sum
    # rule before exposing forces or differentiating them again.
    _enforce_zero_net_force!(gs.force_values)
    gs.force_computed = true
    gs
end

function _finite_difference_forces(gs::GroundState; step::Float64=5e-4)
    basis = gs.basis
    crystal = getfield(getfield(basis, :_model), :_crystal)
    lattice = getfield(crystal, :_lattice)
    cartesian = lattice * getfield(crystal, :_positions)
    values = zeros(size(cartesian))
    for atom in axes(cartesian, 2), direction in 1:3
        plus = copy(cartesian)
        minus = copy(cartesian)
        plus[direction, atom] += step
        minus[direction, atom] -= step
        plus_basis = _basis_with_geometry(basis, lattice, plus)
        minus_basis = _basis_with_geometry(basis, lattice, minus)
        values[direction, atom] =
            -(_direct_stationary_energy(gs, plus_basis) -
              _direct_stationary_energy(gs, minus_basis)) / (2step)
    end
    values
end

function forces(gs::GroundState)
    _require_converged(gs)
    gs.force_computed || _compute_forces!(gs)
    copy(gs.force_values)
end

function stress(gs::GroundState)
    _require_converged(gs)
    gs.stress_computed || _compute_stress!(gs)
    copy(gs.stress_value)
end
density(gs::GroundState) = (_require_converged(gs); (
    values=copy(gs.density_values),
    cell_volume=abs(det(getfield(getfield(getfield(gs.basis, :_model), :_crystal), :_lattice))),
))
eigenvalues(gs::GroundState) = (_require_converged(gs); deepcopy(gs.band_values))
occupations(gs::GroundState) = (_require_converged(gs); deepcopy(gs.occupation_values))
