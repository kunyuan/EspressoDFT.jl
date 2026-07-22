struct Crystal
    _lattice::Matrix{Float64}
    _species::Vector{Symbol}
    _positions::Matrix{Float64}
    _masses::Vector{Float64}
    _positions_are_fractional::Bool
end

function Crystal(lattice::AbstractMatrix, species::AbstractVector{Symbol},
                 positions::AbstractMatrix; masses::AbstractVector,
                 positions_are_fractional::Bool=true)
    _require(size(lattice) == (3, 3), "lattice must be a 3×3 matrix")
    h = Matrix{Float64}(lattice)
    _require(all(isfinite, h), "lattice must contain only finite values")
    _require(abs(det(h)) > 100eps(Float64) * max(opnorm(h)^3, 1.0),
             "lattice must be nonsingular")
    n_atoms = length(species)
    _require(size(positions) == (3, n_atoms),
             "positions must be a 3×N matrix matching species")
    _require(length(masses) == n_atoms, "masses must have one value per species")
    pos = Matrix{Float64}(positions)
    mass = Vector{Float64}(masses)
    _require(all(isfinite, pos), "positions must contain only finite values")
    _require(all(x -> isfinite(x) && x > 0, mass),
             "masses must be finite and positive")
    fractional = positions_are_fractional ? pos : h \ pos
    fractional = mod.(fractional, 1.0)
    Crystal(h, collect(Symbol, species), fractional, mass, positions_are_fractional)
end

function Base.getproperty(crystal::Crystal, name::Symbol)
    name === :lattice && return copy(getfield(crystal, :_lattice))
    name === :species && return copy(getfield(crystal, :_species))
    name === :positions && return copy(getfield(crystal, :_positions))
    name === :masses && return copy(getfield(crystal, :_masses))
    getfield(crystal, name)
end

struct KSModel
    _crystal::Crystal
    _pseudopotentials::Dict{Symbol,UPFData}
    _electron_count::Float64
    _xc::Symbol
end

function KSModel(crystal::Crystal; pseudopotentials::AbstractDict,
                 xc::Symbol=:pbe, charge::Real=0, spin::Symbol=:unpolarized)
    _require(xc in (:lda, :pbe), "unsupported xc functional: $xc")
    _require(_finite_real(charge) && iszero(charge),
             "unsupported nonzero charge: $charge")
    _require(spin == :unpolarized, "unsupported spin mode: $spin")
    required = unique(getfield(crystal, :_species))
    supplied = Set(Symbol(key) for key in keys(pseudopotentials))
    missing = filter(element -> element ∉ supplied, required)
    missing_text = join(string.(missing), ", ")
    isempty(missing) || throw(ArgumentError(
        "missing pseudopotential for species $missing_text"))

    parsed = Dict{Symbol,UPFData}()
    for element in required
        path = pseudopotentials[element]
        path isa AbstractString || throw(ArgumentError(
            "pseudopotential path for $element must be a string"))
        upf = _read_upf(path)
        upf.element == element || throw(ArgumentError(
            "pseudopotential element $(upf.element) does not match species $element"))
        _functional_matches(upf, xc) || throw(ArgumentError(
            "pseudopotential functional $(upf.functional) is incompatible with xc=$xc"))
        parsed[element] = upf
    end
    count = sum(parsed[element].z_valence for element in getfield(crystal, :_species))
    _require(isfinite(count) && isinteger(count),
             "pseudopotential valence gives non-integer electron count $count")
    _require(iseven(round(Int, count)),
             "unpolarized fixed occupations require an even electron count")
    KSModel(crystal, parsed, count, xc)
end

function Base.getproperty(model::KSModel, name::Symbol)
    name === :crystal && return getfield(model, :_crystal)
    name === :electron_count && return getfield(model, :_electron_count)
    name === :xc && return getfield(model, :_xc)
    getfield(model, name)
end

struct SCFOptions
    energy_tolerance::Float64
    density_tolerance::Float64
    maxiter::Int
    extra_bands::Int
end

function SCFOptions(; energy_tolerance::Real=1e-10, density_tolerance::Real=1e-8,
                    maxiter::Integer=100, extra_bands::Integer=4)
    _require(_finite_real(energy_tolerance) && energy_tolerance > 0,
             "energy_tolerance must be finite and positive")
    _require(_finite_real(density_tolerance) && density_tolerance > 0,
             "density_tolerance must be finite and positive")
    _require(maxiter > 0, "maxiter must be positive")
    _require(extra_bands >= 0, "extra_bands must be nonnegative")
    SCFOptions(Float64(energy_tolerance), Float64(density_tolerance),
               Int(maxiter), Int(extra_bands))
end

struct AtomicDisplacement
    atom::Int
    direction::Int
    q::NTuple{3,Float64}
    function AtomicDisplacement(atom::Integer, direction::Integer,
                                q::NTuple{3,<:Real})
        _require(atom >= 1, "atom index must be at least one")
        _require(direction in 1:3, "direction must be in 1:3")
        _require(all(isfinite, q), "q must contain only finite values")
        new(Int(atom), Int(direction), _copy_tuple3(q))
    end
end

AtomicDisplacement(atom::Integer, direction::Integer,
                   q::Tuple{<:Real,<:Real,<:Real}) =
    AtomicDisplacement(atom, direction, _copy_tuple3(q))

struct PlaneWaveBasis
    _model::KSModel
    _Ecut::Float64
    _kgrid::NTuple{3,Int}
    _kpoints::Vector{NTuple{3,Float64}}
    _kweights::Vector{Float64}
    _G_vectors::Vector{Vector{NTuple{3,Int}}}
    _fft_size::NTuple{3,Int}
end

function _centered_grid_coordinate(i::Int, n::Int)
    x = (i - 1) / n
    x >= 0.5 ? x - 1.0 : x
end

function _full_kmesh(kgrid::NTuple{3,Int})
    [(Float64(_centered_grid_coordinate(i, kgrid[1])),
      Float64(_centered_grid_coordinate(j, kgrid[2])),
      Float64(_centered_grid_coordinate(k, kgrid[3])))
     for i in 1:kgrid[1] for j in 1:kgrid[2] for k in 1:kgrid[3]]
end

function _enumerate_gvectors(lattice::Matrix{Float64}, k::NTuple{3,Float64},
                             Ecut::Float64)
    reciprocal = TWO_PI .* inv(lattice)'
    radius = sqrt(2Ecut) / minimum(svdvals(reciprocal))
    ranges = ntuple(axis ->
        floor(Int, -k[axis] - radius) - 1:ceil(Int, -k[axis] + radius) + 1, 3)
    vectors = NTuple{3,Int}[]
    tolerance = 100eps(Ecut)
    for g1 in ranges[1], g2 in ranges[2], g3 in ranges[3]
        g = (g1, g2, g3)
        kinetic = sum(abs2, reciprocal * (collect(k) .+ collect(g))) / 2
        kinetic <= Ecut + tolerance && push!(vectors, g)
    end
    sort!(vectors)
end

function _required_fft_size(lattice::Matrix{Float64}, Ecut::Float64)
    reciprocal = TWO_PI .* inv(lattice)'
    density_cutoff = 4Ecut
    radius = sqrt(2density_cutoff) / minimum(svdvals(reciprocal))
    bound = ceil(Int, radius) + 1
    maximum_abs = zeros(Int, 3)
    for g1 in -bound:bound, g2 in -bound:bound, g3 in -bound:bound
        g = (g1, g2, g3)
        sum(abs2, reciprocal * collect(g)) / 2 < density_cutoff || continue
        for axis in 1:3
            maximum_abs[axis] = max(maximum_abs[axis], abs(g[axis]))
        end
    end
    # The density sphere touches its periodic image at 2*gmax+1.  Round that
    # minimal grid up to the same small-prime FFT orders used by QE/FFTW.
    ntuple(axis -> nextprod((2, 3, 5), 2maximum_abs[axis] + 1), 3)
end

function PlaneWaveBasis(model::KSModel; Ecut::Real,
                        kgrid::NTuple{3,<:Integer},
                        fft_size::Union{Nothing,NTuple{3,<:Integer}}=nothing)
    _require(_finite_real(Ecut) && Ecut > 0, "Ecut must be finite and positive")
    grid = (Int(kgrid[1]), Int(kgrid[2]), Int(kgrid[3]))
    _require(all(x -> x > 0, grid), "kgrid dimensions must be positive")
    kpoints = _full_kmesh(grid)
    weights = fill(inv(Float64(length(kpoints))), length(kpoints))
    lattice = getfield(getfield(model, :_crystal), :_lattice)
    gvectors = [_enumerate_gvectors(lattice, k, Float64(Ecut)) for k in kpoints]
    required = _required_fft_size(lattice, Float64(Ecut))
    selected = if fft_size === nothing
        required
    else
        explicit = (Int(fft_size[1]), Int(fft_size[2]), Int(fft_size[3]))
        _require(all(x -> x > 0, explicit), "fft_size dimensions must be positive")
        _require(all(explicit[i] >= required[i] for i in 1:3),
                 "fft_size is insufficient for the density cutoff; need at least $required")
        explicit
    end
    PlaneWaveBasis(model, Float64(Ecut), grid, kpoints, weights, gvectors, selected)
end

function Base.getproperty(basis::PlaneWaveBasis, name::Symbol)
    name === :model && return getfield(basis, :_model)
    name === :Ecut && return getfield(basis, :_Ecut)
    name === :kpoints && return copy(getfield(basis, :_kpoints))
    name === :kweights && return copy(getfield(basis, :_kweights))
    name === :G_vectors && return deepcopy(getfield(basis, :_G_vectors))
    name === :fft_size && return getfield(basis, :_fft_size)
    getfield(basis, name)
end

function _is_commensurate_q(basis::PlaneWaveBasis, q::NTuple{3,<:Real})
    grid = getfield(basis, :_kgrid)
    all(axis -> isapprox(q[axis] * grid[axis], round(q[axis] * grid[axis]);
                         atol=2e-10, rtol=0), 1:3)
end
