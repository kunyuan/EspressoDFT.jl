abstract type AbstractSCFBackend end
struct ThreadedCPUBackend <: AbstractSCFBackend end

# Optional package extensions add methods without widening the frozen export
# surface or making MPI/CUDA hard dependencies of the CPU package.
function mpi_ground_state end
function gpu_ground_state end

struct NonlocalProjectors
    factors::Matrix{ComplexF64}
    coupling::Matrix{Float64}
    atoms::Vector{Int}
end

const _RADIAL_PROJECTOR_CACHE =
    Dict{Tuple{UInt,Int,Float64},Float64}()
const _RADIAL_PROJECTOR_CACHE_LOCK = ReentrantLock()

function _cached_projector_transform(
    upf::UPFData,
    signature::UInt,
    projector::Int,
    q::Float64,
)
    key = (signature, projector, q)
    cached = lock(_RADIAL_PROJECTOR_CACHE_LOCK) do
        get(_RADIAL_PROJECTOR_CACHE, key, nothing)
    end
    cached === nothing || return cached
    value = _projector_transform(upf, projector, q)
    lock(_RADIAL_PROJECTOR_CACHE_LOCK) do
        get!(_RADIAL_PROJECTOR_CACHE, key, value)
    end
end

mutable struct HamiltonianBatchBuffer
    grid::Array{ComplexF64,4}
    forward_plan::Any
    inverse_plan::Any
    overlaps::Matrix{ComplexF64}
    coupled_overlaps::Matrix{ComplexF64}
end

mutable struct HamiltonianWorkspace
    dims::NTuple{3,Int}
    number_projectors::Int
    batches::Dict{Int,HamiltonianBatchBuffer}
end

mutable struct DensityWorkspace
    buffers::Vector{Array{ComplexF64,3}}
    inverse_plans::Vector{Any}
    partial_densities::Vector{Array{Float64,3}}
end

struct PWKernel
    basis::PlaneWaveBasis
    reciprocal::Matrix{Float64}
    volume::Float64
    ionic_coefficients::Array{ComplexF64,3}
    atomic_ionic_coefficients::Vector{Array{ComplexF64,3}}
    core_density::Array{Float64,3}
    atomic_core_coefficients::Vector{Array{ComplexF64,3}}
    initial_density::Array{Float64,3}
    nonlocal_projectors::Vector{NonlocalProjectors}
    ion_ion_energy::Float64
    fft_indices::Vector{Vector{Int}}
    kinetic_energies::Vector{Vector{Float64}}
    grid_q2::Array{Float64,3}
    hamiltonian_workspaces::Vector{HamiltonianWorkspace}
    density_workspace::DensityWorkspace
end

function _basis_geometry_cache(
    basis::PlaneWaveBasis,
    reciprocal::Matrix{Float64},
    nonlocal_projectors::Vector{NonlocalProjectors},
)
    dims = getfield(basis, :_fft_size)
    linear = LinearIndices(dims)
    fft_indices = [
        [linear[_fft_index(g, dims)...] for g in gvectors]
        for gvectors in getfield(basis, :_G_vectors)
    ]
    kinetic_energies = [
        [sum(abs2, reciprocal * (collect(kpoint) .+ collect(g))) / 2
         for g in gvectors]
        for (kpoint, gvectors) in
            zip(getfield(basis, :_kpoints), getfield(basis, :_G_vectors))
    ]
    grid_q2 = Array{Float64}(undef, dims)
    for index in CartesianIndices(grid_q2)
        grid_q2[index] = sum(abs2, reciprocal *
                            collect(_grid_g(Tuple(index), dims)))
    end
    hamiltonian_workspaces = [
        HamiltonianWorkspace(dims, size(projectors.factors, 2),
                             Dict{Int,HamiltonianBatchBuffer}())
        for projectors in nonlocal_projectors
    ]
    buffers = [zeros(ComplexF64, dims) for _ in fft_indices]
    inverse_plans = Any[
        plan_ifft!(buffer; flags=FFTW.ESTIMATE) for buffer in buffers
    ]
    partial_densities = [zeros(Float64, dims) for _ in fft_indices]
    density_workspace = DensityWorkspace(
        buffers, inverse_plans, partial_densities)
    (; fft_indices, kinetic_energies, grid_q2, hamiltonian_workspaces,
       density_workspace)
end

function PWKernel(
    basis::PlaneWaveBasis,
    reciprocal::Matrix{Float64},
    volume::Float64,
    ionic_coefficients::Array{ComplexF64,3},
    atomic_ionic_coefficients::Vector{Array{ComplexF64,3}},
    core_density::Array{Float64,3},
    atomic_core_coefficients::Vector{Array{ComplexF64,3}},
    initial_density::Array{Float64,3},
    nonlocal_projectors::Vector{NonlocalProjectors},
    ion_ion_energy::Float64,
)
    cache = _basis_geometry_cache(basis, reciprocal, nonlocal_projectors)
    PWKernel(
        basis, reciprocal, volume, ionic_coefficients,
        atomic_ionic_coefficients, core_density, atomic_core_coefficients,
        initial_density, nonlocal_projectors, ion_ion_energy,
        cache.fft_indices, cache.kinetic_energies, cache.grid_q2,
        cache.hamiltonian_workspaces, cache.density_workspace)
end

function _fft_index(g::NTuple{3,Int}, dims::NTuple{3,Int})
    (mod(g[1], dims[1]) + 1, mod(g[2], dims[2]) + 1, mod(g[3], dims[3]) + 1)
end

function _grid_g(index::NTuple{3,Int}, dims::NTuple{3,Int})
    (_signed_frequency(index[1], dims[1]),
     _signed_frequency(index[2], dims[2]),
     _signed_frequency(index[3], dims[3]))
end

function _species_structure_factor(crystal::Crystal, element::Symbol,
                                   g::NTuple{3,Int})
    species = getfield(crystal, :_species)
    positions = getfield(crystal, :_positions)
    sum(cis(-TWO_PI * dot(collect(g), positions[:, atom]))
        for atom in eachindex(species) if species[atom] == element;
        init=0.0 + 0.0im)
end

function _radial_grid_coefficients(basis::PlaneWaveBasis, transform::Function)
    model = getfield(basis, :_model)
    crystal = getfield(model, :_crystal)
    pseudos = getfield(model, :_pseudopotentials)
    dims = getfield(basis, :_fft_size)
    reciprocal = TWO_PI .* inv(getfield(crystal, :_lattice))'
    volume = abs(det(getfield(crystal, :_lattice)))
    density_cutoff = 4getfield(basis, :_Ecut)
    coefficients = zeros(ComplexF64, dims)
    cache = Dict{Tuple{Symbol,Float64},Float64}()
    for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]
        g = _grid_g((i, j, k), dims)
        q = norm(reciprocal * collect(g))
        q^2 / 2 <= density_cutoff + 100eps(density_cutoff) || continue
        value = 0.0 + 0.0im
        for element in keys(pseudos)
            radial = get!(cache, (element, q)) do
                transform(pseudos[element], q)
            end
            value += radial * _species_structure_factor(crystal, element, g)
        end
        coefficients[i, j, k] = value / volume
    end
    coefficients
end

function _single_atom_radial_grid_coefficients(
    basis::PlaneWaveBasis, atom::Int, transform::Function)
    model = getfield(basis, :_model)
    crystal = getfield(model, :_crystal)
    element = getfield(crystal, :_species)[atom]
    upf = getfield(model, :_pseudopotentials)[element]
    position = getfield(crystal, :_positions)[:, atom]
    dims = getfield(basis, :_fft_size)
    reciprocal = TWO_PI .* inv(getfield(crystal, :_lattice))'
    volume = abs(det(getfield(crystal, :_lattice)))
    density_cutoff = 4getfield(basis, :_Ecut)
    coefficients = zeros(ComplexF64, dims)
    cache = Dict{Float64,Float64}()
    for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]
        g = _grid_g((i, j, k), dims)
        q = norm(reciprocal * collect(g))
        q^2 / 2 <= density_cutoff + 100eps(density_cutoff) || continue
        radial = get!(cache, q) do
            transform(upf, q)
        end
        coefficients[i, j, k] = radial *
            cis(-TWO_PI * dot(collect(g), position)) / volume
    end
    coefficients
end

function _real_from_coefficients(coefficients::Array{ComplexF64,3})
    real.(ifft(coefficients) .* length(coefficients))
end

function _normalized_initial_density(basis::PlaneWaveBasis,
                                     coefficients::Array{ComplexF64,3})
    density = _real_from_coefficients(coefficients)
    model = getfield(basis, :_model)
    volume = abs(det(getfield(getfield(model, :_crystal), :_lattice)))
    target = getfield(model, :_electron_count)
    integral = sum(density) * volume / length(density)
    _require(integral > 0, "UPF atomic density has non-positive electron count")
    density .*= target / integral
    density
end

function _real_spherical_harmonics(l::Int, direction::AbstractVector{<:Real})
    x, y, z = direction
    l == 0 && return [inv(sqrt(4pi))]
    l == 1 && return sqrt(3 / (4pi)) .* [x, y, z]
    l == 2 && return [
        sqrt(15 / (4pi)) * x * y,
        sqrt(15 / (4pi)) * y * z,
        sqrt(5 / (16pi)) * (3z^2 - 1),
        sqrt(15 / (4pi)) * x * z,
        sqrt(15 / (16pi)) * (x^2 - y^2),
    ]
    throw(ArgumentError("nonlocal projector angular momentum l=$l is unsupported"))
end

function _nonlocal_projectors(basis::PlaneWaveBasis, kindex::Int,
                              reciprocal::Matrix{Float64}, volume::Float64)
    model = getfield(basis, :_model)
    crystal = getfield(model, :_crystal)
    positions = getfield(crystal, :_positions)
    species = getfield(crystal, :_species)
    pseudos = getfield(model, :_pseudopotentials)
    pseudo_signatures = Dict(
        element => hash(upf.raw, hash(upf.path))
        for (element, upf) in pseudos
    )
    kpoint = getfield(basis, :_kpoints)[kindex]
    gvectors = getfield(basis, :_G_vectors)[kindex]
    momenta = [reciprocal * (collect(kpoint) .+ collect(g)) for g in gvectors]
    magnitudes = norm.(momenta)
    descriptors = Tuple{Int,Int,Int,Int}[] # atom, l, real-harmonic, projector
    for atom in eachindex(species)
        upf = pseudos[species[atom]]
        for l in sort(unique(upf.projector_l))
            projector_indices = findall(==(l), upf.projector_l)
            for harmonic in 1:(2l + 1), projector in projector_indices
                push!(descriptors, (atom, l, harmonic, projector))
            end
        end
    end

    factors = zeros(ComplexF64, length(gvectors), length(descriptors))
    coupling = zeros(Float64, length(descriptors), length(descriptors))
    radial_cache = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    harmonic_cache = Dict{Tuple{Int,Int},Vector{Float64}}()
    normalization = 4pi / sqrt(volume)
    for (column, (atom, l, harmonic, projector)) in pairs(descriptors)
        upf = pseudos[species[atom]]
        radial = get!(radial_cache, (species[atom], projector)) do
            [_cached_projector_transform(
                 upf, pseudo_signatures[species[atom]], projector, q)
             for q in magnitudes]
        end
        for row in eachindex(gvectors)
            direction = iszero(magnitudes[row]) ? [0.0, 0.0, 1.0] :
                        momenta[row] ./ magnitudes[row]
            harmonics = get!(harmonic_cache, (l, row)) do
                _real_spherical_harmonics(l, direction)
            end
            phase = cis(-TWO_PI * dot(collect(gvectors[row]), positions[:, atom]))
            factors[row, column] = normalization * radial[row] *
                                   harmonics[harmonic] * phase
        end
        for (other, (other_atom, other_l, other_harmonic,
                     other_projector)) in pairs(descriptors)
            atom == other_atom && l == other_l && harmonic == other_harmonic || continue
            coupling[column, other] = upf.dij_ry[projector, other_projector] / 2
        end
    end
    NonlocalProjectors(factors, coupling, first.(descriptors))
end

_apply_nonlocal(projectors::NonlocalProjectors, vectors::AbstractVecOrMat) =
    projectors.factors * (projectors.coupling * (projectors.factors' * vectors))

function _nonlocal_matrix(projectors::NonlocalProjectors)
    Matrix(Hermitian(projectors.factors * projectors.coupling * projectors.factors'))
end

function _ewald_energy(model::KSModel)
    crystal = getfield(model, :_crystal)
    lattice = getfield(crystal, :_lattice)
    positions = lattice * getfield(crystal, :_positions)
    species = getfield(crystal, :_species)
    pseudos = getfield(model, :_pseudopotentials)
    charges = [pseudos[element].z_valence for element in species]
    volume = abs(det(lattice))
    reciprocal = TWO_PI .* inv(lattice)'
    shortest = minimum(norm.(eachcol(lattice)))
    eta = 5 / shortest
    real_shell = max(2, ceil(Int, 7 / (eta * minimum(svdvals(lattice)))))
    real_energy = 0.0
    for atom_i in eachindex(species), atom_j in eachindex(species),
        n1 in -real_shell:real_shell, n2 in -real_shell:real_shell,
        n3 in -real_shell:real_shell
        atom_i == atom_j && n1 == 0 && n2 == 0 && n3 == 0 && continue
        displacement = positions[:, atom_i] - positions[:, atom_j] +
                       lattice * [n1, n2, n3]
        distance = norm(displacement)
        real_energy += 0.5 * charges[atom_i] * charges[atom_j] *
                       erfc(eta * distance) / distance
    end

    reciprocal_cutoff = 2eta * sqrt(-log(1e-14))
    reciprocal_shell = ceil(Int, reciprocal_cutoff / minimum(svdvals(reciprocal))) + 1
    reciprocal_energy = 0.0
    fractional = getfield(crystal, :_positions)
    for g1 in -reciprocal_shell:reciprocal_shell,
        g2 in -reciprocal_shell:reciprocal_shell,
        g3 in -reciprocal_shell:reciprocal_shell
        g1 == 0 && g2 == 0 && g3 == 0 && continue
        g = (g1, g2, g3)
        cartesian = reciprocal * collect(g)
        q2 = sum(abs2, cartesian)
        q2 <= reciprocal_cutoff^2 || continue
        structure = sum(charges[atom] *
                        cis(-TWO_PI * dot(collect(g), fractional[:, atom]))
                        for atom in eachindex(species))
        reciprocal_energy += 2pi / volume * exp(-q2 / (4eta^2)) / q2 * abs2(structure)
    end
    self_energy = -eta / sqrt(pi) * sum(abs2, charges)
    # The ionic subsystem is charged. Its uniform-background term cancels the
    # corresponding G=0 electronic/ionic contribution in the neutral crystal.
    background_energy = -pi * sum(charges)^2 / (2eta^2 * volume)
    real_energy + reciprocal_energy + self_energy + background_energy
end

function _build_kernel(basis::PlaneWaveBasis)
    crystal = getfield(getfield(basis, :_model), :_crystal)
    reciprocal = TWO_PI .* inv(getfield(crystal, :_lattice))'
    volume = abs(det(getfield(crystal, :_lattice)))
    nat = length(getfield(crystal, :_species))
    atomic_ionic = [_single_atom_radial_grid_coefficients(
                        basis, atom, _local_radial_transform) for atom in 1:nat]
    ionic = sum(atomic_ionic)
    atomic_core = [_single_atom_radial_grid_coefficients(
                       basis, atom, _core_density_transform) for atom in 1:nat]
    core_coefficients = sum(atomic_core)
    atomic_coefficients = _radial_grid_coefficients(basis, _atomic_density_transform)
    core = _real_from_coefficients(core_coefficients)
    initial = _normalized_initial_density(basis, atomic_coefficients)
    kpoints = getfield(basis, :_kpoints)
    nonlocal = Vector{NonlocalProjectors}(undef, length(kpoints))
    Threads.@threads for kindex in eachindex(kpoints)
        nonlocal[kindex] = _nonlocal_projectors(basis, kindex, reciprocal, volume)
    end
    PWKernel(basis, reciprocal, volume, ionic, atomic_ionic, core, atomic_core,
             initial, nonlocal, _ewald_energy(getfield(basis, :_model)))
end

function _hartree(density::Array{Float64,3}, kernel::PWKernel)
    coefficients = fft(density) / length(density)
    potential_coefficients = zeros(ComplexF64, size(density))
    energy_value = 0.0
    dims = size(density)
    for i in axes(density, 1), j in axes(density, 2), k in axes(density, 3)
        q2 = kernel.grid_q2[i, j, k]
        iszero(q2) && continue
        q2 / 2 <= 4getfield(kernel.basis, :_Ecut) || continue
        potential_coefficients[i, j, k] = 4pi * coefficients[i, j, k] / q2
        energy_value += 2pi * kernel.volume * abs2(coefficients[i, j, k]) / q2
    end
    potential_coefficients, energy_value
end

function _hamiltonian(kernel::PWKernel, kindex::Int,
                      local_coefficients::Array{ComplexF64,3})
    basis = kernel.basis
    kpoint = getfield(basis, :_kpoints)[kindex]
    gvectors = getfield(basis, :_G_vectors)[kindex]
    matrix = _nonlocal_matrix(kernel.nonlocal_projectors[kindex])
    dims = size(local_coefficients)
    for column in eachindex(gvectors), row in eachindex(gvectors)
        difference = ntuple(axis -> gvectors[row][axis] - gvectors[column][axis], 3)
        matrix[row, column] += local_coefficients[_fft_index(difference, dims)...]
    end
    for index in eachindex(gvectors)
        momentum = kernel.reciprocal * (collect(kpoint) .+ collect(gvectors[index]))
        matrix[index, index] += sum(abs2, momentum) / 2
    end
    Matrix(Hermitian(matrix))
end

struct KSHamiltonian <: AbstractMatrix{ComplexF64}
    kernel::PWKernel
    kindex::Int
    local_potential::Array{Float64,3}
    kinetic::Vector{Float64}
    workspace::HamiltonianWorkspace
end

KSHamiltonian(kernel::PWKernel, kindex::Int,
              local_potential::Array{Float64,3},
              kinetic::Vector{Float64}) =
    KSHamiltonian(kernel, kindex, local_potential, kinetic,
                  kernel.hamiltonian_workspaces[kindex])

Base.size(operator::KSHamiltonian) =
    (length(operator.kinetic), length(operator.kinetic))

function _batch_buffer!(workspace::HamiltonianWorkspace, number_vectors::Int)
    get!(workspace.batches, number_vectors) do
        grid = zeros(ComplexF64, workspace.dims..., number_vectors)
        inverse_plan = plan_ifft!(
            grid, (1, 2, 3); flags=FFTW.ESTIMATE)
        forward_plan = plan_fft!(
            grid, (1, 2, 3); flags=FFTW.ESTIMATE)
        HamiltonianBatchBuffer(
            grid,
            forward_plan,
            inverse_plan,
            zeros(ComplexF64, workspace.number_projectors, number_vectors),
            zeros(ComplexF64, workspace.number_projectors, number_vectors),
        )
    end
end

function LinearAlgebra.mul!(output::AbstractVecOrMat, operator::KSHamiltonian,
                            input::AbstractVecOrMat)
    input_matrix = input isa AbstractVector ? reshape(input, :, 1) : input
    output_matrix = output isa AbstractVector ? reshape(output, :, 1) : output
    size(input_matrix, 1) == length(operator.kinetic) || throw(DimensionMismatch())
    size(output_matrix) == size(input_matrix) || throw(DimensionMismatch())
    dims = size(operator.local_potential)
    number_vectors = size(input_matrix, 2)
    batch = _batch_buffer!(operator.workspace, number_vectors)
    grid = batch.grid
    fill!(grid, 0)
    indices = operator.kernel.fft_indices[operator.kindex]
    grid_matrix = reshape(grid, :, number_vectors)
    for vector in 1:number_vectors, row in eachindex(indices)
        grid_matrix[indices[row], vector] = input_matrix[row, vector]
    end
    batch.inverse_plan * grid
    grid .*= reshape(operator.local_potential, dims..., 1)
    batch.forward_plan * grid
    for vector in 1:number_vectors, row in eachindex(indices)
        output_matrix[row, vector] = operator.kinetic[row] * input_matrix[row, vector] +
                                     grid_matrix[indices[row], vector]
    end
    projectors = operator.kernel.nonlocal_projectors[operator.kindex]
    if !isempty(batch.overlaps)
        mul!(batch.overlaps, projectors.factors', input_matrix)
        mul!(batch.coupled_overlaps, projectors.coupling, batch.overlaps)
        mul!(output_matrix, projectors.factors, batch.coupled_overlaps,
             true, true)
    end
    output
end

function _initial_subspace(kinetic::Vector{Float64}, number_bands::Int,
                           previous::Union{Nothing,Matrix{ComplexF64}})
    if previous !== nothing && size(previous) == (length(kinetic), number_bands)
        # An exactly folded or previously converged occupied subspace can
        # contain zero-residual directions.  A deterministic infinitesimal
        # complement component preserves the subspace to solver tolerance
        # while giving the restarted block iteration full numerical rank.
        initial = copy(previous)
        for column in 1:number_bands, row in eachindex(kinetic)
            initial[row, column] += 1e-7 *
                cis(0.22360679774997896 * row * column) / (1 + kinetic[row])
        end
        return Matrix(qr(initial).Q[:, 1:number_bands])
    end
    indices = sortperm(kinetic)[1:number_bands]
    initial = zeros(ComplexF64, length(kinetic), number_bands)
    for (column, row) in pairs(indices)
        initial[row, column] = 1
    end
    # A tiny deterministic dense component prevents exact free-electron
    # degeneracies from making the first block Gram matrix rank deficient.
    for column in 1:number_bands, row in eachindex(kinetic)
        initial[row, column] += 1e-5 *
            cis(0.17320508075688773 * row * column) / (1 + kinetic[row])
    end
    Matrix(qr(initial).Q[:, 1:number_bands])
end

function _iterative_eigensolve(kernel::PWKernel, kindex::Int,
                               local_coefficients::Array{ComplexF64,3},
                               number_bands::Int,
                               previous::Union{Nothing,Matrix{ComplexF64}};
                               target_residual::Float64=1e-10)
    _iterative_eigensolve_real(
        kernel, kindex, _real_from_coefficients(local_coefficients),
        number_bands, previous; target_residual)
end

function _iterative_eigensolve_real(
    kernel::PWKernel,
    kindex::Int,
    local_potential::Array{Float64,3},
    number_bands::Int,
    previous::Union{Nothing,Matrix{ComplexF64}};
    target_residual::Float64=1e-10,
)
    kinetic = kernel.kinetic_energies[kindex]
    operator = KSHamiltonian(
        kernel, kindex, local_potential, kinetic)
    _block_eigensolve(
        operator, kinetic, number_bands, previous; target_residual)
end

function _block_eigensolve(
    operator::AbstractMatrix{ComplexF64},
    kinetic::Vector{Float64},
    number_bands::Int,
    previous::Union{Nothing,Matrix{ComplexF64}};
    target_residual::Float64=1e-10,
)
    subspace = _initial_subspace(kinetic, number_bands, previous)
    applied = similar(subspace)
    mul!(applied, operator, subspace)
    # The default wavefunction residual sits comfortably below the 1e-8
    # density-response gate. Response source construction may override it
    # because its covariant finite difference divides orbital errors by dk.
    # Medium supercells combine many folded, nearly degenerate occupied
    # states.  Four blocks are not enough to separate that cluster reliably:
    # the Ritz residual can plateau even though every matrix-vector product is
    # accurate.  Keep a wider thick-restart space while remaining O(NG*Nb)
    # in stored vectors.
    maximum_subspace = max(8number_bands, number_bands + 24)
    last_residual = Inf

    for _ in 1:200
        projected = Hermitian(subspace' * applied)
        decomposition = eigen(projected, 1:number_bands)
        values = real.(decomposition.values)
        vectors = subspace * decomposition.vectors
        applied_vectors = applied * decomposition.vectors
        residual = applied_vectors .- vectors .* transpose(values)
        residual_norms = [norm(view(residual, :, band))
                          for band in 1:number_bands]
        last_residual = maximum(residual_norms)
        last_residual <= target_residual && return values, vectors

        unconverged = findall(>(target_residual), residual_norms)
        corrections = copy(residual[:, unconverged])
        for (column, band) in pairs(unconverged)
            denominator = kinetic .- values[band]
            regularized = ifelse.(denominator .>= 0, 1.0, -1.0) .*
                          max.(abs.(denominator), 0.5)
            corrections[:, column] ./= regularized
        end
        # Stable twice-applied projection followed by an SVD rank decision
        # prevents a nearly dependent residual block from entering the basis.
        corrections .-= subspace * (subspace' * corrections)
        corrections .-= subspace * (subspace' * corrections)
        factorization = svd(corrections; full=false)
        threshold = max(1e-12, first(factorization.S) * 1e-10)
        keep = findall(>(threshold), factorization.S)
        if isempty(keep)
            # The current Ritz vectors can be accurate while all remaining
            # preconditioned corrections lie numerically inside their span.
            # Re-seed an infinitesimal complement instead of failing on that
            # rank decision; the outer iteration still enforces the residual.
            subspace = _initial_subspace(kinetic, number_bands, vectors)
            applied = similar(subspace)
            mul!(applied, operator, subspace)
            continue
        end
        expansion = factorization.U[:, keep]
        applied_expansion = similar(expansion)
        mul!(applied_expansion, operator, expansion)

        if size(subspace, 2) + size(expansion, 2) > maximum_subspace
            subspace = hcat(vectors, expansion)
            applied = hcat(applied_vectors, applied_expansion)
        else
            subspace = hcat(subspace, expansion)
            applied = hcat(applied, applied_expansion)
        end
    end
    error("iterative eigensolver did not converge; maximum residual=$last_residual")
end

function _solve_electron_kpoint(
    kernel::PWKernel,
    local_coefficients::Array{ComplexF64,3},
    local_potential::Union{Nothing,Array{Float64,3}},
    number_bands::Int,
    previous::Union{Nothing,Vector{Matrix{ComplexF64}}},
    target_residual::Float64,
    kindex::Int,
)
    basis = kernel.basis
    dimension = length(getfield(basis, :_G_vectors)[kindex])
    number_bands <= dimension || error(
        "plane-wave basis is smaller than requested band count")
    if _use_dense_eigensolver(dimension, number_bands)
        matrix = _hamiltonian(kernel, kindex, local_coefficients)
        decomposition = eigen(Hermitian(matrix), 1:number_bands)
        return Vector{Float64}(decomposition.values),
               Matrix{ComplexF64}(decomposition.vectors)
    end
    prior = previous === nothing ? nothing : previous[kindex]
    _iterative_eigensolve_real(
        kernel, kindex, something(local_potential), number_bands, prior;
        target_residual)
end

function _solve_electrons_backend(
    ::ThreadedCPUBackend,
    kernel::PWKernel,
    local_coefficients::Array{ComplexF64,3},
    number_bands::Int,
    previous::Union{Nothing,Vector{Matrix{ComplexF64}}},
    target_residual::Float64,
)
    basis = kernel.basis
    nk = length(getfield(basis, :_kpoints))
    values = Vector{Vector{Float64}}(undef, nk)
    vectors = Vector{Matrix{ComplexF64}}(undef, nk)
    dimensions = length.(getfield(basis, :_G_vectors))
    local_potential = any(dimension ->
        !_use_dense_eigensolver(dimension, number_bands), dimensions) ?
        _real_from_coefficients(local_coefficients) : nothing
    Threads.@threads for kindex in 1:nk
        values[kindex], vectors[kindex] = _solve_electron_kpoint(
            kernel, local_coefficients, local_potential, number_bands,
            previous, target_residual, kindex)
    end
    values, vectors
end

_use_dense_eigensolver(dimension::Int, number_bands::Int) =
    dimension <= max(64, 4number_bands)

function _solve_electrons(
    kernel::PWKernel,
    local_coefficients::Array{ComplexF64,3},
    number_bands::Int;
    previous::Union{Nothing,Vector{Matrix{ComplexF64}}}=nothing,
    target_residual::Float64=1e-10,
    backend::AbstractSCFBackend=ThreadedCPUBackend(),
)
    _solve_electrons_backend(
        backend, kernel, local_coefficients, number_bands, previous,
        target_residual)
end

function _density_from_orbitals(kernel::PWKernel,
                                orbitals::Vector{Matrix{ComplexF64}},
                                noccupied::Int)
    basis = kernel.basis
    dims = getfield(basis, :_fft_size)
    weights = getfield(basis, :_kweights)
    workspace = kernel.density_workspace
    scale = prod(dims) / sqrt(kernel.volume)
    Threads.@threads for kindex in eachindex(orbitals)
        coefficients = workspace.buffers[kindex]
        plan = workspace.inverse_plans[kindex]
        partial_density = workspace.partial_densities[kindex]
        fill!(partial_density, 0)
        indices = kernel.fft_indices[kindex]
        for band in 1:noccupied
            fill!(coefficients, 0)
            for row in eachindex(indices)
                coefficients[indices[row]] = orbitals[kindex][row, band]
            end
            plan * coefficients
            factor = 2weights[kindex] * scale^2
            @inbounds @simd for index in eachindex(partial_density)
                partial_density[index] += factor * abs2(coefficients[index])
            end
        end
    end
    density = zeros(Float64, dims)
    for partial_density in workspace.partial_densities
        density .+= partial_density
    end
    density
end

function _mix_density(density::Array{Float64,3}, output::Array{Float64,3},
                      kernel::PWKernel; mixing::Float64=0.7,
                      screening::Float64=1.0)
    preconditioned = _precondition_density_residual(
        density, output, kernel; screening)
    mixed = density .+ mixing .* preconditioned
    _normalize_density!(mixed, kernel)
end

function _precondition_density_residual(
    density::Array{Float64,3}, output::Array{Float64,3}, kernel::PWKernel;
    screening::Float64=1.0,
)
    residual_coefficients = fft(output .- density) / length(density)
    dims = size(density)
    for i in axes(density, 1), j in axes(density, 2), k in axes(density, 3)
        g = _grid_g((i, j, k), dims)
        if all(iszero, g)
            residual_coefficients[i, j, k] = 0
        else
            q2 = kernel.grid_q2[i, j, k]
            residual_coefficients[i, j, k] *= q2 / (q2 + screening)
        end
    end
    real.(ifft(residual_coefficients) .* length(density))
end

function _normalize_density!(mixed::Array{Float64,3}, kernel::PWKernel)
    target_average = getfield(getfield(kernel.basis, :_model), :_electron_count) /
                     kernel.volume
    mixed .+= target_average - sum(mixed) / length(mixed)
    mixed
end

mutable struct DensityMixer
    damping::Float64
    screening::Float64
    max_history::Int
    densities::Vector{Array{Float64,3}}
    residuals::Vector{Array{Float64,3}}
end

function DensityMixer(; damping::Real=0.7, screening::Real=1.0,
                      max_history::Integer=8)
    _require(isfinite(damping) && 0 < damping <= 1,
             "density-mixer damping must lie in (0,1]")
    _require(isfinite(screening) && screening >= 0,
             "density-mixer screening must be finite and nonnegative")
    _require(max_history >= 1, "density-mixer history must be positive")
    DensityMixer(
        Float64(damping),
        Float64(screening),
        Int(max_history),
        Array{Float64,3}[],
        Array{Float64,3}[],
    )
end

function _history_value(
    history::Vector{Array{Float64,3}},
    current::Array{Float64,3},
    first_index::Int,
    offset::Int,
)
    index = first_index + offset
    index <= length(history) ? history[index] : current
end

function _difference_dot(
    left_after::Array{Float64,3}, left_before::Array{Float64,3},
    right_after::Array{Float64,3}, right_before::Array{Float64,3},
)
    value = 0.0
    @inbounds @simd for index in eachindex(left_after)
        value += (left_after[index] - left_before[index]) *
                 (right_after[index] - right_before[index])
    end
    value
end

function _difference_current_dot(
    after::Array{Float64,3}, before::Array{Float64,3},
    current::Array{Float64,3},
)
    value = 0.0
    @inbounds @simd for index in eachindex(after)
        value += (after[index] - before[index]) * current[index]
    end
    value
end

function _anderson_mix!(
    mixer::DensityMixer,
    density::Array{Float64,3},
    output::Array{Float64,3},
    kernel::PWKernel,
)
    residual = _precondition_density_residual(
        density, output, kernel; screening=mixer.screening)
    nhistory = min(length(mixer.densities), mixer.max_history)
    mixed = density .+ mixer.damping .* residual

    if nhistory > 0
        first_index = length(mixer.densities) - nhistory + 1
        gram = zeros(Float64, nhistory, nhistory)
        right_hand_side = zeros(Float64, nhistory)
        for column in 1:nhistory
            f_before = _history_value(
                mixer.residuals, residual, first_index, column - 1)
            f_after = _history_value(
                mixer.residuals, residual, first_index, column)
            right_hand_side[column] =
                _difference_current_dot(f_after, f_before, residual)
            for row in 1:column
                g_before = _history_value(
                    mixer.residuals, residual, first_index, row - 1)
                g_after = _history_value(
                    mixer.residuals, residual, first_index, row)
                value = _difference_dot(
                    f_after, f_before, g_after, g_before)
                gram[row, column] = value
                gram[column, row] = value
            end
        end
        scale = maximum(abs, gram; init=0.0)
        regularization = max(1e-14, 1e-10scale)
        @inbounds for index in 1:nhistory
            gram[index, index] += regularization
        end
        coefficients = try
            Symmetric(gram) \ right_hand_side
        catch
            fill(NaN, nhistory)
        end
        if all(isfinite, coefficients)
            @inbounds for column in 1:nhistory
                x_before = _history_value(
                    mixer.densities, density, first_index, column - 1)
                x_after = _history_value(
                    mixer.densities, density, first_index, column)
                f_before = _history_value(
                    mixer.residuals, residual, first_index, column - 1)
                f_after = _history_value(
                    mixer.residuals, residual, first_index, column)
                coefficient = coefficients[column]
                @simd for index in eachindex(mixed)
                    mixed[index] -= coefficient * (
                        x_after[index] - x_before[index] +
                        mixer.damping * (f_after[index] - f_before[index]))
                end
            end
        else
            empty!(mixer.densities)
            empty!(mixer.residuals)
        end
    end

    push!(mixer.densities, copy(density))
    push!(mixer.residuals, residual)
    while length(mixer.densities) > mixer.max_history
        popfirst!(mixer.densities)
        popfirst!(mixer.residuals)
    end
    _normalize_density!(mixed, kernel)
end

function _electronic_step(kernel::PWKernel, density_value::Array{Float64,3},
                          number_bands::Int, noccupied::Int;
                          previous_orbitals::Union{Nothing,Vector{Matrix{ComplexF64}}}=nothing,
                          external_coefficients::Union{Nothing,Array{ComplexF64,3}}=nothing,
                          backend::AbstractSCFBackend=ThreadedCPUBackend())
    hartree_coefficients, hartree_energy = _hartree(density_value, kernel)
    xc_energy, xc_potential = _xc_energy_potential(
        getfield(getfield(kernel.basis, :_model), :_xc), density_value,
        kernel.core_density, kernel.reciprocal, kernel.volume,
        4getfield(kernel.basis, :_Ecut))
    xc_coefficients = fft(xc_potential) / length(xc_potential)
    local_coefficients = kernel.ionic_coefficients .+ hartree_coefficients .+ xc_coefficients
    external_coefficients === nothing || (local_coefficients .+= external_coefficients)
    band_values, orbitals = _solve_electrons(
        kernel, local_coefficients, number_bands;
        previous=previous_orbitals, backend)
    output_density = _density_from_orbitals(kernel, orbitals, noccupied)
    weights = getfield(kernel.basis, :_kweights)
    band_energy = sum(weights[k] * 2sum(band_values[k][1:noccupied])
                      for k in eachindex(weights))
    xc_double_count = kernel.volume * sum(density_value .* xc_potential) /
                      length(density_value)
    total = band_energy - hartree_energy + xc_energy - xc_double_count +
            kernel.ion_ion_energy
    (; total, output_density, band_values, orbitals, hartree_energy,
       xc_energy, xc_potential)
end
