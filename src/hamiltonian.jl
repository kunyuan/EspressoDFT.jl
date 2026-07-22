struct NonlocalProjectors
    factors::Matrix{ComplexF64}
    coupling::Matrix{Float64}
    atoms::Vector{Int}
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
            [_projector_transform(upf, projector, q) for q in magnitudes]
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
        g = _grid_g((i, j, k), dims)
        all(iszero, g) && continue
        q2 = sum(abs2, kernel.reciprocal * collect(g))
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
end

Base.size(operator::KSHamiltonian) =
    (length(operator.kinetic), length(operator.kinetic))

function LinearAlgebra.mul!(output::AbstractVecOrMat, operator::KSHamiltonian,
                            input::AbstractVecOrMat)
    input_matrix = input isa AbstractVector ? reshape(input, :, 1) : input
    output_matrix = output isa AbstractVector ? reshape(output, :, 1) : output
    size(input_matrix, 1) == length(operator.kinetic) || throw(DimensionMismatch())
    size(output_matrix) == size(input_matrix) || throw(DimensionMismatch())
    dims = size(operator.local_potential)
    number_vectors = size(input_matrix, 2)
    grids = zeros(ComplexF64, dims..., number_vectors)
    gvectors = getfield(operator.kernel.basis, :_G_vectors)[operator.kindex]
    for vector in 1:number_vectors, (row, g) in pairs(gvectors)
        grids[_fft_index(g, dims)..., vector] = input_matrix[row, vector]
    end
    real_space = ifft(grids, (1, 2, 3))
    real_space .*= reshape(operator.local_potential, dims..., 1)
    local_grids = fft(real_space, (1, 2, 3))
    for vector in 1:number_vectors, (row, g) in pairs(gvectors)
        output_matrix[row, vector] = operator.kinetic[row] * input_matrix[row, vector] +
                                     local_grids[_fft_index(g, dims)..., vector]
    end
    output_matrix .+= _apply_nonlocal(
        operator.kernel.nonlocal_projectors[operator.kindex], input_matrix)
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
                               target_residual::Float64=5e-8)
    basis = kernel.basis
    kpoint = getfield(basis, :_kpoints)[kindex]
    gvectors = getfield(basis, :_G_vectors)[kindex]
    kinetic = [sum(abs2, kernel.reciprocal *
                   (collect(kpoint) .+ collect(g))) / 2 for g in gvectors]
    operator = KSHamiltonian(kernel, kindex,
                             _real_from_coefficients(local_coefficients), kinetic)
    subspace = _initial_subspace(kinetic, number_bands, previous)
    applied = similar(subspace)
    mul!(applied, operator, subspace)
    # The default wavefunction residual sits below the 1e-8 density-response
    # gate.  Response source construction requests a tighter value explicitly
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

function _solve_electrons(kernel::PWKernel, local_coefficients::Array{ComplexF64,3},
                          number_bands::Int;
                          previous::Union{Nothing,Vector{Matrix{ComplexF64}}}=nothing,
                          target_residual::Float64=5e-8)
    basis = kernel.basis
    nk = length(getfield(basis, :_kpoints))
    values = Vector{Vector{Float64}}(undef, nk)
    vectors = Vector{Matrix{ComplexF64}}(undef, nk)
    Threads.@threads for kindex in 1:nk
        dimension = length(getfield(basis, :_G_vectors)[kindex])
        number_bands <= dimension || error(
            "plane-wave basis is smaller than requested band count")
        # LAPACK's selected-spectrum path is both faster and more robust for
        # medium cells.  Above this boundary the dense matrix footprint grows
        # rapidly, so the independent matrix-free block solver takes over.
        if dimension <= 1800
            matrix = _hamiltonian(kernel, kindex, local_coefficients)
            decomposition = eigen(Hermitian(matrix), 1:number_bands)
            values[kindex] = decomposition.values
            vectors[kindex] = decomposition.vectors
        else
            prior = previous === nothing ? nothing : previous[kindex]
            values[kindex], vectors[kindex] = _iterative_eigensolve(
                kernel, kindex, local_coefficients, number_bands, prior;
                target_residual)
        end
    end
    values, vectors
end

function _density_from_orbitals(kernel::PWKernel,
                                orbitals::Vector{Matrix{ComplexF64}},
                                noccupied::Int)
    basis = kernel.basis
    dims = getfield(basis, :_fft_size)
    density = zeros(Float64, dims)
    weights = getfield(basis, :_kweights)
    for kindex in eachindex(orbitals)
        gvectors = getfield(basis, :_G_vectors)[kindex]
        for band in 1:noccupied
            coefficients = zeros(ComplexF64, dims)
            for (row, g) in pairs(gvectors)
                coefficients[_fft_index(g, dims)...] = orbitals[kindex][row, band]
            end
            periodic_orbital = ifft(coefficients) .* (prod(dims) / sqrt(kernel.volume))
            density .+= 2weights[kindex] .* abs2.(periodic_orbital)
        end
    end
    density
end

function _mix_density(density::Array{Float64,3}, output::Array{Float64,3},
                      kernel::PWKernel; mixing::Float64=0.7,
                      screening::Float64=1.0)
    residual_coefficients = fft(output .- density) / length(density)
    dims = size(density)
    for i in axes(density, 1), j in axes(density, 2), k in axes(density, 3)
        g = _grid_g((i, j, k), dims)
        if all(iszero, g)
            residual_coefficients[i, j, k] = 0
        else
            q2 = sum(abs2, kernel.reciprocal * collect(g))
            residual_coefficients[i, j, k] *= mixing * q2 / (q2 + screening)
        end
    end
    mixed = density .+ real.(ifft(residual_coefficients) .* length(density))
    target_average = getfield(getfield(kernel.basis, :_model), :_electron_count) /
                     kernel.volume
    mixed .+= target_average - sum(mixed) / length(mixed)
    mixed
end

function _electronic_step(kernel::PWKernel, density_value::Array{Float64,3},
                          number_bands::Int, noccupied::Int;
                          previous_orbitals::Union{Nothing,Vector{Matrix{ComplexF64}}}=nothing,
                          external_coefficients::Union{Nothing,Array{ComplexF64,3}}=nothing)
    hartree_coefficients, hartree_energy = _hartree(density_value, kernel)
    xc_energy, xc_potential = _xc_energy_potential(
        getfield(getfield(kernel.basis, :_model), :_xc), density_value,
        kernel.core_density, kernel.reciprocal, kernel.volume)
    xc_coefficients = fft(xc_potential) / length(xc_potential)
    local_coefficients = kernel.ionic_coefficients .+ hartree_coefficients .+ xc_coefficients
    external_coefficients === nothing || (local_coefficients .+= external_coefficients)
    band_values, orbitals = _solve_electrons(
        kernel, local_coefficients, number_bands; previous=previous_orbitals)
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
