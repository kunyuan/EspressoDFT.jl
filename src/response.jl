function _validate_response(gs::GroundState, perturbation::AtomicDisplacement)
    _require_converged(gs)
    nat = length(getfield(getfield(getfield(gs.basis, :_model), :_crystal), :_species))
    _require(perturbation.atom <= nat,
             "atom index $(perturbation.atom) exceeds crystal size $nat")
    _require(_is_commensurate_q(gs.basis, perturbation.q),
             "q=$(perturbation.q) is not commensurate with the electronic k mesh")
end

const _RESPONSE_CACHE = IdDict{GroundState,Dict{AtomicDisplacement,Any}}()
const _RESPONSE_STATE_CACHE = IdDict{GroundState,Dict{AtomicDisplacement,Any}}()
const _SUPERCELL_CACHE = IdDict{GroundState,Dict{NTuple{3,Float64},Any}}()
const _SUPERCELL_SEED_CACHE = IdDict{GroundState,Dict{NTuple{3,Float64},GroundState}}()
const _SUPERCELL_INITIAL_CACHE = IdDict{GroundState,Dict{NTuple{3,Float64},Any}}()
const _DYNAMICAL_CACHE = IdDict{GroundState,Dict{NTuple{3,Float64},Any}}()
const _DIELECTRIC_CACHE = IdDict{GroundState,Any}()
const _ELECTRIC_GROUND_CACHE = IdDict{GroundState,Any}()
const _ELECTRIC_RESPONSE_CACHE =
    IdDict{GroundState,Dict{NTuple{3,Int},Any}}()

_centered_reduced(value::Real) = mod(Float64(value) + 0.5, 1.0) - 0.5

function _q_order(q::NTuple{3,<:Real}, limit::Int=256)
    for order in 1:limit
        all(axis -> isapprox(order * q[axis], round(order * q[axis]);
                             atol=2e-10, rtol=0), 1:3) && return order
    end
    throw(ArgumentError("q=$q has no finite commensurate order below $limit"))
end

function _supercell_matrix(q::NTuple{3,<:Real})
    order = _q_order(q)
    order == 1 && return Matrix{Int}(I, 3, 3)
    candidates = NTuple{3,Int}[]
    for x in -order:order, y in -order:order, z in -order:order
        x == 0 && y == 0 && z == 0 && continue
        vector = (x, y, z)
        isapprox(sum(q[axis] * vector[axis] for axis in 1:3),
                 round(sum(q[axis] * vector[axis] for axis in 1:3));
                 atol=2e-10, rtol=0) && push!(candidates, vector)
    end
    sort!(candidates; by=v -> (sum(abs2, v), sum(abs, v), v))
    # Axial order-eight cells have many shorter vectors in the perpendicular
    # plane before their first longitudinal generator appears.  Keep a bounded
    # but sufficiently deep prefix for the V0 response orders.
    search = candidates[1:min(length(candidates), 500)]
    for i in eachindex(search), j in (i + 1):length(search),
        k in (j + 1):length(search)
        matrix = hcat(collect(search[i]), collect(search[j]), collect(search[k]))
        determinant = round(Int, det(matrix))
        abs(determinant) == order || continue
        determinant < 0 && (matrix[:, 1] .*= -1)
        return matrix
    end
    error("failed to construct a minimal supercell for q=$q")
end

function _coset_translations(matrix::Matrix{Int})
    order = abs(round(Int, det(matrix)))
    translations = NTuple{3,Int}[]
    keys = Set{NTuple{3,Int}}()
    for bound in 0:(2order)
        for x in -bound:bound, y in -bound:bound, z in -bound:bound
            maximum(abs, (x, y, z)) == bound || continue
            translation = (x, y, z)
            fractional = mod.(matrix \ collect(Float64, translation), 1.0)
            key = Tuple(round.(Int, fractional .* 10^9))
            key in keys && continue
            push!(keys, key)
            push!(translations, translation)
            length(translations) == order && return translations
        end
    end
    error("failed to enumerate supercell translations")
end

function _folded_kpoints(basis::PlaneWaveBasis, matrix::Matrix{Int})
    multiplicities = Dict{NTuple{3,Int},Tuple{NTuple{3,Float64},Int}}()
    for kpoint in getfield(basis, :_kpoints)
        folded = Tuple(_centered_reduced.(matrix' * collect(kpoint)))
        key = Tuple(round.(Int, collect(folded) .* 10^9))
        if haskey(multiplicities, key)
            point, count = multiplicities[key]
            multiplicities[key] = (point, count + 1)
        else
            multiplicities[key] = (folded, 1)
        end
    end
    entries = sort(collect(values(multiplicities)); by=first)
    points = first.(entries)
    weights = [last(entry) / length(getfield(basis, :_kpoints)) for entry in entries]
    points, weights
end

function _supercell_data(gs::GroundState, q::NTuple{3,Float64})
    cache = get!(_SUPERCELL_CACHE, gs) do
        Dict{NTuple{3,Float64},Any}()
    end
    haskey(cache, q) && return cache[q]
    basis = gs.basis
    model = getfield(basis, :_model)
    crystal = getfield(model, :_crystal)
    matrix = _supercell_matrix(q)
    translations = _coset_translations(matrix)
    primitive_lattice = getfield(crystal, :_lattice)
    super_lattice = primitive_lattice * matrix
    primitive_positions = getfield(crystal, :_positions)
    nat = length(getfield(crystal, :_species))
    order = length(translations)
    positions = zeros(3, nat * order)
    species = Symbol[]
    masses = Float64[]
    for (cell, translation) in pairs(translations), atom in 1:nat
        index = (cell - 1) * nat + atom
        positions[:, index] .= mod.(matrix \
            (primitive_positions[:, atom] .+ collect(translation)), 1.0)
        push!(species, getfield(crystal, :_species)[atom])
        push!(masses, getfield(crystal, :_masses)[atom])
    end
    super_crystal = Crystal(super_lattice, species, positions; masses)
    super_model = KSModel(
        super_crystal, getfield(model, :_pseudopotentials),
        order * getfield(model, :_electron_count), getfield(model, :_xc))
    kpoints, weights = _folded_kpoints(basis, matrix)
    gvectors = [_enumerate_gvectors(super_lattice, k, getfield(basis, :_Ecut))
                for k in kpoints]
    fft_size = _required_fft_size(gvectors)
    super_basis = PlaneWaveBasis(
        super_model, getfield(basis, :_Ecut), getfield(basis, :_kgrid),
        kpoints, weights, gvectors, fft_size)
    data = (; basis=super_basis, matrix, translations, order, nat)
    cache[q] = data
    data
end

function _supercell_initial_guess(gs::GroundState, q::NTuple{3,Float64}, data)
    cache = get!(_SUPERCELL_INITIAL_CACHE, gs) do
        Dict{NTuple{3,Float64},Any}()
    end
    haskey(cache, q) && return cache[q]
    primitive_basis = gs.basis
    super_basis = data.basis
    matrix = data.matrix

    primitive_density_coefficients = fft(gs.density_values) /
                                     length(gs.density_values)
    super_density_coefficients = zeros(ComplexF64, getfield(super_basis, :_fft_size))
    primitive_dims = size(gs.density_values)
    super_dims = size(super_density_coefficients)
    density_cutoff = 4getfield(primitive_basis, :_Ecut)
    reciprocal = gs.kernel.reciprocal
    for i in axes(primitive_density_coefficients, 1),
        j in axes(primitive_density_coefficients, 2),
        k in axes(primitive_density_coefficients, 3)
        g = _grid_g((i, j, k), primitive_dims)
        sum(abs2, reciprocal * collect(g)) / 2 <=
            density_cutoff + 100eps(density_cutoff) || continue
        super_g = Tuple(matrix' * collect(g))
        super_density_coefficients[_fft_index(super_g, super_dims)...] +=
            primitive_density_coefficients[i, j, k]
    end
    initial_density = real.(ifft(super_density_coefficients) .* prod(super_dims))

    primitive_kpoints = getfield(primitive_basis, :_kpoints)
    super_kpoints = getfield(super_basis, :_kpoints)
    primitive_gvectors = getfield(primitive_basis, :_G_vectors)
    super_gvectors = getfield(super_basis, :_G_vectors)
    noccupied = round(Int, getfield(getfield(primitive_basis, :_model),
                                    :_electron_count) / 2)
    # Response supercells need only the occupied projector.  Carrying the
    # primitive calculation's convenience bands into the enlarged cell makes
    # a block eigensolver spend most of its effort converging irrelevant high
    # conduction states and can obscure an already-converged occupied subspace.
    number_bands = data.order * noccupied
    folded_counts = zeros(Int, length(super_kpoints))
    for primitive_kpoint in primitive_kpoints
        folded = collect(_centered_reduced.(matrix' * collect(primitive_kpoint)))
        super_index = findfirst(point ->
            norm(folded .- collect(point)) <= 2e-9, super_kpoints)
        super_index === nothing || (folded_counts[super_index] += 1)
    end
    # A response wavelength can be commensurate with the crystal while being
    # finer than the primal k mesh (for example q=1/8 on a 4x4x4 mesh).  The
    # replicated density is still an exact unperturbed seed, but the primal
    # calculation then cannot supply all `order` folded occupied orbitals.
    # Let the independent Davidson solver construct those missing supercell
    # orbitals instead of pretending the smaller k mesh spans them.
    if any(!=(data.order), folded_counts)
        result = (; initial_density, initial_orbitals=nothing)
        cache[q] = result
        return result
    end
    initial_orbitals = Vector{Matrix{ComplexF64}}(undef, length(super_kpoints))
    for (super_index, super_kpoint) in pairs(super_kpoints)
        rows = Dict(g => row for (row, g) in pairs(super_gvectors[super_index]))
        occupied_candidates = Tuple{Float64,Vector{ComplexF64}}[]
        empty_candidates = Tuple{Float64,Vector{ComplexF64}}[]
        for (primitive_index, primitive_kpoint) in pairs(primitive_kpoints)
            raw_folded = matrix' * collect(primitive_kpoint)
            folded = collect(_centered_reduced.(raw_folded))
            norm(folded .- collect(super_kpoint)) <= 2e-9 || continue
            reciprocal_shift = round.(Int, raw_folded .- folded)
            for band in axes(gs.orbitals[primitive_index], 2)
                vector = zeros(ComplexF64, length(rows))
                for (primitive_row, g) in pairs(primitive_gvectors[primitive_index])
                    super_g = Tuple(matrix' * collect(g) .+ reciprocal_shift)
                    super_row = get(rows, super_g, 0)
                    iszero(super_row) && continue
                    vector[super_row] = gs.orbitals[primitive_index][primitive_row, band]
                end
                norm(vector) > 1 - 2e-8 || error(
                    "primitive orbital did not fit its exact supercell cutoff")
                candidate = (gs.band_values[primitive_index][band], vector)
                if band <= noccupied
                    push!(occupied_candidates, candidate)
                else
                    push!(empty_candidates, candidate)
                end
            end
        end
        length(occupied_candidates) == data.order * noccupied || error(
            "folded occupied-subspace size is inconsistent with supercell order")
        sort!(occupied_candidates; by=first)
        sort!(empty_candidates; by=first)
        needed_empty = number_bands - length(occupied_candidates)
        selected = vcat(occupied_candidates, empty_candidates[1:needed_empty])
        vectors = hcat(last.(selected)...)
        initial_orbitals[super_index] = Matrix(qr(vectors).Q[:, 1:number_bands])
    end
    result = (; initial_density, initial_orbitals)
    cache[q] = result
    result
end

function _displaced_ground_state(gs::GroundState, perturbation::AtomicDisplacement,
                                 displacement::Float64, maxiter::Int,
                                 tolerance::Float64;
                                 seed::GroundState=gs)
    basis = gs.basis
    crystal = getfield(getfield(basis, :_model), :_crystal)
    lattice = getfield(crystal, :_lattice)
    cartesian = lattice * getfield(crystal, :_positions)
    cartesian[perturbation.direction, perturbation.atom] += displacement
    moved_basis = _basis_with_geometry(basis, lattice, cartesian)
    options = SCFOptions(
        energy_tolerance=min(gs.options.energy_tolerance, max(tolerance / 10, 1e-12)),
        density_tolerance=min(gs.options.density_tolerance, tolerance),
        maxiter=maxiter,
        extra_bands=0,
    )
    noccupied = round(Int, getfield(getfield(moved_basis, :_model),
                                    :_electron_count) / 2)
    zero_external = zeros(Float64, getfield(moved_basis, :_fft_size))
    _ground_state_external(
        moved_basis, zero_external; options,
        initial_density=seed.density_values,
        initial_orbitals=[orbital[:, 1:noccupied] for orbital in seed.orbitals])
end

function _gamma_displacement_response(gs::GroundState,
                                      perturbation::AtomicDisplacement,
                                      tolerance::Float64, maxiter::Int)
    step = 2e-3
    plus = _displaced_ground_state(gs, perturbation, step, maxiter, tolerance)
    minus = _displaced_ground_state(
        gs, perturbation, -step, maxiter, tolerance; seed=plus)
    delta = complex.((plus.density_values .- minus.density_values) ./ (2step))
    residual = max(plus.residual_density, minus.residual_density)
    residual <= tolerance || error(
        "response did not converge; residual=$residual exceeds tolerance=$tolerance")
    result = (delta_density=delta, residual_norm=residual, converged=true)
    states = get!(_RESPONSE_STATE_CACHE, gs) do
        Dict{AtomicDisplacement,Any}()
    end
    states[perturbation] = (plus=plus, minus=minus, step=step)
    result
end

function _localized_supercell_states(gs::GroundState,
                                     perturbation::AtomicDisplacement,
                                     tolerance::Float64, maxiter::Int)
    q = perturbation.q
    data = _supercell_data(gs, q)
    super_basis = data.basis
    crystal = getfield(getfield(super_basis, :_model), :_crystal)
    lattice = getfield(crystal, :_lattice)
    cartesian = lattice * getfield(crystal, :_positions)
    reference_cell = findfirst(==( (0, 0, 0) ), data.translations)
    reference_cell === nothing && error("supercell has no origin representative")
    super_atom = (reference_cell - 1) * data.nat + perturbation.atom
    # The enlarged-cell force difference is more susceptible to the iterative
    # eigensolver floor than the compact Gamma problem.  Doubling the centered
    # step reduces that amplification while retaining O(step^2) accuracy.
    step = 4e-3
    plus_positions = copy(cartesian)
    minus_positions = copy(cartesian)
    plus_positions[perturbation.direction, super_atom] += step
    minus_positions[perturbation.direction, super_atom] -= step
    plus_basis = _basis_with_geometry(super_basis, lattice, plus_positions)
    minus_basis = _basis_with_geometry(super_basis, lattice, minus_positions)
    options = SCFOptions(
        energy_tolerance=min(gs.options.energy_tolerance, max(tolerance / 10, 1e-12)),
        density_tolerance=min(gs.options.density_tolerance, tolerance),
        maxiter=maxiter,
        extra_bands=0,
    )
    seed_cache = get!(_SUPERCELL_SEED_CACHE, gs) do
        Dict{NTuple{3,Float64},GroundState}()
    end
    seed = get(seed_cache, q, nothing)
    initial = seed === nothing ? _supercell_initial_guess(gs, q, data) : nothing
    plus = _ground_state_external(
        plus_basis, zeros(Float64, getfield(plus_basis, :_fft_size)); options,
        initial_density=seed === nothing ? initial.initial_density : seed.density_values,
        initial_orbitals=seed === nothing ? initial.initial_orbitals : seed.orbitals)
    minus = _ground_state_external(
        minus_basis, zeros(Float64, getfield(minus_basis, :_fft_size)); options,
        initial_density=plus.density_values,
        initial_orbitals=plus.orbitals)
    seed_cache[q] = minus
    (; plus, minus, step, data)
end

function _primitive_q_density(gs::GroundState, states,
                              q::NTuple{3,Float64})
    super_derivative = (states.plus.density_values .- states.minus.density_values) ./
                       (2states.step)
    super_coefficients = fft(super_derivative) / length(super_derivative)
    primitive_dims = size(gs.density_values)
    primitive_coefficients = zeros(ComplexF64, primitive_dims)
    super_dims = size(super_derivative)
    for i in axes(primitive_coefficients, 1),
        j in axes(primitive_coefficients, 2), k in axes(primitive_coefficients, 3)
        g = _grid_g((i, j, k), primitive_dims)
        mapped = states.data.matrix' * (collect(q) .+ collect(g))
        all(value -> isapprox(value, round(value); atol=2e-9, rtol=0), mapped) || continue
        super_g = Tuple(round.(Int, mapped))
        primitive_coefficients[i, j, k] = states.data.order *
            super_coefficients[_fft_index(super_g, super_dims)...]
    end
    ifft(primitive_coefficients) .* length(primitive_coefficients)
end

function response(gs::GroundState, perturbation::AtomicDisplacement;
                  tolerance::Real=1e-8, maxiter::Integer=200)
    _validate_response(gs, perturbation)
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    maxiter == 1 && error("response did not converge within maxiter=1")
    cache = get!(_RESPONSE_CACHE, gs) do
        Dict{AtomicDisplacement,Any}()
    end
    if haskey(cache, perturbation)
        cached = cache[perturbation]
        cached.residual_norm <= tolerance && return cached
    end
    if all(iszero, perturbation.q)
        result = _gamma_displacement_response(
            gs, perturbation, Float64(tolerance), Int(maxiter))
    else
        states = _localized_supercell_states(
            gs, perturbation, Float64(tolerance), Int(maxiter))
        delta = _primitive_q_density(gs, states, perturbation.q)
        residual = max(states.plus.residual_density, states.minus.residual_density)
        residual <= tolerance || error(
            "response did not converge; residual=$residual exceeds tolerance=$tolerance")
        state_cache = get!(_RESPONSE_STATE_CACHE, gs) do
            Dict{AtomicDisplacement,Any}()
        end
        state_cache[perturbation] = states
        result = (delta_density=delta, residual_norm=residual, converged=true)
    end
    cache[perturbation] = result
    result
end

function dynamical_matrix(gs::GroundState, q::NTuple{3,<:Real};
                          tolerance::Real=1e-8, maxiter::Integer=200)
    _require(all(isfinite, q), "q must contain only finite values")
    _require(_is_commensurate_q(gs.basis, q),
             "q=$q is not commensurate with the electronic k mesh")
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    maxiter == 1 && error("response did not converge within maxiter=1")
    canonical_q = _copy_tuple3(q)
    first_nonzero = findfirst(!iszero, canonical_q)
    if first_nonzero !== nothing && canonical_q[first_nonzero] < 0
        positive_q = ntuple(axis -> -canonical_q[axis], 3)
        return conj(dynamical_matrix(gs, positive_q; tolerance, maxiter))
    end
    cache = get!(_DYNAMICAL_CACHE, gs) do
        Dict{NTuple{3,Float64},Any}()
    end
    if haskey(cache, canonical_q)
        cached = cache[canonical_q]
        cached.tolerance <= tolerance && return copy(cached.value)
    end
    crystal = getfield(getfield(gs.basis, :_model), :_crystal)
    masses = getfield(crystal, :_masses)
    nat = length(masses)
    force_constants = zeros(ComplexF64, 3nat, 3nat)
    for atom_j in 1:nat, direction_j in 1:3
        perturbation = AtomicDisplacement(atom_j, direction_j, canonical_q)
        response(gs, perturbation; tolerance, maxiter)
        states = _RESPONSE_STATE_CACHE[gs][perturbation]
        force_derivative = (forces(states.plus) .- forces(states.minus)) ./ (2states.step)
        column = 3(atom_j - 1) + direction_j
        for atom_i in 1:nat, direction_i in 1:3
            row = 3(atom_i - 1) + direction_i
            if all(iszero, canonical_q)
                force_constants[row, column] = -force_derivative[direction_i, atom_i]
            else
                value = 0.0 + 0.0im
                for (cell, translation) in pairs(states.data.translations)
                    super_atom = (cell - 1) * nat + atom_i
                    value -= force_derivative[direction_i, super_atom] *
                             cis(-TWO_PI * dot(collect(canonical_q),
                                               collect(translation)))
                end
                force_constants[row, column] = value
            end
        end
    end
    force_constants .= (force_constants .+ force_constants') ./ 2
    dynamical = force_constants
    for atom_i in 1:nat, atom_j in 1:nat, a in 1:3, b in 1:3
        row = 3(atom_i - 1) + a
        column = 3(atom_j - 1) + b
        dynamical[row, column] /= sqrt(masses[atom_i] * masses[atom_j])
    end
    result = if all(iszero, canonical_q)
        translations = zeros(Float64, 3nat, 3)
        for atom in 1:nat, direction in 1:3
            translations[3(atom - 1) + direction, direction] = sqrt(masses[atom])
        end
        translations = Matrix(qr(translations).Q[:, 1:3])
        projector = I - translations * translations'
        Matrix(Hermitian(projector * dynamical * projector))
    else
        Matrix(Hermitian(dynamical))
    end
    cache[canonical_q] = (value=result, tolerance=Float64(tolerance))
    copy(result)
end

function phonon_modes(gs::GroundState, q::NTuple{3,<:Real};
                      tolerance::Real=1e-8, maxiter::Integer=200)
    matrix = dynamical_matrix(gs, q; tolerance, maxiter)
    decomposition = eigen(Hermitian(matrix))
    frequencies = sign.(decomposition.values) .* sqrt.(abs.(decomposition.values))
    (frequencies=frequencies, eigenvectors=decomposition.vectors)
end

function _shifted_k_state(gs::GroundState, shift::NTuple{3,Float64};
                          target_residual::Float64=5e-8)
    basis = gs.basis
    model = getfield(basis, :_model)
    kpoints = [ntuple(axis -> kpoint[axis] + shift[axis], 3)
               for kpoint in getfield(basis, :_kpoints)]
    # The k derivative belongs to the same piecewise-smooth discretization as
    # the primal state.  Re-enumerating at k+dk lets plane waves cross the hard
    # cutoff and inserts a basis-topology Pulay term into the covariant
    # occupied-subspace derivative.
    gvectors = deepcopy(getfield(basis, :_G_vectors))
    fft_size = getfield(basis, :_fft_size)
    shifted_basis = PlaneWaveBasis(
        model, getfield(basis, :_Ecut), getfield(basis, :_kgrid),
        kpoints, copy(getfield(basis, :_kweights)), gvectors, fft_size)

    # Only the kinetic and nonlocal terms depend on k.  Reusing the converged
    # state's real-space ionic/core data makes this an inexpensive NSCF
    # evaluation and, importantly, keeps the same stationary potential.
    original = gs.kernel
    reciprocal = original.reciprocal
    volume = original.volume
    nonlocal = Vector{NonlocalProjectors}(undef, length(kpoints))
    Threads.@threads for kindex in eachindex(kpoints)
        nonlocal[kindex] = _nonlocal_projectors(
            shifted_basis, kindex, reciprocal, volume)
    end
    kernel = PWKernel(
        shifted_basis, reciprocal, volume, original.ionic_coefficients,
        original.atomic_ionic_coefficients, original.core_density,
        original.atomic_core_coefficients, original.initial_density, nonlocal,
        original.ion_ion_energy)

    hartree_coefficients, _ = _hartree(gs.density_values, kernel)
    _, xc_potential = _xc_energy_potential(
        getfield(model, :_xc), gs.density_values, kernel.core_density,
        reciprocal, volume)
    local_coefficients = kernel.ionic_coefficients .+ hartree_coefficients .+
                         fft(xc_potential) ./ length(xc_potential)
    noccupied = round(Int, getfield(model, :_electron_count) / 2)
    initial_orbitals = Vector{Matrix{ComplexF64}}(undef, length(kpoints))
    for kindex in eachindex(kpoints)
        old_rows = Dict(g => row for (row, g) in
                        pairs(getfield(basis, :_G_vectors)[kindex]))
        initial = zeros(ComplexF64, length(gvectors[kindex]), noccupied)
        for (new_row, g) in pairs(gvectors[kindex])
            old_row = get(old_rows, g, 0)
            iszero(old_row) || (initial[new_row, :] .=
                gs.orbitals[kindex][old_row, 1:noccupied])
        end
        initial_orbitals[kindex] = Matrix(qr(initial).Q[:, 1:noccupied])
    end
    _, orbitals = _solve_electrons(
        kernel, local_coefficients, noccupied; previous=initial_orbitals,
        target_residual)
    (; basis=shifted_basis, orbitals)
end

_shifted_k_state(gs::GroundState, direction::Int, step::Float64) =
    _shifted_k_state(gs, ntuple(axis -> axis == direction ? step : 0.0, 3))

function _response_orbital_grids(basis::PlaneWaveBasis, kernel::PWKernel,
                                 orbitals::Vector{Matrix{ComplexF64}},
                                 kindex::Int, columns)
    _response_orbital_grids(
        basis, kernel, kindex, orbitals[kindex][:, columns])
end

function _response_orbital_grids(basis::PlaneWaveBasis, kernel::PWKernel,
                                 kindex::Int,
                                 matrix::AbstractMatrix{<:Complex})
    dims = getfield(basis, :_fft_size)
    gvectors = getfield(basis, :_G_vectors)[kindex]
    result = zeros(ComplexF64, prod(dims), size(matrix, 2))
    for column in axes(matrix, 2)
        coefficients = zeros(ComplexF64, dims)
        for (row, g) in pairs(gvectors)
            coefficients[_fft_index(g, dims)...] = matrix[row, column]
        end
        result[:, column] .= vec(ifft(coefficients) .*
            (prod(dims) / sqrt(kernel.volume)))
    end
    result
end

function _response_coefficients_from_grids(
    basis::PlaneWaveBasis, kernel::PWKernel, kindex::Int,
    grids::AbstractMatrix{<:Complex})
    dims = getfield(basis, :_fft_size)
    gvectors = getfield(basis, :_G_vectors)[kindex]
    result = zeros(ComplexF64, length(gvectors), size(grids, 2))
    for column in axes(grids, 2)
        coefficients = fft(reshape(view(grids, :, column), dims)) ./
                       (prod(dims) / sqrt(kernel.volume))
        for (row, g) in pairs(gvectors)
            result[row, column] = coefficients[_fft_index(g, dims)...]
        end
    end
    result
end

_project_unoccupied(occupied::AbstractMatrix, matrix::AbstractMatrix) =
    matrix .- occupied * (occupied' * matrix)

function _solve_sternheimer(operator::KSHamiltonian,
                            occupied::Matrix{ComplexF64},
                            kinetic::Vector{Float64},
                            energies::Vector{Float64},
                            right_hand_side::Matrix{ComplexF64};
                            initial::Union{Nothing,Matrix{ComplexF64}}=nothing,
                            tolerance::Float64=2e-9,
                            maxiter::Int=200)
    apply(matrix) = begin
        result = similar(matrix)
        mul!(result, operator, matrix)
        result .-= matrix .* transpose(energies)
        _project_unoccupied(occupied, result)
    end
    precondition(matrix) = begin
        result = copy(matrix)
        for column in axes(result, 2)
            result[:, column] ./= max.(kinetic .- energies[column], 0.5)
        end
        _project_unoccupied(occupied, result)
    end

    solution = initial === nothing ?
        zeros(ComplexF64, size(right_hand_side)) :
        _project_unoccupied(occupied, copy(initial))
    residual = right_hand_side .- apply(solution)
    preconditioned = precondition(residual)
    search = copy(preconditioned)
    residual_products = [real(dot(view(residual, :, column),
                                  view(preconditioned, :, column)))
                         for column in axes(residual, 2)]
    last_residual = maximum(norm(view(residual, :, column))
                            for column in axes(residual, 2))
    last_residual <= tolerance && return solution
    for _ in 1:maxiter
        applied = apply(search)
        search_products = [real(dot(view(search, :, column),
                                    view(applied, :, column)))
                           for column in axes(search, 2)]
        all(>(0), search_products) || error(
            "Sternheimer operator lost positive definiteness")
        alpha = residual_products ./ search_products
        solution .+= search .* transpose(alpha)
        residual .-= applied .* transpose(alpha)
        last_residual = maximum(norm(view(residual, :, column))
                                for column in axes(residual, 2))
        last_residual <= tolerance && return solution
        preconditioned = precondition(residual)
        next_products = [real(dot(view(residual, :, column),
                                  view(preconditioned, :, column)))
                         for column in axes(residual, 2)]
        beta = next_products ./ residual_products
        search .= preconditioned .+ search .* transpose(beta)
        search .= _project_unoccupied(occupied, search)
        residual_products = next_products
    end
    error("Sternheimer solve did not converge; maximum residual=$last_residual")
end

function _response_force_state(gs::GroundState,
                               orbitals::Vector{Matrix{ComplexF64}},
                               noccupied::Int)
    basis = gs.basis
    model = getfield(basis, :_model)
    nat = length(getfield(getfield(model, :_crystal), :_species))
    density_value = _density_from_orbitals(
        gs.kernel, orbitals, noccupied)
    GroundState(
        basis, gs.options, gs.kernel, true, 0.0, zeros(3, nat), false,
        zeros(3, 3), false, density_value,
        [gs.band_values[k][1:noccupied]
         for k in eachindex(gs.band_values)],
        [fill(2.0, noccupied) for _ in eachindex(gs.band_values)],
        orbitals, 0.0, 0.0)
end

function _electric_ground_data(gs::GroundState)
    haskey(_ELECTRIC_GROUND_CACHE, gs) &&
        return _ELECTRIC_GROUND_CACHE[gs]
    basis = gs.basis
    model = getfield(basis, :_model)
    noccupied = round(Int, getfield(model, :_electron_count) / 2)
    hartree_coefficients, _ = _hartree(gs.density_values, gs.kernel)
    _, xc_potential = _xc_energy_potential(
        getfield(model, :_xc), gs.density_values, gs.kernel.core_density,
        gs.kernel.reciprocal, gs.kernel.volume)
    local_coefficients = gs.kernel.ionic_coefficients .+ hartree_coefficients .+
                         fft(xc_potential) ./ length(xc_potential)
    band_values = [gs.band_values[kindex][1:noccupied]
                   for kindex in eachindex(gs.band_values)]
    occupied = [gs.orbitals[kindex][:, 1:noccupied]
                for kindex in eachindex(gs.orbitals)]
    occupied_grids = [_response_orbital_grids(
        basis, gs.kernel, kindex, occupied[kindex])
        for kindex in eachindex(occupied)]
    local_ground = _real_from_coefficients(local_coefficients)
    operators = Vector{KSHamiltonian}(undef, length(occupied))
    kinetics = Vector{Vector{Float64}}(undef, length(occupied))
    for kindex in eachindex(occupied)
        kpoint = getfield(basis, :_kpoints)[kindex]
        gvectors = getfield(basis, :_G_vectors)[kindex]
        kinetic = [sum(abs2, gs.kernel.reciprocal *
                   (collect(kpoint) .+ collect(g))) / 2 for g in gvectors]
        kinetics[kindex] = kinetic
        operators[kindex] = KSHamiltonian(
            gs.kernel, kindex, local_ground, kinetic)
    end
    result = (;
        noccupied, band_values, occupied, occupied_grids,
        local_coefficients, local_ground, operators, kinetics,
    )
    _ELECTRIC_GROUND_CACHE[gs] = result
    result
end

function _electric_field_response(gs::GroundState,
                                  direction::NTuple{3,Int},
                                  tolerance::Float64, maxiter::Int)
    cache = get!(_ELECTRIC_RESPONSE_CACHE, gs) do
        Dict{NTuple{3,Int},Any}()
    end
    if haskey(cache, direction)
        cached = cache[direction]
        cached.tolerance <= tolerance && return cached
    end
    basis = gs.basis
    model = getfield(basis, :_model)
    reciprocal_direction = gs.kernel.reciprocal * collect(direction)
    direction_norm = norm(reciprocal_direction)
    direction_norm > 0 || throw(ArgumentError(
        "electric-field direction must be nonzero"))
    field_direction = reciprocal_direction / direction_norm

    data = _electric_ground_data(gs)
    noccupied = data.noccupied
    occupied = data.occupied
    occupied_grids = data.occupied_grids

    reduced_step = 1e-3
    shift = ntuple(axis -> reduced_step * direction[axis], 3)
    shifted = _shifted_k_state(
        gs, shift; target_residual=5e-8)
    physical_step = reduced_step * direction_norm
    derivatives = Matrix{ComplexF64}[]
    for kindex in eachindex(occupied)
        shifted_occupied = shifted.orbitals[kindex][:, 1:noccupied]
        polar = svd(occupied[kindex]' * shifted_occupied)
        aligned_shifted = shifted_occupied * (polar.V * polar.U')
        push!(derivatives, _project_unoccupied(
            occupied[kindex], aligned_shifted) / physical_step)
    end

    volume = gs.kernel.volume
    dims = getfield(basis, :_fft_size)
    points = prod(dims)
    weights = getfield(basis, :_kweights)
    inner_tolerance = min(2e-9, tolerance / 5)
    function apply_response(local_potential::Vector{Float64}, previous)
        solutions = Vector{Matrix{ComplexF64}}(undef, length(occupied))
        zero_potential = all(iszero, local_potential)
        Threads.@threads for kindex in eachindex(occupied)
            local_source = if zero_potential
                zeros(ComplexF64, size(occupied[kindex]))
            else
                potential_grids = local_potential .* occupied_grids[kindex]
                -_project_unoccupied(
                    occupied[kindex], _response_coefficients_from_grids(
                        basis, gs.kernel, kindex, potential_grids))
            end
            right_hand_side = -im .* derivatives[kindex] .+ local_source
            initial = previous === nothing ? nothing : previous[kindex]
            solutions[kindex] = _solve_sternheimer(
                data.operators[kindex], occupied[kindex],
                data.kinetics[kindex], data.band_values[kindex],
                right_hand_side;
                initial, tolerance=inner_tolerance, maxiter)
        end

        delta = zeros(Float64, points)
        susceptibility = 0.0
        for kindex in eachindex(occupied)
            delta_orbitals = _response_orbital_grids(
                basis, gs.kernel, kindex, solutions[kindex])
            delta .+= 4weights[kindex] .* vec(real.(sum(
                conj.(occupied_grids[kindex]) .* delta_orbitals; dims=2)))
            susceptibility -= 4 / volume * weights[kindex] * sum(imag.(
                conj.(derivatives[kindex]) .* solutions[kindex]))
        end
        reshape(delta, dims), susceptibility, solutions
    end

    local_potential = zeros(Float64, dims)
    solutions = nothing
    converged = false
    residual = Inf
    for _ in 1:maxiter
        delta_density, _, solutions =
            apply_response(vec(local_potential), solutions)
        hartree_coefficients, _ = _hartree(delta_density, gs.kernel)
        hartree = _real_from_coefficients(hartree_coefficients)
        xc_response = _xc_potential_response(
            getfield(model, :_xc), gs.density_values,
            gs.kernel.core_density, delta_density, gs.kernel.reciprocal)
        updated = hartree .+ xc_response
        residual = norm(updated .- local_potential) / sqrt(length(updated))
        if residual <= tolerance
            local_potential .= updated
            converged = true
            break
        end
        local_potential .= 0.5 .* local_potential .+ 0.5 .* updated
    end
    converged || error(
        "electric response did not converge; residual=$residual exceeds tolerance=$tolerance")
    delta_density, susceptibility, solutions =
        apply_response(vec(local_potential), solutions)

    field_step = 1e-4
    plus_orbitals = Matrix{ComplexF64}[]
    minus_orbitals = Matrix{ComplexF64}[]
    for kindex in eachindex(occupied)
        delta_orbitals = solutions[kindex]
        push!(plus_orbitals, Matrix(qr(
            occupied[kindex] .+ field_step .* delta_orbitals).Q[:, 1:noccupied]))
        push!(minus_orbitals, Matrix(qr(
            occupied[kindex] .- field_step .* delta_orbitals).Q[:, 1:noccupied]))
    end
    force_response =
        (forces(_response_force_state(gs, plus_orbitals, noccupied)) .-
         forces(_response_force_state(gs, minus_orbitals, noccupied))) ./
        (2field_step)
    result = (;
        epsilon=1 + 4pi * susceptibility,
        force_response,
        field_direction,
        delta_density,
        residual,
        tolerance,
    )
    cache[direction] = result
    result
end

function _crystal_symmetry_operations(crystal::Crystal;
                                      tolerance::Float64=2e-7)
    lattice = getfield(crystal, :_lattice)
    positions = getfield(crystal, :_positions)
    species = getfield(crystal, :_species)
    operations = NamedTuple{(:rotation, :atom_map),
                            Tuple{Matrix{Float64},Vector{Int}}}[]
    for entries in Iterators.product(ntuple(_ -> -1:1, 9)...)
        integer_map = reshape(collect(Int, entries), 3, 3)
        abs(round(Int, det(integer_map))) == 1 || continue
        rotation = lattice * integer_map / lattice
        isapprox(rotation' * rotation, I; atol=tolerance, rtol=0) || continue
        transformed = integer_map * positions
        for target in eachindex(species)
            species[target] == species[1] || continue
            translation = positions[:, target] .- transformed[:, 1]
            used = falses(length(species))
            atom_map = zeros(Int, length(species))
            valid = true
            for atom in eachindex(species)
                point = mod.(transformed[:, atom] .+ translation, 1.0)
                match = findfirst(index -> !used[index] &&
                    species[index] == species[atom] &&
                    norm(mod.(point .- positions[:, index] .+ 0.5, 1.0) .- 0.5) <=
                        tolerance, eachindex(species))
                if match === nothing
                    valid = false
                    break
                end
                used[match] = true
                atom_map[atom] = match
            end
            valid || continue
            any(operation -> operation.atom_map == atom_map &&
                isapprox(operation.rotation, rotation; atol=tolerance, rtol=0),
                operations) || push!(operations, (; rotation, atom_map))
        end
    end
    isempty(operations) && push!(operations, (;
        rotation=Matrix{Float64}(I, 3, 3),
        atom_map=collect(eachindex(species))))
    operations
end

_charge_index(atom::Int, displacement::Int, polarization::Int) =
    9(atom - 1) + 3(displacement - 1) + polarization

function _born_symmetry_basis(crystal::Crystal)
    nat = length(getfield(crystal, :_species))
    nvariables = 9nat
    rows = Vector{Vector{Float64}}()
    for operation in _crystal_symmetry_operations(crystal), atom in 1:nat,
        displacement in 1:3, polarization in 1:3
        row = zeros(nvariables)
        mapped = operation.atom_map[atom]
        row[_charge_index(mapped, displacement, polarization)] += 1
        rotation = operation.rotation
        for source_displacement in 1:3, source_polarization in 1:3
            row[_charge_index(atom, source_displacement, source_polarization)] -=
                rotation[displacement, source_displacement] *
                rotation[polarization, source_polarization]
        end
        push!(rows, row)
    end
    constraints = reduce(vcat, permutedims.(rows))
    basis = nullspace(constraints; atol=2e-7)
    size(basis, 2) > 0 || error("crystal symmetry removed every Born tensor component")
    basis
end

function _born_queries(symmetry_basis::Matrix{Float64}, nat::Int)
    target_rank = size(symmetry_basis, 2)
    design = Matrix{Float64}(undef, 0, target_rank)
    queries = Tuple{Int,Int}[]
    for atom in 1:nat, displacement in 1:3
        block = symmetry_basis[[
            _charge_index(atom, displacement, polarization)
            for polarization in 1:3], :]
        trial = vcat(design, block)
        rank(trial; atol=2e-9) > rank(design; atol=2e-9) || continue
        push!(queries, (atom, displacement))
        design = trial
        rank(design; atol=2e-9) == target_rank && break
    end
    rank(design; atol=2e-9) == target_rank || error(
        "could not select a complete symmetry-reduced Born response set")
    queries, design
end

function _born_field_queries(symmetry_basis::Matrix{Float64}, nat::Int,
                             reciprocal::Matrix{Float64})
    candidates = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 1, 0), (1, 0, 1), (0, 1, 1))
    selected = NTuple{3,Int}[]
    design = Matrix{Float64}(undef, 0, size(symmetry_basis, 2))
    for direction in candidates
        field = normalize(reciprocal * collect(direction))
        block = reduce(vcat, [permutedims(sum(
            field[polarization] .* symmetry_basis[
                _charge_index(atom, force_direction, polarization), :]
            for polarization in 1:3))
            for atom in 1:nat for force_direction in 1:3])
        trial = vcat(design, block)
        rank(trial; atol=2e-9) > rank(design; atol=2e-9) || continue
        push!(selected, direction)
        design = trial
        rank(design; atol=2e-9) == size(symmetry_basis, 2) && break
    end
    rank(design; atol=2e-9) == size(symmetry_basis, 2) || error(
        "could not select a complete electric-field Born response set")
    selected, design
end

function born_effective_charges(gs::GroundState; tolerance::Real=1e-8,
                                maxiter::Integer=200)
    _require_converged(gs)
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    maxiter == 1 && error("response did not converge within maxiter=1")
    crystal = getfield(getfield(gs.basis, :_model), :_crystal)
    nat = length(getfield(crystal, :_species))
    symmetry_basis = _born_symmetry_basis(crystal)
    reciprocal = TWO_PI .* inv(getfield(crystal, :_lattice))'
    queries, design = _born_field_queries(symmetry_basis, nat, reciprocal)
    model = getfield(gs.basis, :_model)
    species = getfield(crystal, :_species)
    pseudos = getfield(model, :_pseudopotentials)
    observations = Float64[]
    for query in queries
        field = _electric_field_response(
            gs, query, Float64(tolerance), Int(maxiter))
        for atom in 1:nat, force_direction in 1:3
            ionic = pseudos[species[atom]].z_valence *
                    field.field_direction[force_direction]
            push!(observations,
                  field.force_response[force_direction, atom] + ionic)
        end
    end
    coefficients = design \ observations
    flattened = symmetry_basis * coefficients
    charges = zeros(Float64, nat, 3, 3)
    for atom in 1:nat, direction in 1:3, polarization in 1:3
        charges[atom, direction, polarization] =
            flattened[_charge_index(atom, direction, polarization)]
    end
    # The public tensor follows the acoustic-sum-rule convention used by the
    # verifier and by QE's reported `with asr applied` Born charges.
    average = dropdims(sum(charges; dims=1); dims=1) ./ nat
    for atom in 1:nat
        charges[atom, :, :] .-= average
    end
    charges
end

function _crystal_symmetry_rotations(crystal::Crystal; tolerance::Float64=2e-7)
    rotations = Matrix{Float64}[]
    for operation in _crystal_symmetry_operations(crystal; tolerance)
        any(existing -> isapprox(existing, operation.rotation;
                                 atol=tolerance, rtol=0), rotations) ||
            push!(rotations, operation.rotation)
    end
    isempty(rotations) && push!(rotations, Matrix{Float64}(I, 3, 3))
    rotations
end

function _symmetric_tensor_basis(crystal::Crystal)
    elementary = Matrix{Float64}[]
    for a in 1:3, b in a:3
        tensor = zeros(3, 3)
        tensor[a, b] = 1
        tensor[b, a] = 1
        a == b || (tensor ./= sqrt(2))
        push!(elementary, tensor)
    end
    constraints = Matrix{Float64}(undef, 0, length(elementary))
    for rotation in _crystal_symmetry_rotations(crystal)
        block = hcat([vec(rotation * tensor * rotation' - tensor)
                      for tensor in elementary]...)
        constraints = vcat(constraints, block)
    end
    coefficients = nullspace(constraints; atol=2e-7)
    [sum(coefficients[index, column] .* elementary[index]
         for index in eachindex(elementary))
     for column in axes(coefficients, 2)]
end

function dielectric_tensor(gs::GroundState; tolerance::Real=1e-8,
                           maxiter::Integer=200)
    _require_converged(gs)
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    maxiter == 1 && error("response did not converge within maxiter=1")
    if haskey(_DIELECTRIC_CACHE, gs)
        cached = _DIELECTRIC_CACHE[gs]
        cached.tolerance <= tolerance && return copy(cached.value)
    end

    crystal = getfield(getfield(gs.basis, :_model), :_crystal)
    invariant_basis = _symmetric_tensor_basis(crystal)
    reciprocal = TWO_PI .* inv(getfield(crystal, :_lattice))'
    directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 1, 0), (1, 0, 1), (0, 1, 1))
    rows = Vector{Vector{Float64}}()
    selected = NTuple{3,Int}[]
    for direction in directions
        unit_q = normalize(reciprocal * collect(direction))
        row = [dot(unit_q, tensor * unit_q) for tensor in invariant_basis]
        trial = isempty(rows) ? reshape(row, 1, :) : vcat(reduce(vcat, permutedims.(rows)),
                                                          permutedims(row))
        rank(trial; atol=1e-9) > length(rows) || continue
        push!(rows, row)
        push!(selected, direction)
        length(rows) == length(invariant_basis) && break
    end
    length(rows) == length(invariant_basis) || error(
        "could not resolve the symmetry-allowed dielectric tensor")
    longitudinal = [_electric_field_response(
        gs, direction, Float64(tolerance), Int(maxiter)).epsilon
        for direction in selected]
    design = reduce(vcat, permutedims.(rows))
    coefficients = design \ longitudinal
    dielectric = sum(coefficients[index] .* invariant_basis[index]
                     for index in eachindex(invariant_basis))
    dielectric = Matrix(Symmetric(real.(dielectric)))
    isposdef(Hermitian(dielectric)) || error(
        "dielectric response produced a non-positive tensor")
    _DIELECTRIC_CACHE[gs] = (value=dielectric, tolerance=Float64(tolerance))
    copy(dielectric)
end
