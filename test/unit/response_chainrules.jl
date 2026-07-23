@testset "commensurate q helpers construct the minimal supercell quotient" begin
    @test EspressoDFT._centered_reduced(0.75) == -0.25
    @test EspressoDFT._q_order((0.25, 0.25, 0.25)) == 4
    matrix = EspressoDFT._supercell_matrix((0.25, 0.25, 0.25))
    @test abs(round(Int, det(matrix))) == 4
    @test all(isinteger, matrix' * collect((0.25, 0.25, 0.25)))
    translations = EspressoDFT._coset_translations(matrix)
    @test length(translations) == 4
    @test length(unique(translations)) == 4
    @test_throws ArgumentError EspressoDFT._q_order((sqrt(2), 0.0, 0.0), 16)
end

@testset "occupied projection and Sternheimer solve satisfy their equations" begin
    basis = SYNTHETIC_BASIS
    original = SYNTHETIC_KERNEL
    dimension = length(basis.G_vectors[1])
    empty_projectors = EspressoDFT.NonlocalProjectors(
        zeros(ComplexF64, dimension, 0),
        zeros(0, 0),
        Int[],
    )
    kernel = EspressoDFT.PWKernel(
        basis,
        original.reciprocal,
        original.volume,
        original.ionic_coefficients,
        original.atomic_ionic_coefficients,
        original.core_density,
        original.atomic_core_coefficients,
        original.initial_density,
        [empty_projectors],
        original.ion_ion_energy,
    )
    kinetic = Float64.(0:(dimension - 1))
    operator = EspressoDFT.KSHamiltonian(
        kernel, 1, zeros(basis.fft_size), kinetic)
    occupied = zeros(ComplexF64, dimension, 1)
    occupied[1, 1] = 1
    trial = reshape(ComplexF64[cis(0.2index) for index in 1:dimension], :, 1)
    projected = EspressoDFT._project_unoccupied(occupied, trial)
    @test occupied' * projected ≈ zeros(ComplexF64, 1, 1) atol=1e-14

    rhs = zeros(ComplexF64, dimension, 1)
    rhs[2, 1] = 1
    solution = EspressoDFT._solve_sternheimer(
        operator,
        occupied,
        kinetic,
        [0.0],
        rhs;
        tolerance=1e-12,
        maxiter=20,
    )
    @test solution[2, 1] ≈ 1.0 atol=1e-12
    @test norm(operator * solution - solution .* [0.0]' - rhs) <= 1e-12
    @test occupied' * solution ≈ zeros(ComplexF64, 1, 1) atol=1e-14
end

@testset "public accessors expose copies and their reverse rules preserve cotangents" begin
    gs = synthetic_ground_state()
    force_copy = forces(gs)
    force_copy[1, 1] = 99
    @test forces(gs)[1, 1] == 0.1
    density_copy = density(gs)
    density_copy.values[1] = 99
    @test density(gs).values[1] != 99

    energy_value, energy_pullback = ChainRulesCore.rrule(energy, gs)
    @test energy_value == -1.25
    _, energy_bar = energy_pullback(2.5)
    @test energy_bar.total_energy == 2.5

    force_value, force_pullback = ChainRulesCore.rrule(forces, gs)
    seed_force = fill(0.3, size(force_value))
    _, force_bar = force_pullback(seed_force)
    @test force_bar.force_values == seed_force

    density_value, density_pullback = ChainRulesCore.rrule(density, gs)
    seed_density = fill(0.4, size(density_value.values))
    _, density_bar = density_pullback(
        Tangent{typeof(density_value)}(; values=seed_density))
    @test density_bar.density_values == seed_density
end

@testset "Crystal reverse rule handles Cartesian coordinates and lattice coupling" begin
    lattice = [4.0 0.2 0.0; 0.0 5.0 0.1; 0.0 0.0 6.0]
    positions = reshape([0.8, 1.5, 2.4], 3, 1)
    crystal, pullback = ChainRulesCore.rrule(
        Crystal,
        lattice,
        [:He],
        positions;
        masses=[4.0],
        positions_are_fractional=false,
    )
    fractional_seed = reshape([0.3, -0.2, 0.4], 3, 1)
    lattice_seed = [0.1 0.0 0.0; 0.0 -0.2 0.0; 0.0 0.0 0.3]
    _, lattice_bar, species_bar, positions_bar = pullback(
        Tangent{Crystal}(;
            _lattice=lattice_seed,
            _positions=fractional_seed,
        ))
    expected_positions = lattice' \ fractional_seed
    expected_lattice = lattice_seed -
                       expected_positions * crystal.positions'
    @test species_bar isa NoTangent
    @test positions_bar ≈ expected_positions
    @test lattice_bar ≈ expected_lattice
end
