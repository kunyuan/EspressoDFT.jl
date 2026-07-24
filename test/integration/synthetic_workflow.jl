const SYNTHETIC_OPTIONS = SCFOptions(
    energy_tolerance=1e-8,
    density_tolerance=1e-7,
    maxiter=100,
    extra_bands=1,
)

@testset "synthetic public ground-state workflow preserves invariants" begin
    gs = ground_state(SYNTHETIC_BASIS; options=SYNTHETIC_OPTIONS)
    @test isfinite(energy(gs))
    @test hasproperty(gs, :iterations)
    @test gs.iterations == length(gs.energy_history)
    @test gs.iterations == length(gs.density_residual_history)
    @test last(gs.density_residual_history) <=
          SYNTHETIC_OPTIONS.density_tolerance
    copied_history = gs.energy_history
    copied_history[1] = Inf
    @test isfinite(first(gs.energy_history))
    @test size(forces(gs)) == (3, 1)
    @test vec(sum(forces(gs); dims=2)) ≈ zeros(3) atol=1e-12
    @test stress(gs) ≈ stress(gs)' atol=1e-12
    @test all(isfinite, stress(gs))

    rho = density(gs)
    electron_count = sum(rho.values) * rho.cell_volume / length(rho.values)
    @test electron_count ≈ SYNTHETIC_BASIS.model.electron_count atol=2e-7
    @test length(eigenvalues(gs)) == length(SYNTHETIC_BASIS.kpoints)
    @test length(occupations(gs)) == length(SYNTHETIC_BASIS.kpoints)
    @test sum(occupations(gs)[1]) == 2.0
    @test_throws ErrorException ground_state(
        SYNTHETIC_BASIS;
        options=SCFOptions(maxiter=1, extra_bands=0),
    )
end

@testset "synthetic QE-compatible workflow agrees with native construction" begin
    mktempdir() do directory
        pseudo = write_synthetic_upf(directory)
        text = synthetic_qe_input(pseudo)
        text = replace(
            text,
            "conv_thr = 2.0d-10" => "conv_thr = 2.0d-8",
            "electron_maxstep = 25" => "electron_maxstep = 100",
        )
        parsed = read_qe_input(IOBuffer(text))
        parsed_state = run_qe_input(parsed)
        native_state = ground_state(SYNTHETIC_BASIS; options=SYNTHETIC_OPTIONS)
        @test energy(parsed_state) ≈ energy(native_state) atol=2e-8 rtol=2e-8
        @test density(parsed_state).values ≈
              density(native_state).values atol=2e-7 rtol=2e-7
    end
end

@testset "synthetic Gamma response and phonon workflow is bounded" begin
    gs = ground_state(SYNTHETIC_BASIS; options=SYNTHETIC_OPTIONS)
    perturbation = AtomicDisplacement(1, 1, (0.0, 0.0, 0.0))
    result = response(gs, perturbation; tolerance=1e-6, maxiter=80)
    @test result.converged
    @test result.residual_norm <= 1e-6
    @test size(result.delta_density) == SYNTHETIC_BASIS.fft_size
    @test all(isfinite, result.delta_density)
    @test norm(result.delta_density) > 1e-8

    dynamical = dynamical_matrix(
        gs, (0.0, 0.0, 0.0); tolerance=1e-6, maxiter=80)
    modes = phonon_modes(
        gs, (0.0, 0.0, 0.0); tolerance=1e-6, maxiter=80)
    @test size(dynamical) == (3, 3)
    @test dynamical ≈ dynamical' atol=5e-10
    @test length(modes.frequencies) == 3
    @test modes.eigenvectors' * modes.eigenvectors ≈ I atol=5e-10
    @test_throws ArgumentError response(
        gs, perturbation; tolerance=0.0, maxiter=80)
end
