using EspressoDFT
using ChainRulesCore
using LinearAlgebra
using Test

include("helpers.jl")

@testset "public export boundary" begin
    expected = Set([
        :Crystal, :KSModel, :PlaneWaveBasis, :SCFOptions, :QEInput,
        :AtomicDisplacement, :read_qe_input, :run_qe_input, :ground_state,
        :energy, :forces, :stress, :density, :eigenvalues, :occupations,
        :response, :dynamical_matrix, :phonon_modes,
        :born_effective_charges, :dielectric_tensor,
    ])
    @test Set(filter(!=(:EspressoDFT), names(EspressoDFT))) == expected
    project = read(joinpath(pkgdir(EspressoDFT), "Project.toml"), String)
    @test !occursin(r"(?im)^DFTK\s*=", project)
    @test !occursin(r"(?im)^Quantum.*Espresso", project)
end

@testset "crystal validation and canonicalization" begin
    h = [2.0 0 0; 0 3.0 0; 0 0 4.0]
    crystal = Crystal(h, [:H], reshape([1.25, -0.25, 2.0], 3, 1);
                      masses=[1836.0])
    @test crystal.positions[:, 1] == [0.25, 0.75, 0.0]
    cartesian = Crystal(h, [:H], h * crystal.positions;
                        masses=[1836.0], positions_are_fractional=false)
    @test cartesian.positions ≈ crystal.positions
    leaked = crystal.lattice
    leaked[1, 1] = 99
    @test crystal.lattice == h
    @test_throws ArgumentError Crystal(zeros(3, 3), [:H], zeros(3, 1); masses=[1.0])
end

@testset "options and displacements" begin
    @test SCFOptions().extra_bands == 4
    @test_throws ArgumentError SCFOptions(maxiter=0)
    p = AtomicDisplacement(1, 3, (0, 0.25, 0))
    @test p.q == (0.0, 0.25, 0.0)
    @test_throws ArgumentError AtomicDisplacement(0, 1, (0, 0, 0))
end

@testset "UPF radial quadrature" begin
    odd_values = collect(1.0:5.0)
    even_values = collect(1.0:6.0)
    @test EspressoDFT._simpson_integral(odd_values, ones(5)) == 12.0
    @test EspressoDFT._simpson_integral(even_values, ones(6)) == 17.5

    complex_values = complex.(even_values, reverse(even_values))
    expected = (sum((4.0, 2.0, 4.0, 2.0) .* complex_values[2:5]) +
                complex_values[1] - 0.25complex_values[4] +
                complex_values[5] + 1.25complex_values[6]) / 3
    @test EspressoDFT._simpson_integral(complex_values, ones(6)) == expected
end

@testset "density FFT grid sizing" begin
    lattice = 10.0 .* Matrix{Float64}(I, 3, 3)
    @test EspressoDFT._required_fft_size(lattice, 10.0) == (30, 30, 30)
end

@testset "periodic force acoustic sum rule" begin
    values = [1.0 2.0 -2.0; -3.0 4.0 5.0; 0.5 -0.2 0.9]
    returned = EspressoDFT._enforce_zero_net_force!(values)
    @test returned === values
    @test vec(sum(values; dims=2)) ≈ zeros(3) atol=10eps(Float64)
end

@testset "PBE energy and potential share one functional" begin
    dims = (6, 7, 8)
    lattice = [6.0 0.3 0.1; 0.0 5.0 0.2; 0.0 0.0 7.0]
    reciprocal = 2pi * transpose(inv(lattice))
    rho = reshape([0.08 + 0.02sin(0.17index) + 0.01cos(0.31index)
                   for index in 1:prod(dims)], dims)
    tangent = reshape([sin(0.73index) for index in 1:prod(dims)], dims)
    tangent .-= sum(tangent) / length(tangent)
    core = zeros(dims)
    volume = abs(det(lattice))
    _, potential = EspressoDFT._xc_energy_potential(
        :pbe, rho, core, reciprocal, volume)
    step = 1e-5
    plus = EspressoDFT._xc_energy_potential(
        :pbe, rho .+ step .* tangent, core, reciprocal, volume)[1]
    minus = EspressoDFT._xc_energy_potential(
        :pbe, rho .- step .* tangent, core, reciprocal, volume)[1]
    finite_difference = (plus - minus) / (2step)
    analytic = volume * sum(potential .* tangent) / length(tangent)
    @test isapprox(analytic, finite_difference; atol=2e-8, rtol=2e-7)

    analytic_kernel = EspressoDFT._xc_potential_response(
        :pbe, rho, core, tangent, reciprocal)
    plus_potential = EspressoDFT._xc_energy_potential(
        :pbe, rho .+ step .* tangent, core, reciprocal, volume)[2]
    minus_potential = EspressoDFT._xc_energy_potential(
        :pbe, rho .- step .* tangent, core, reciprocal, volume)[2]
    finite_kernel = (plus_potential .- minus_potential) ./ (2step)
    @test analytic_kernel ≈ finite_kernel atol=2e-7 rtol=2e-6
end

@testset "QE PBE gradient thresholds" begin
    dims = (4, 4, 4)
    reciprocal = 2pi .* Matrix{Float64}(I, 3, 3)
    uniform_density = fill(0.2, dims)
    zero_core = zeros(dims)
    pbe_energy, pbe_potential = EspressoDFT._xc_energy_potential(
        :pbe, uniform_density, zero_core, reciprocal, 1.0)
    lda_x, lda_vx = EspressoDFT._libxc_lda_component(
        EspressoDFT._XC_LDA_X, vec(uniform_density))
    lda_c, lda_vc = EspressoDFT._libxc_lda_component(
        EspressoDFT._XC_LDA_C_PW, vec(uniform_density))
    expected_energy = sum(uniform_density .* reshape(lda_x .+ lda_c, dims)) /
                      length(uniform_density)
    @test pbe_energy ≈ expected_energy atol=1e-13 rtol=1e-13
    @test pbe_potential ≈ reshape(lda_vx .+ lda_vc, dims) atol=1e-13 rtol=1e-13
end

@testset "symmetry-reduced polar response basis" begin
    a = 10.6
    fcc = [0.0 a / 2 a / 2; a / 2 0.0 a / 2; a / 2 a / 2 0.0]
    nacl = Crystal(fcc, [:Na, :Cl],
                   [0.0 0.5; 0.0 0.5; 0.0 0.5]; masses=[1.0, 1.0])
    nacl_basis = EspressoDFT._born_symmetry_basis(nacl)
    nacl_queries, nacl_design = EspressoDFT._born_queries(nacl_basis, 2)
    @test size(nacl_basis, 2) == 2
    @test nacl_queries == [(1, 1), (2, 1)]
    @test rank(nacl_design) == 2

    a, c, u = 5.9, 9.65, 0.382
    hexagonal = [a -a / 2 0.0; 0.0 sqrt(3) * a / 2 0.0; 0.0 0.0 c]
    aln = Crystal(hexagonal, [:Al, :Al, :N, :N],
                  [0.0 2 / 3 0.0 2 / 3;
                   0.0 1 / 3 0.0 1 / 3;
                   0.0 1 / 2 u 1 / 2 + u]; masses=ones(4))
    aln_basis = EspressoDFT._born_symmetry_basis(aln)
    aln_queries, aln_design = EspressoDFT._born_queries(aln_basis, 4)
    @test size(aln_basis, 2) == 4
    @test aln_queries == [(1, 1), (1, 3), (3, 1), (3, 3)]
    @test rank(aln_design) == 4
end

include("unit/upf_radial.jl")
include("unit/hamiltonian.jl")
include("unit/qe_input.jl")
include("unit/response_chainrules.jl")
include("integration/synthetic_workflow.jl")
