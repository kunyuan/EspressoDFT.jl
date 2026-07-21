using EspressoDFT
using LinearAlgebra
using Test

@testset "public export boundary" begin
    expected = Set([
        :Crystal, :KSModel, :PlaneWaveBasis, :SCFOptions, :QEInput,
        :AtomicDisplacement, :read_qe_input, :run_qe_input, :ground_state,
        :energy, :forces, :stress, :density, :eigenvalues, :occupations,
        :response, :dynamical_matrix, :phonon_modes,
        :born_effective_charges, :dielectric_tensor,
    ])
    @test Set(filter(!=(:EspressoDFT), names(EspressoDFT))) == expected
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
