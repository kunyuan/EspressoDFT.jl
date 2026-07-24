using CUDA
using EspressoDFT
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "helpers.jl"))

CUDA.functional() || error("CUDA smoke test requires a functional GPU")
options = SCFOptions(
    energy_tolerance=1e-8,
    density_tolerance=1e-7,
    maxiter=80,
    extra_bands=0,
)
# Ecut=2 makes NG=81, deliberately crossing the toy dense threshold so the
# smoke test exercises CUDA FFT, kinetic, and nonlocal Hamiltonian kernels.
cuda_basis = PlaneWaveBasis(
    SYNTHETIC_BASIS.model; Ecut=2.0, kgrid=(1, 1, 1))
@test !EspressoDFT._use_dense_eigensolver(
    length(cuda_basis.G_vectors[1]), 1)
cpu = ground_state(cuda_basis; options)
gpu = EspressoDFT.gpu_ground_state(cuda_basis; options)

@test gpu.converged
@test energy(gpu) ≈ energy(cpu) atol=5e-9 rtol=5e-9
@test gpu.density_residual_history[end] <= options.density_tolerance
