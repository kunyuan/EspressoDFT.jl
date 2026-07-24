using EspressoDFT
using LinearAlgebra
using MPI
using Test

include(joinpath(@__DIR__, "..", "helpers.jl"))

MPI.Init()
communicator = MPI.COMM_WORLD
rank = MPI.Comm_rank(communicator)
options = SCFOptions(
    energy_tolerance=1e-8,
    density_tolerance=1e-7,
    maxiter=80,
    extra_bands=0,
)
mpi_basis = PlaneWaveBasis(
    SYNTHETIC_BASIS.model; Ecut=0.8, kgrid=(2, 1, 1))

distributed = EspressoDFT.mpi_ground_state(
    mpi_basis; options, communicator)
distributed_energy = energy(distributed)
minimum_energy = MPI.Allreduce(distributed_energy, min, communicator)
maximum_energy = MPI.Allreduce(distributed_energy, max, communicator)
reference_energy = rank == 0 ?
    energy(ground_state(mpi_basis; options)) : 0.0
reference_energy = MPI.bcast(reference_energy, 0, communicator)

@test distributed.converged
@test minimum_energy == maximum_energy
@test distributed_energy ≈ reference_energy atol=5e-11 rtol=5e-11
