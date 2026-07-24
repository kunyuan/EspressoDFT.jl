module EspressoDFTMPIExt

using EspressoDFT
using MPI

struct MPIBackend{C} <: EspressoDFT.AbstractSCFBackend
    communicator::C
end

function EspressoDFT.mpi_ground_state(
    basis::EspressoDFT.PlaneWaveBasis;
    options::EspressoDFT.SCFOptions=EspressoDFT.SCFOptions(),
    communicator=MPI.COMM_WORLD,
)
    MPI.Initialized() || MPI.Init()
    EspressoDFT._ground_state_backend(
        basis, options, MPIBackend(communicator))
end

function EspressoDFT._solve_electrons_backend(
    backend::MPIBackend,
    kernel::EspressoDFT.PWKernel,
    local_coefficients::Array{ComplexF64,3},
    number_bands::Int,
    previous::Union{Nothing,Vector{Matrix{ComplexF64}}},
    target_residual::Float64,
)
    communicator = backend.communicator
    rank = MPI.Comm_rank(communicator)
    number_ranks = MPI.Comm_size(communicator)
    basis = kernel.basis
    dimensions = length.(getfield(basis, :_G_vectors))
    nk = length(dimensions)
    local_potential = any(dimension ->
        !EspressoDFT._use_dense_eigensolver(dimension, number_bands),
        dimensions) ? EspressoDFT._real_from_coefficients(local_coefficients) :
                      nothing

    values = [zeros(Float64, number_bands) for _ in 1:nk]
    vectors = [zeros(ComplexF64, dimension, number_bands)
               for dimension in dimensions]
    for kindex in 1:nk
        owner = mod(kindex - 1, number_ranks)
        if rank == owner
            values[kindex], vectors[kindex] =
                EspressoDFT._solve_electron_kpoint(
                    kernel, local_coefficients, local_potential, number_bands,
                    previous, target_residual, kindex)
        end
        MPI.Bcast!(values[kindex], owner, communicator)
        MPI.Bcast!(vectors[kindex], owner, communicator)
    end
    values, vectors
end

end
