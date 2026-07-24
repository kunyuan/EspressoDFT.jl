module EspressoDFTCUDAExt

using CUDA
using EspressoDFT
using FFTW
using LinearAlgebra

mutable struct CUDABatchBuffer
    input::CuMatrix{ComplexF64}
    output::CuMatrix{ComplexF64}
    grid::CuArray{ComplexF64,4}
end

mutable struct CUDAKPointWorkspace
    kinetic::CuVector{Float64}
    indices::CuVector{Int}
    factors::CuMatrix{ComplexF64}
    coupling::CuMatrix{Float64}
    batches::Dict{Int,CUDABatchBuffer}
end

mutable struct CUDABackend <: EspressoDFT.AbstractSCFBackend
    workspaces::IdDict{EspressoDFT.PWKernel,Vector{CUDAKPointWorkspace}}
end

CUDABackend() = CUDABackend(
    IdDict{EspressoDFT.PWKernel,Vector{CUDAKPointWorkspace}}())

struct CUDAKSHamiltonian <: AbstractMatrix{ComplexF64}
    workspace::CUDAKPointWorkspace
    local_potential::CuArray{Float64,3}
    dims::NTuple{3,Int}
end

Base.size(operator::CUDAKSHamiltonian) =
    (length(operator.workspace.kinetic), length(operator.workspace.kinetic))

function _scatter_kernel!(grid, input, indices, grid_length, rows)
    linear = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if linear <= length(input)
        row = mod(linear - 1, rows) + 1
        column = div(linear - 1, rows) + 1
        grid[indices[row] + (column - 1) * grid_length] = input[row, column]
    end
    return
end

function _gather_kernel!(output, grid, input, indices, kinetic,
                         grid_length, rows)
    linear = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if linear <= length(input)
        row = mod(linear - 1, rows) + 1
        column = div(linear - 1, rows) + 1
        output[row, column] =
            kinetic[row] * input[row, column] +
            grid[indices[row] + (column - 1) * grid_length]
    end
    return
end

function _batch_buffer!(
    workspace::CUDAKPointWorkspace,
    dims::NTuple{3,Int},
    number_vectors::Int,
)
    get!(workspace.batches, number_vectors) do
        rows = length(workspace.kinetic)
        CUDABatchBuffer(
            CUDA.zeros(ComplexF64, rows, number_vectors),
            CUDA.zeros(ComplexF64, rows, number_vectors),
            CUDA.zeros(ComplexF64, dims..., number_vectors),
        )
    end
end

function LinearAlgebra.mul!(
    output::AbstractVecOrMat,
    operator::CUDAKSHamiltonian,
    input::AbstractVecOrMat,
)
    input_matrix = input isa AbstractVector ? reshape(input, :, 1) : input
    output_matrix = output isa AbstractVector ? reshape(output, :, 1) : output
    size(input_matrix, 1) == size(operator, 1) || throw(DimensionMismatch())
    size(output_matrix) == size(input_matrix) || throw(DimensionMismatch())
    number_vectors = size(input_matrix, 2)
    batch = _batch_buffer!(
        operator.workspace, operator.dims, number_vectors)
    copyto!(batch.input, input_matrix)
    fill!(batch.grid, 0)
    threads = 256
    blocks = cld(length(batch.input), threads)
    grid_length = prod(operator.dims)
    rows = size(input_matrix, 1)
    @cuda threads=threads blocks=blocks _scatter_kernel!(
        batch.grid, batch.input, operator.workspace.indices,
        grid_length, rows)
    real_space = ifft(batch.grid, (1, 2, 3))
    real_space .*= reshape(
        operator.local_potential, operator.dims..., 1)
    local_grid = fft(real_space, (1, 2, 3))
    @cuda threads=threads blocks=blocks _gather_kernel!(
        batch.output, local_grid, batch.input, operator.workspace.indices,
        operator.workspace.kinetic, grid_length, rows)
    if !isempty(operator.workspace.factors)
        batch.output .+= operator.workspace.factors * (
            operator.workspace.coupling * (
                operator.workspace.factors' * batch.input))
    end
    copyto!(output_matrix, batch.output)
    output
end

function _cuda_workspaces(kernel::EspressoDFT.PWKernel)
    [
        CUDAKPointWorkspace(
            CuArray(kernel.kinetic_energies[kindex]),
            CuArray(kernel.fft_indices[kindex]),
            CuArray(projectors.factors),
            CuArray(projectors.coupling),
            Dict{Int,CUDABatchBuffer}(),
        )
        for (kindex, projectors) in
            pairs(kernel.nonlocal_projectors)
    ]
end

function EspressoDFT._solve_electrons_backend(
    backend::CUDABackend,
    kernel::EspressoDFT.PWKernel,
    local_coefficients::Array{ComplexF64,3},
    number_bands::Int,
    previous::Union{Nothing,Vector{Matrix{ComplexF64}}},
    target_residual::Float64,
)
    CUDA.functional() || error("CUDA is loaded but no functional GPU is available")
    workspaces = get!(backend.workspaces, kernel) do
        _cuda_workspaces(kernel)
    end
    basis = kernel.basis
    dimensions = length.(getfield(basis, :_G_vectors))
    local_potential =
        EspressoDFT._real_from_coefficients(local_coefficients)
    values = Vector{Vector{Float64}}(undef, length(dimensions))
    vectors = Vector{Matrix{ComplexF64}}(undef, length(dimensions))
    for kindex in eachindex(dimensions)
        if EspressoDFT._use_dense_eigensolver(
            dimensions[kindex], number_bands)
            values[kindex], vectors[kindex] =
                EspressoDFT._solve_electron_kpoint(
                    kernel, local_coefficients, local_potential, number_bands,
                    previous, target_residual, kindex)
            continue
        end
        operator = CUDAKSHamiltonian(
            workspaces[kindex], CuArray(local_potential),
            getfield(basis, :_fft_size))
        prior = previous === nothing ? nothing : previous[kindex]
        values[kindex], vectors[kindex] = EspressoDFT._block_eigensolve(
            operator, kernel.kinetic_energies[kindex], number_bands, prior;
            target_residual)
    end
    values, vectors
end

function EspressoDFT.gpu_ground_state(
    basis::EspressoDFT.PlaneWaveBasis;
    options::EspressoDFT.SCFOptions=EspressoDFT.SCFOptions(),
    device::Integer=0,
)
    CUDA.functional() || error("CUDA is loaded but no functional GPU is available")
    CUDA.device!(device)
    EspressoDFT._ground_state_backend(
        basis, options, CUDABackend())
end

end
