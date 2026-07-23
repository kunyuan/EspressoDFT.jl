@testset "reciprocal indexing and real transforms are mutually consistent" begin
    dims = (6, 5, 4)
    for g in ((0, 0, 0), (1, -2, 3), (-3, 2, -1))
        index = EspressoDFT._fft_index(g, dims)
        @test all(1 .<= collect(index) .<= collect(dims))
        @test all(mod(EspressoDFT._grid_g(index, dims)[axis] - g[axis],
                      dims[axis]) == 0 for axis in 1:3)
    end

    coefficients = zeros(ComplexF64, dims)
    coefficients[1, 1, 1] = 2.5
    field = EspressoDFT._real_from_coefficients(coefficients)
    @test field ≈ fill(2.5, dims)
end

@testset "nonlocal projector action equals its explicit Hermitian matrix" begin
    factors = ComplexF64[1 0; 0.2im 1; -0.3 0.4]
    coupling = [1.2 0.1; 0.1 -0.4]
    projectors = EspressoDFT.NonlocalProjectors(factors, coupling, [1, 1])
    matrix = EspressoDFT._nonlocal_matrix(projectors)
    vectors = ComplexF64[1 0.2im; -0.5 1; 0.7 -0.3im]
    @test matrix ≈ matrix'
    @test EspressoDFT._apply_nonlocal(projectors, vectors) ≈ matrix * vectors
end

@testset "matrix-free Hamiltonian agrees with dense construction" begin
    basis = SYNTHETIC_BASIS
    kernel = SYNTHETIC_KERNEL
    local_coefficients = copy(kernel.ionic_coefficients)
    dense = EspressoDFT._hamiltonian(kernel, 1, local_coefficients)
    kpoint = basis.kpoints[1]
    kinetic = [
        sum(abs2, kernel.reciprocal * (collect(kpoint) .+ collect(g))) / 2
        for g in basis.G_vectors[1]
    ]
    operator = EspressoDFT.KSHamiltonian(
        kernel,
        1,
        EspressoDFT._real_from_coefficients(local_coefficients),
        kinetic,
    )
    vector = ComplexF64[cis(0.37index) / (1 + index)
                        for index in axes(dense, 1)]
    output = similar(vector)
    mul!(output, operator, vector)
    @test dense ≈ dense' atol=5e-13
    @test output ≈ dense * vector atol=5e-11 rtol=5e-11
    @test_throws DimensionMismatch mul!(
        zeros(ComplexF64, length(vector) - 1), operator, vector)
end

@testset "Hartree and density mixing preserve physical invariants" begin
    kernel = SYNTHETIC_KERNEL
    uniform = fill(0.4, SYNTHETIC_BASIS.fft_size)
    potential, energy_value = EspressoDFT._hartree(uniform, kernel)
    @test iszero(energy_value)
    @test all(iszero, potential)

    modulated = copy(uniform)
    for index in CartesianIndices(modulated)
        modulated[index] += 0.01cos(2pi * (index[1] - 1) / size(modulated, 1))
    end
    potential, energy_value = EspressoDFT._hartree(modulated, kernel)
    @test energy_value > 0
    @test any(!iszero, potential)

    mixed = EspressoDFT._mix_density(uniform, modulated, kernel)
    target_average = SYNTHETIC_BASIS.model.electron_count / kernel.volume
    @test sum(mixed) / length(mixed) ≈ target_average atol=5e-15
    @test all(isfinite, mixed)
end

@testset "block eigensolver agrees with direct diagonalization on a small basis" begin
    kernel = SYNTHETIC_KERNEL
    local_coefficients = copy(kernel.ionic_coefficients)
    dense = EspressoDFT._hamiltonian(kernel, 1, local_coefficients)
    expected = eigvals(Hermitian(dense))[1]
    values, vectors = EspressoDFT._iterative_eigensolve(
        kernel, 1, local_coefficients, 1, nothing; target_residual=1e-9)
    @test values[1] ≈ expected atol=2e-9 rtol=2e-9
    @test norm(dense * vectors[:, 1] - values[1] * vectors[:, 1]) <= 1e-9
    @test vectors' * vectors ≈ I atol=2e-12
end
