@testset "public QE parser constructs the same canonical objects from path and IO" begin
    mktempdir() do directory
        pseudo = write_synthetic_upf(directory)
        text = synthetic_qe_input(pseudo)
        path = joinpath(directory, "pw.in")
        write(path, text)

        from_path = read_qe_input(path)
        from_io = read_qe_input(IOBuffer(text))
        @test from_path isa QEInput
        @test from_path.model.crystal.lattice ≈ from_io.model.crystal.lattice
        @test from_path.model.crystal.positions ≈ from_io.model.crystal.positions
        @test from_path.model.electron_count == 2.0
        @test from_path.basis.Ecut == 0.8
        @test from_path.basis.kpoints == [(0.0, 0.0, 0.0)]
        @test from_path.options.energy_tolerance == 1e-10
        @test from_path.options.maxiter == 25

        shifted = synthetic_qe_input(pseudo; k_shift=(1, 0, 0))
        @test_throws ArgumentError read_qe_input(IOBuffer(shifted))
        @test_throws ArgumentError read_qe_input(
            IOBuffer(replace(text, "calculation = 'scf'" => "calculation = 'relax'")))
        @test_throws ArgumentError read_qe_input(
            IOBuffer(replace(text, "ecutwfc = 1.6d0" =>
                                  "mystery_keyword = 1\n  ecutwfc = 1.6d0")))
    end
end

@testset "QE scalar parsing is strict and handles Fortran notation" begin
    @test EspressoDFT._parse_qe_value("2.5d-3") == 2.5e-3
    @test EspressoDFT._parse_qe_value(".TRUE.") === true
    @test EspressoDFT._parse_qe_value("'Mixed Case'") == "Mixed Case"
    @test_throws ArgumentError EspressoDFT._parse_qe_value("1.2.3")
    @test_throws ArgumentError read_qe_input("/definitely/missing/pw.in")
end
