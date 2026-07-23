@testset "UPF parser validates the supported NC subset" begin
    mktempdir() do directory
        path = write_synthetic_upf(directory)
        upf = EspressoDFT._read_upf(path)
        @test upf.element == :He
        @test upf.z_valence == 2.0
        @test upf.pseudo_type == "NC"
        @test upf.relativistic == "scalar"
        @test upf.core_correction
        @test length(upf.radial_grid) == 5
        @test upf.projector_l == [0]
        @test upf.dij_ry == reshape([1.2], 1, 1)

        bad_version = joinpath(directory, "bad-version.upf")
        write(bad_version, replace(synthetic_upf_xml(),
                                   "version=\"2.0.1\"" => "version=\"2.0.0\""))
        @test_throws ArgumentError EspressoDFT._read_upf(bad_version)

        ultrasoft = joinpath(directory, "ultrasoft.upf")
        write(ultrasoft, replace(synthetic_upf_xml(),
                                 "is_ultrasoft=\"F\"" => "is_ultrasoft=\"T\""))
        @test_throws ArgumentError EspressoDFT._read_upf(ultrasoft)

        missing_nlcc = joinpath(directory, "missing-nlcc.upf")
        write(missing_nlcc, replace(
            synthetic_upf_xml(),
            "<PP_NLCC>0.10 0.08 0.05 0.02 0.0</PP_NLCC>" => "",
        ))
        @test_throws ArgumentError EspressoDFT._read_upf(missing_nlcc)
    end
end

@testset "radial special functions and transforms have independent limits" begin
    @test EspressoDFT._spherical_bessel(0, 0.0) == 1.0
    @test EspressoDFT._spherical_bessel(1, 0.0) == 0.0
    @test EspressoDFT._spherical_bessel(2, 0.0) == 0.0
    @test EspressoDFT._spherical_bessel(0, 0.7) ≈ sin(0.7) / 0.7
    @test EspressoDFT._spherical_bessel(1, 0.7) ≈
          sin(0.7) / 0.7^2 - cos(0.7) / 0.7
    @test EspressoDFT._legendre(0, 0.3) == 1.0
    @test EspressoDFT._legendre(2, 0.3) ≈ (3 * 0.3^2 - 1) / 2

    upf = synthetic_upf()
    expected_atomic = EspressoDFT._simpson_integral(
        upf.atomic_density_radial, upf.radial_weights)
    @test EspressoDFT._atomic_density_transform(upf, 0.0) ≈ expected_atomic
    @test isfinite(EspressoDFT._local_radial_transform(upf, 0.4))
    @test isfinite(EspressoDFT._projector_transform(upf, 1, 0.4))
    @test EspressoDFT._core_density_transform(
        synthetic_upf(core_correction=false), 0.4) == 0.0
    @test EspressoDFT._core_density_transform(upf, 0.4) > 0

    @test_throws DimensionMismatch EspressoDFT._simpson_integral(
        ones(4), ones(3))
    @test_throws ArgumentError EspressoDFT._simpson_integral(ones(2), ones(2))
end
