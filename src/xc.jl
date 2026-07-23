const _XC_UNPOLARIZED = 1
const _XC_LDA_X = 1
const _XC_LDA_C_PZ = 9
const _XC_LDA_C_PW = 12
const _XC_GGA_X_PBE = 101
const _XC_GGA_C_PBE = 130
const _QE_GGA_RHO_THRESHOLD = 1e-6
const _QE_GGA_GRHO_THRESHOLD = 1e-10

function _with_xc_function(body::Function, function_id::Int)
    handle = ccall((:xc_func_alloc, Libxc_jll.libxc), Ptr{Cvoid}, ())
    handle == C_NULL && error("Libxc allocation failed")
    initialized = false
    try
        status = ccall((:xc_func_init, Libxc_jll.libxc), Cint,
                       (Ptr{Cvoid}, Cint, Cint),
                       handle, function_id, _XC_UNPOLARIZED)
        status == 0 || error("Libxc initialization failed for functional $function_id")
        initialized = true
        body(handle)
    finally
        initialized && ccall((:xc_func_end, Libxc_jll.libxc), Cvoid, (Ptr{Cvoid},), handle)
        ccall((:xc_func_free, Libxc_jll.libxc), Cvoid, (Ptr{Cvoid},), handle)
    end
end

function _libxc_lda_component(function_id::Int, rho::Vector{Float64})
    exc = zeros(length(rho))
    vrho = zeros(length(rho))
    _with_xc_function(function_id) do handle
        ccall((:xc_lda_exc_vxc, Libxc_jll.libxc), Cvoid,
              (Ptr{Cvoid}, Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
              handle, length(rho), rho, exc, vrho)
    end
    exc, vrho
end

function _libxc_gga_component(function_id::Int, rho::Vector{Float64},
                              sigma::Vector{Float64})
    exc = zeros(length(rho))
    vrho = zeros(length(rho))
    vsigma = zeros(length(rho))
    _with_xc_function(function_id) do handle
        ccall((:xc_gga_exc_vxc, Libxc_jll.libxc), Cvoid,
              (Ptr{Cvoid}, Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}),
              handle, length(rho), rho, sigma, exc, vrho, vsigma)
    end
    exc, vrho, vsigma
end

function _libxc_lda_kernel_component(function_id::Int,
                                     rho::Vector{Float64})
    v2rho2 = zeros(length(rho))
    _with_xc_function(function_id) do handle
        ccall((:xc_lda_fxc, Libxc_jll.libxc), Cvoid,
              (Ptr{Cvoid}, Csize_t, Ptr{Float64}, Ptr{Float64}),
              handle, length(rho), rho, v2rho2)
    end
    v2rho2
end

function _libxc_gga_kernel_component(function_id::Int,
                                     rho::Vector{Float64},
                                     sigma::Vector{Float64})
    v2rho2 = zeros(length(rho))
    v2rhosigma = zeros(length(rho))
    v2sigma2 = zeros(length(rho))
    _with_xc_function(function_id) do handle
        ccall((:xc_gga_fxc, Libxc_jll.libxc), Cvoid,
              (Ptr{Cvoid}, Csize_t, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
              handle, length(rho), rho, sigma,
              v2rho2, v2rhosigma, v2sigma2)
    end
    v2rho2, v2rhosigma, v2sigma2
end

function _qe_pbe_energy_potential_arrays(rho::Vector{Float64},
                                         sigma::Vector{Float64})
    exc_x, vrho_x = _libxc_lda_component(_XC_LDA_X, rho)
    exc_c, vrho_c = _libxc_lda_component(_XC_LDA_C_PW, rho)
    exc = exc_x .+ exc_c
    vrho = vrho_x .+ vrho_c
    vsigma = zeros(length(rho))

    c1 = 0.75 / pi
    c2 = 3.093667726280136
    pbe_kappa = 0.804
    pbe_mu = 0.2195149727645171
    ga = 0.0310906908696548950
    beta = 0.06672455060314922
    pi34 = 0.6203504908994
    xkf = 1.919158292677513
    xks = 1.128379167095513
    for index in eachindex(rho)
        density = rho[index]
        grho = sigma[index]
        if density <= _QE_GGA_RHO_THRESHOLD ||
           grho <= _QE_GGA_GRHO_THRESHOLD
            continue
        end

        # PBE exchange correction to the uniform-gas term (Perdew, Burke,
        # Ernzerhof, PRL 77, 3865), in Hartree atomic units.  The compatibility
        # constants reproduce QE 7.5's published numerical convention.
        gradient_norm = sqrt(grho)
        fermi_wavevector = c2 * cbrt(density)
        dscaled_dgradient = 0.5 / fermi_wavevector
        scaled_gradient = gradient_norm * dscaled_dgradient / density
        f1 = scaled_gradient^2 * pbe_mu / pbe_kappa
        f2 = 1 + f1
        enhancement_correction = pbe_kappa - pbe_kappa / f2
        uniform_exchange = -c1 * fermi_wavevector
        exchange_per_particle = uniform_exchange * enhancement_correction
        dfx = 2pbe_mu * scaled_gradient / f2^2
        density_derivative = -4scaled_gradient / 3
        v1x = exchange_per_particle + uniform_exchange / 3 * enhancement_correction +
              uniform_exchange * dfx * density_derivative
        v2x = uniform_exchange * dfx * dscaled_dgradient / gradient_norm

        # PBE correlation correction on the PW92 local baseline.  Expressing
        # it through the local correlation energy and potential keeps the
        # energy and its first density derivative on one code path.
        rs = pi34 / cbrt(density)
        kf = xkf / rs
        screening_wavevector = xks * sqrt(kf)
        reduced_gradient = gradient_norm / (2screening_wavevector * density)
        exponential = exp(-exc_c[index] / ga)
        af = beta / ga / (exponential - 1)
        bf = exponential * (vrho_c[index] - exc_c[index])
        y = af * reduced_gradient^2
        denominator = 1 + y + y^2
        xy = (1 + y) / denominator
        qy = y^2 * (2 + y) / denominator^2
        s1 = 1 + beta / ga * reduced_gradient^2 * xy
        h0 = ga * log(s1)
        dh0 = beta * reduced_gradient^2 / s1 *
              (-7xy / 3 - qy * (af * bf / beta - 7 / 3))
        v1c = h0 + dh0
        v2c = beta / (2screening_wavevector^2 * density) *
              (xy - qy) / s1

        exc[index] += exchange_per_particle + h0
        vrho[index] += v1x + v1c
        vsigma[index] = (v2x + v2c) / 2
    end
    exc, vrho, vsigma
end

function _signed_frequency(index::Int, n::Int)
    value = index - 1
    value <= fld(n, 2) ? value : value - n
end

function _gradient_real(values::Array{Float64,3}, reciprocal::Matrix{Float64},
                        density_cutoff::Real=Inf)
    dims = size(values)
    coefficients = fft(values) / length(values)
    gradient = ntuple(_ -> zeros(Float64, dims), 3)
    for component in 1:3
        derivative = similar(coefficients)
        for i in axes(coefficients, 1), j in axes(coefficients, 2), k in axes(coefficients, 3)
            g = (_signed_frequency(i, dims[1]), _signed_frequency(j, dims[2]),
                 _signed_frequency(k, dims[3]))
            cartesian_g = reciprocal * collect(g)
            derivative[i, j, k] = if sum(abs2, cartesian_g) / 2 <= density_cutoff
                im * cartesian_g[component] * coefficients[i, j, k]
            else
                zero(eltype(derivative))
            end
        end
        gradient[component] .= real.(ifft(derivative) .* length(values))
    end
    gradient
end

function _divergence_real(fields::NTuple{3,Array{Float64,3}},
                          reciprocal::Matrix{Float64},
                          density_cutoff::Real=Inf)
    dims = size(fields[1])
    divergence_coefficients = zeros(ComplexF64, dims)
    for component in 1:3
        coefficients = fft(fields[component]) / length(fields[component])
        for i in axes(coefficients, 1), j in axes(coefficients, 2), k in axes(coefficients, 3)
            g = (_signed_frequency(i, dims[1]), _signed_frequency(j, dims[2]),
                 _signed_frequency(k, dims[3]))
            cartesian_g = reciprocal * collect(g)
            sum(abs2, cartesian_g) / 2 <= density_cutoff || continue
            divergence_coefficients[i, j, k] +=
                im * cartesian_g[component] * coefficients[i, j, k]
        end
    end
    real.(ifft(divergence_coefficients) .* prod(dims))
end

function _xc_energy_potential(xc::Symbol, valence_density::Array{Float64,3},
                              core_density::Array{Float64,3},
                              reciprocal::Matrix{Float64}, volume::Float64,
                              density_cutoff::Real=Inf)
    total_density = max.(valence_density .+ core_density, 1e-14)
    flat_rho = vec(total_density)
    if xc == :lda
        exc_x, vrho_x = _libxc_lda_component(_XC_LDA_X, flat_rho)
        # QE's documented `input_dft='LDA'` compatibility target is the
        # Slater exchange plus Perdew-Zunger correlation combination.
        exc_c, vrho_c = _libxc_lda_component(_XC_LDA_C_PZ, flat_rho)
        exc = exc_x .+ exc_c
        potential = reshape(vrho_x .+ vrho_c, size(total_density))
        energy_value = volume * sum(total_density .* reshape(exc, size(total_density))) /
                       length(total_density)
    else
        gradient = _gradient_real(total_density, reciprocal, density_cutoff)
        sigma = vec(sum(component .^ 2 for component in gradient))
        exc, vrho, vsigma_flat = _qe_pbe_energy_potential_arrays(flat_rho, sigma)
        vsigma = reshape(vsigma_flat, size(total_density))
        flux = ntuple(component -> 2 .* vsigma .* gradient[component], 3)
        potential = reshape(vrho, size(total_density)) .-
                    _divergence_real(flux, reciprocal, density_cutoff)
        energy_value = volume * sum(total_density .* reshape(exc, size(total_density))) /
                       length(total_density)
    end
    energy_value, potential
end

function _xc_potential_response(xc::Symbol,
                                valence_density::Array{Float64,3},
                                core_density::Array{Float64,3},
                                delta_density::Array{Float64,3},
                                reciprocal::Matrix{Float64},
                                density_cutoff::Real=Inf)
    total_density = max.(valence_density .+ core_density, 1e-14)
    rho = vec(total_density)
    delta_rho = vec(delta_density)
    if xc == :lda
        exchange = _libxc_lda_kernel_component(_XC_LDA_X, rho)
        correlation = _libxc_lda_kernel_component(_XC_LDA_C_PZ, rho)
        return reshape((exchange .+ correlation) .* delta_rho,
                       size(total_density))
    end

    gradient = _gradient_real(total_density, reciprocal, density_cutoff)
    delta_gradient = _gradient_real(delta_density, reciprocal, density_cutoff)
    sigma = vec(sum(component .^ 2 for component in gradient))
    delta_sigma = vec(2sum(gradient[component] .* delta_gradient[component]
                           for component in 1:3))
    _, _, exchange_vsigma = _libxc_gga_component(
        _XC_GGA_X_PBE, rho, sigma)
    _, _, correlation_vsigma = _libxc_gga_component(
        _XC_GGA_C_PBE, rho, sigma)
    exchange_kernel = _libxc_gga_kernel_component(
        _XC_GGA_X_PBE, rho, sigma)
    correlation_kernel = _libxc_gga_kernel_component(
        _XC_GGA_C_PBE, rho, sigma)
    v2rho2 = exchange_kernel[1] .+ correlation_kernel[1]
    v2rhosigma = exchange_kernel[2] .+ correlation_kernel[2]
    v2sigma2 = exchange_kernel[3] .+ correlation_kernel[3]
    lda_v2rho2 = _libxc_lda_kernel_component(_XC_LDA_X, rho) .+
                 _libxc_lda_kernel_component(_XC_LDA_C_PW, rho)
    vsigma_flat = exchange_vsigma .+ correlation_vsigma
    for index in eachindex(rho)
        if rho[index] <= _QE_GGA_RHO_THRESHOLD ||
           sigma[index] <= _QE_GGA_GRHO_THRESHOLD
            v2rho2[index] = lda_v2rho2[index]
            v2rhosigma[index] = 0
            v2sigma2[index] = 0
            vsigma_flat[index] = 0
        end
    end
    delta_vrho = reshape(
        v2rho2 .* delta_rho .+ v2rhosigma .* delta_sigma,
        size(total_density))
    delta_vsigma = reshape(
        v2rhosigma .* delta_rho .+ v2sigma2 .* delta_sigma,
        size(total_density))
    vsigma = reshape(vsigma_flat, size(total_density))
    flux = ntuple(component ->
        2 .* delta_vsigma .* gradient[component] .+
        2 .* vsigma .* delta_gradient[component], 3)
    delta_vrho .- _divergence_real(flux, reciprocal, density_cutoff)
end
