const _XC_UNPOLARIZED = 1
const _XC_LDA_X = 1
const _XC_LDA_C_PZ = 9
const _XC_LDA_C_PW = 12
const _XC_GGA_X_PBE = 101
const _XC_GGA_C_PBE = 130

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

function _signed_frequency(index::Int, n::Int)
    value = index - 1
    value <= fld(n, 2) ? value : value - n
end

function _gradient_real(values::Array{Float64,3}, reciprocal::Matrix{Float64})
    dims = size(values)
    coefficients = fft(values) / length(values)
    gradient = ntuple(_ -> zeros(Float64, dims), 3)
    for component in 1:3
        derivative = similar(coefficients)
        for i in axes(coefficients, 1), j in axes(coefficients, 2), k in axes(coefficients, 3)
            g = (_signed_frequency(i, dims[1]), _signed_frequency(j, dims[2]),
                 _signed_frequency(k, dims[3]))
            derivative[i, j, k] = im * dot(reciprocal[component, :], collect(g)) *
                                  coefficients[i, j, k]
        end
        gradient[component] .= real.(ifft(derivative) .* length(values))
    end
    gradient
end

function _divergence_real(fields::NTuple{3,Array{Float64,3}},
                          reciprocal::Matrix{Float64})
    dims = size(fields[1])
    divergence_coefficients = zeros(ComplexF64, dims)
    for component in 1:3
        coefficients = fft(fields[component]) / length(fields[component])
        for i in axes(coefficients, 1), j in axes(coefficients, 2), k in axes(coefficients, 3)
            g = (_signed_frequency(i, dims[1]), _signed_frequency(j, dims[2]),
                 _signed_frequency(k, dims[3]))
            divergence_coefficients[i, j, k] +=
                im * dot(reciprocal[component, :], collect(g)) * coefficients[i, j, k]
        end
    end
    real.(ifft(divergence_coefficients) .* prod(dims))
end

function _xc_energy_potential(xc::Symbol, valence_density::Array{Float64,3},
                              core_density::Array{Float64,3},
                              reciprocal::Matrix{Float64}, volume::Float64)
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
        gradient = _gradient_real(total_density, reciprocal)
        sigma = vec(sum(component .^ 2 for component in gradient))
        exc_x, vrho_x, vsigma_x = _libxc_gga_component(
            _XC_GGA_X_PBE, flat_rho, sigma)
        exc_c, vrho_c, vsigma_c = _libxc_gga_component(
            _XC_GGA_C_PBE, flat_rho, sigma)
        exc = exc_x .+ exc_c
        vrho = vrho_x .+ vrho_c
        vsigma_flat = vsigma_x .+ vsigma_c
        vsigma = reshape(vsigma_flat, size(total_density))
        flux = ntuple(component -> 2 .* vsigma .* gradient[component], 3)
        potential = reshape(vrho, size(total_density)) .-
                    _divergence_real(flux, reciprocal)
        energy_value = volume * sum(total_density .* reshape(exc, size(total_density))) /
                       length(total_density)
    end
    energy_value, potential
end

function _xc_potential_response(xc::Symbol,
                                valence_density::Array{Float64,3},
                                core_density::Array{Float64,3},
                                delta_density::Array{Float64,3},
                                reciprocal::Matrix{Float64})
    total_density = max.(valence_density .+ core_density, 1e-14)
    rho = vec(total_density)
    delta_rho = vec(delta_density)
    if xc == :lda
        exchange = _libxc_lda_kernel_component(_XC_LDA_X, rho)
        correlation = _libxc_lda_kernel_component(_XC_LDA_C_PZ, rho)
        return reshape((exchange .+ correlation) .* delta_rho,
                       size(total_density))
    end

    gradient = _gradient_real(total_density, reciprocal)
    delta_gradient = _gradient_real(delta_density, reciprocal)
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
    delta_vrho = reshape(
        v2rho2 .* delta_rho .+ v2rhosigma .* delta_sigma,
        size(total_density))
    delta_vsigma = reshape(
        v2rhosigma .* delta_rho .+ v2sigma2 .* delta_sigma,
        size(total_density))
    vsigma = reshape(exchange_vsigma .+ correlation_vsigma,
                     size(total_density))
    flux = ntuple(component ->
        2 .* delta_vsigma .* gradient[component] .+
        2 .* vsigma .* delta_gradient[component], 3)
    delta_vrho .- _divergence_real(flux, reciprocal)
end
