function _simpson_integral(values::AbstractVector{<:Number},
                           radial_weights::AbstractVector{<:Real})
    length(values) == length(radial_weights) || throw(DimensionMismatch())
    n = length(values)
    n >= 3 || throw(ArgumentError("radial mesh needs at least three points"))
    result = zero(promote_type(eltype(values), Float64))
    for index in 1:n
        coefficient = if index == 1 || index == n
            1 / 3
        elseif iseven(index)
            4 / 3
        else
            2 / 3
        end
        result += coefficient * values[index] * radial_weights[index]
    end
    result
end

function _spherical_bessel(l::Int, x::Real)
    ax = abs(x)
    if ax < 1e-5
        l == 0 && return 1 - x^2 / 6 + x^4 / 120
        l == 1 && return x / 3 - x^3 / 30 + x^5 / 840
        l == 2 && return x^2 / 15 - x^4 / 210
        return x^l / prod(1:2:(2l + 1))
    end
    j0 = sin(x) / x
    l == 0 && return j0
    j1 = sin(x) / x^2 - cos(x) / x
    l == 1 && return j1
    previous, current = j0, j1
    for order in 1:(l - 1)
        previous, current = current, (2order + 1) / x * current - previous
    end
    current
end

function _legendre(l::Int, x::Real)
    l == 0 && return one(x)
    l == 1 && return x
    previous, current = one(x), x
    for order in 1:(l - 1)
        previous, current = current,
            ((2order + 1) * x * current - order * previous) / (order + 1)
    end
    current
end

function _local_radial_transform(upf::UPFData, q::Real)
    r = upf.radial_grid
    # QE's NC-UPF transform evaluates the short-range remainder only through
    # 10 bohr.  Beyond that radius ONCV files contain a nominal Coulomb tail,
    # but decimal serialization leaves tiny V(r)+Z/r cancellation noise whose
    # volume-weighted integral would otherwise shift every band by a constant.
    # Keep an odd number of radial samples for composite Simpson quadrature.
    last_index = something(findlast(<=(10.0), r), length(r))
    iseven(last_index) && (last_index -= 1)
    radial = @view r[1:last_index]
    weights = @view upf.radial_weights[1:last_index]
    short_range = similar(radial)
    for index in eachindex(radial)
        if iszero(radial[index])
            short_range[index] = 0
        else
            # UPF potentials are in Rydberg. The Coulomb tail is -2Z/r Ry,
            # hence -Z/r after conversion to Hartree.
            short_range[index] = upf.local_potential_ry[index] / 2 +
                                 upf.z_valence / radial[index]
        end
    end
    integrand = [iszero(radius) ? 0.0 :
                 radius^2 * short_range[index] * _spherical_bessel(0, q * radius)
                 for (index, radius) in pairs(radial)]
    correction = 4pi * _simpson_integral(integrand, weights)
    iszero(q) ? correction : correction - 4pi * upf.z_valence / q^2
end

function _atomic_density_transform(upf::UPFData, q::Real)
    values = [upf.atomic_density_radial[index] *
              _spherical_bessel(0, q * upf.radial_grid[index])
              for index in eachindex(upf.radial_grid)]
    _simpson_integral(values, upf.radial_weights)
end

function _core_density_transform(upf::UPFData, q::Real)
    upf.core_correction || return 0.0
    values = [4pi * upf.radial_grid[index]^2 * upf.core_density[index] *
              _spherical_bessel(0, q * upf.radial_grid[index])
              for index in eachindex(upf.radial_grid)]
    _simpson_integral(values, upf.radial_weights)
end

function _projector_transform(upf::UPFData, projector::Int, q::Real)
    l = upf.projector_l[projector]
    values = [upf.radial_grid[index] * upf.projectors[projector][index] *
              _spherical_bessel(l, q * upf.radial_grid[index])
              for index in eachindex(upf.radial_grid)]
    # Integrate on the full UPF radial mesh. `cutoff_radius_index` marks the
    # compact support but is not a new quadrature endpoint; treating it as one
    # changes the Simpson weight of the last nonzero projector sample.
    _simpson_integral(values, upf.radial_weights)
end
