const BOHR_TO_ANGSTROM = 0.529177210903
const ANGSTROM_TO_BOHR = inv(BOHR_TO_ANGSTROM)
const AMU_TO_ELECTRON_MASS = 1822.888486209
const TWO_PI = 2pi

_finite_real(x) = x isa Real && isfinite(x)

function _require(condition::Bool, message::AbstractString)
    condition || throw(ArgumentError(message))
    nothing
end

_copy_tuple3(x) = (Float64(x[1]), Float64(x[2]), Float64(x[3]))
