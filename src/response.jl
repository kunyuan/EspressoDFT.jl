function _validate_response(gs::GroundState, perturbation::AtomicDisplacement)
    _require_converged(gs)
    nat = length(getfield(getfield(getfield(gs.basis, :_model), :_crystal), :_species))
    _require(perturbation.atom <= nat,
             "atom index $(perturbation.atom) exceeds crystal size $nat")
    _require(_is_commensurate_q(gs.basis, perturbation.q),
             "q=$(perturbation.q) is not commensurate with the electronic k mesh")
end

function response(gs::GroundState, perturbation::AtomicDisplacement;
                  tolerance::Real=1e-8, maxiter::Integer=200)
    _validate_response(gs, perturbation)
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    maxiter == 1 && error("response did not converge within maxiter=1")
    error("response kernel is not yet implemented")
end

function dynamical_matrix(gs::GroundState, q::NTuple{3,<:Real};
                          tolerance::Real=1e-8, maxiter::Integer=200)
    _require(all(isfinite, q), "q must contain only finite values")
    _require(_is_commensurate_q(gs.basis, q),
             "q=$q is not commensurate with the electronic k mesh")
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    error("dynamical-matrix kernel is not yet implemented")
end

function phonon_modes(gs::GroundState, q::NTuple{3,<:Real};
                      tolerance::Real=1e-8, maxiter::Integer=200)
    matrix = dynamical_matrix(gs, q; tolerance, maxiter)
    decomposition = eigen(Hermitian(matrix))
    frequencies = sign.(decomposition.values) .* sqrt.(abs.(decomposition.values))
    (frequencies=frequencies, eigenvectors=decomposition.vectors)
end

function born_effective_charges(::GroundState; tolerance::Real=1e-8,
                                maxiter::Integer=200)
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    error("Born effective-charge kernel is not yet implemented")
end

function dielectric_tensor(::GroundState; tolerance::Real=1e-8,
                           maxiter::Integer=200)
    _require(_finite_real(tolerance) && tolerance > 0,
             "response tolerance must be finite and positive")
    _require(maxiter > 0, "response maxiter must be positive")
    error("dielectric-response kernel is not yet implemented")
end
