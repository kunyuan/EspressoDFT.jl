struct GroundState
    basis::PlaneWaveBasis
    options::SCFOptions
    converged::Bool
    total_energy::Float64
    force_values::Matrix{Float64}
    stress_value::Matrix{Float64}
    density_values::Array{Float64,3}
    band_values::Vector{Vector{Float64}}
    occupation_values::Vector{Vector{Float64}}
    residual_energy::Float64
    residual_density::Float64
end

function ground_state(::PlaneWaveBasis; options::SCFOptions=SCFOptions())
    options.maxiter == 1 && error("SCF did not converge within maxiter=1")
    error("SCF kernel is not yet implemented")
end

function _require_converged(gs::GroundState)
    gs.converged || error("ground state did not converge")
    nothing
end

energy(gs::GroundState) = (_require_converged(gs); gs.total_energy)
forces(gs::GroundState) = (_require_converged(gs); copy(gs.force_values))
stress(gs::GroundState) = (_require_converged(gs); copy(gs.stress_value))
density(gs::GroundState) = (_require_converged(gs); (
    values=copy(gs.density_values),
    cell_volume=abs(det(getfield(getfield(getfield(gs.basis, :_model), :_crystal), :_lattice))),
))
eigenvalues(gs::GroundState) = (_require_converged(gs); deepcopy(gs.band_values))
occupations(gs::GroundState) = (_require_converged(gs); deepcopy(gs.occupation_values))
