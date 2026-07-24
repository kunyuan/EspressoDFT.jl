function synthetic_upf(; core_correction::Bool=true)
    radial_grid = collect(0.0:0.25:1.0)
    radial_weights = fill(0.25, length(radial_grid))
    local_potential = [-20.0; [-4 / radius for radius in radial_grid[2:end]]]
    atomic_density = [0.0, 0.5, 1.0, 0.5, 0.0]
    projector = [0.0, 0.2, 0.4, 0.2, 0.0]
    core_density = core_correction ? [0.10, 0.08, 0.05, 0.02, 0.0] :
                                     zeros(length(radial_grid))
    EspressoDFT.UPFData(
        "synthetic-He.upf",
        :He,
        2.0,
        "PBE",
        "NC",
        "scalar",
        core_correction,
        -1.0,
        radial_grid,
        radial_weights,
        local_potential,
        [0],
        [length(radial_grid)],
        [projector],
        reshape([1.2], 1, 1),
        atomic_density,
        core_density,
        "synthetic unit-test pseudopotential",
    )
end

function synthetic_upf_xml(; core_correction::Bool=true)
    core_flag = core_correction ? "T" : "F"
    nlcc = core_correction ?
        "<PP_NLCC>0.10 0.08 0.05 0.02 0.0</PP_NLCC>" : ""
    """
    <UPF version="2.0.1">
      <PP_HEADER element="He" pseudo_type="NC" is_ultrasoft="F" is_paw="F"
        has_so="F" relativistic="scalar" z_valence="2.0" functional="PBE"
        core_correction="$core_flag" total_psenergy="-1.0" mesh_size="5"
        number_of_proj="1"/>
      <PP_R>0.0 0.25 0.50 0.75 1.00</PP_R>
      <PP_RAB>0.25 0.25 0.25 0.25 0.25</PP_RAB>
      <PP_LOCAL>-20.0 -16.0 -8.0 -5.333333333333333 -4.0</PP_LOCAL>
      <PP_RHOATOM>0.0 0.5 1.0 0.5 0.0</PP_RHOATOM>
      <PP_BETA.1 angular_momentum="0" cutoff_radius_index="5">
        0.0 0.2 0.4 0.2 0.0
      </PP_BETA.1>
      <PP_DIJ>1.2</PP_DIJ>
      $nlcc
    </UPF>
    """
end

function write_synthetic_upf(directory::AbstractString;
                             core_correction::Bool=true)
    path = joinpath(directory, "He.synthetic.upf")
    write(path, synthetic_upf_xml(; core_correction))
    path
end

function synthetic_basis()
    lattice = 8.0 .* Matrix{Float64}(I, 3, 3)
    crystal = Crystal(
        lattice,
        [:He],
        zeros(3, 1);
        masses=[4.0 * EspressoDFT.AMU_TO_ELECTRON_MASS],
    )
    model = EspressoDFT.KSModel(
        crystal,
        Dict(:He => synthetic_upf()),
        2.0,
        :pbe,
    )
    PlaneWaveBasis(model; Ecut=0.8, kgrid=(1, 1, 1))
end

const SYNTHETIC_BASIS = synthetic_basis()
const SYNTHETIC_KERNEL = EspressoDFT._build_kernel(SYNTHETIC_BASIS)

function synthetic_ground_state()
    basis = SYNTHETIC_BASIS
    kernel = SYNTHETIC_KERNEL
    dimension = length(basis.G_vectors[1])
    orbital = zeros(ComplexF64, dimension, 1)
    orbital[1, 1] = 1
    volume = abs(det(basis.model.crystal.lattice))
    GroundState = EspressoDFT.GroundState
    GroundState(
        basis,
        SCFOptions(),
        kernel,
        true,
        -1.25,
        reshape([0.1, -0.2, 0.3], 3, 1),
        true,
        [0.01 0.002 0.0; 0.002 -0.02 0.001; 0.0 0.001 0.03],
        true,
        fill(2 / volume, basis.fft_size),
        [[-0.5]],
        [[2.0]],
        [orbital],
        1e-12,
        1e-10,
        1,
        [-1.25],
        [1e-10],
    )
end

function synthetic_qe_input(pseudopotential_path::AbstractString;
                            k_shift::NTuple{3,Int}=(0, 0, 0))
    pseudo_dir = dirname(abspath(pseudopotential_path))
    pseudo_name = basename(pseudopotential_path)
    """
    &CONTROL
      calculation = 'scf',
      prefix = 'synthetic',
      pseudo_dir = '$pseudo_dir',
      tprnfor = .true.,
      tstress = .true.
    /
    &SYSTEM
      ibrav = 0,
      nat = 1,
      ntyp = 1,
      ecutwfc = 1.6d0,
      ecutrho = 6.4d0,
      occupations = 'fixed',
      nspin = 1,
      input_dft = 'PBE'
    /
    &ELECTRONS
      conv_thr = 2.0d-10,
      electron_maxstep = 25
    /
    ATOMIC_SPECIES
    He 4.0 $pseudo_name
    CELL_PARAMETERS bohr
    8.0 0.0 0.0
    0.0 8.0 0.0
    0.0 0.0 8.0
    ATOMIC_POSITIONS crystal
    He 0.0 0.0 0.0
    K_POINTS automatic
    1 1 1 $(join(k_shift, " "))
    """
end
