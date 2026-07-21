struct QEInput
    _model::KSModel
    _basis::PlaneWaveBasis
    _options::SCFOptions
end

function Base.getproperty(input::QEInput, name::Symbol)
    name === :model && return getfield(input, :_model)
    name === :basis && return getfield(input, :_basis)
    name === :options && return getfield(input, :_options)
    getfield(input, name)
end

const _QE_ALLOWED_NAMELISTS = Set(["control", "system", "electrons"])
const _QE_ALLOWED_KEYS = Dict(
    "control" => Set(["calculation", "prefix", "pseudo_dir", "outdir",
                       "tprnfor", "tstress"]),
    "system" => Set(["ibrav", "nat", "ntyp", "ecutwfc", "ecutrho",
                      "tot_charge", "input_dft", "occupations", "nspin",
                      "celldm(1)", "a"]),
    "electrons" => Set(["conv_thr", "electron_maxstep", "mixing_beta"]),
)

_strip_qe_comment(line::AbstractString) = first(split(line, '!'; limit=2))

function _parse_qe_value(text::AbstractString)
    value = strip(text)
    if (startswith(value, "'") && endswith(value, "'")) ||
       (startswith(value, "\"") && endswith(value, "\""))
        return value[2:end-1]
    end
    normalized = lowercase(value)
    normalized in (".true.", "true", "t") && return true
    normalized in (".false.", "false", "f") && return false
    numeric = replace(value, r"[dD]" => "E")
    try
        return parse(Int, numeric)
    catch
    end
    try
        return parse(Float64, numeric)
    catch
    end
    throw(ArgumentError("malformed QE value: $value"))
end

function _parse_namelists(text::AbstractString)
    namelists = Dict{String,Dict{String,Any}}()
    spans = UnitRange{Int}[]
    for matched in eachmatch(
        r"&([A-Za-z_][A-Za-z0-9_]*)[ \t]*(.*?)^[ \t]*/[ \t]*$"ims,
        text,
    )
        name = lowercase(matched.captures[1])
        name in _QE_ALLOWED_NAMELISTS || throw(ArgumentError(
            "unsupported QE namelist $name"))
        haskey(namelists, name) && throw(ArgumentError("duplicate QE namelist $name"))
        body = join(_strip_qe_comment.(split(matched.captures[2], '\n')), "\n")
        values = Dict{String,Any}()
        assignment = r"([A-Za-z_][A-Za-z0-9_]*(?:\(\s*\d+\s*\))?)\s*=\s*(('[^']*')|(\"[^\"]*\")|([^,\n]+))"i
        consumed = falses(ncodeunits(body))
        for item in eachmatch(assignment, body)
            key = lowercase(replace(strip(item.captures[1]), r"\s+" => ""))
            key in _QE_ALLOWED_KEYS[name] || throw(ArgumentError(
                "unsupported QE keyword $key in &$name"))
            haskey(values, key) && throw(ArgumentError("duplicate QE keyword $key"))
            values[key] = _parse_qe_value(item.captures[2])
            for index in item.offset:(item.offset + ncodeunits(item.match) - 1)
                consumed[index] = true
            end
        end
        remainder = String([Char(codeunit(body, i)) for i in eachindex(consumed)
                            if !consumed[i]])
        remainder = replace(remainder, ',' => ' ')
        isempty(strip(remainder)) || throw(ArgumentError(
            "malformed or unsupported QE input in &$name: $(strip(remainder))"))
        namelists[name] = values
        push!(spans, matched.offset:(matched.offset + ncodeunits(matched.match) - 1))
    end
    Set(keys(namelists)) == _QE_ALLOWED_NAMELISTS || throw(ArgumentError(
        "QE input must contain CONTROL, SYSTEM, and ELECTRONS namelists"))
    card_text = collect(codeunits(text))
    for span in spans
        card_text[span] .= UInt8(' ')
    end
    namelists, String(card_text)
end

function _required_qe(values::AbstractDict, key::AbstractString, namelist::AbstractString)
    haskey(values, key) || throw(ArgumentError("missing QE keyword $key in &$namelist"))
    values[key]
end

function _as_int(value, field::AbstractString)
    value isa Integer && return Int(value)
    value isa Real && isinteger(value) && return Int(round(value))
    throw(ArgumentError("QE field $field must be an integer"))
end

function _as_float(value, field::AbstractString)
    value isa Real && isfinite(value) && return Float64(value)
    throw(ArgumentError("QE field $field must be a finite number"))
end

function _normalize_qe_string(value, field::AbstractString)
    value isa AbstractString || throw(ArgumentError("QE field $field must be a string"))
    lowercase(strip(value))
end

function _parse_card_header(line::AbstractString)
    matched = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s+([A-Za-z]+))?\s*$", line)
    matched === nothing && return nothing
    (lowercase(matched.captures[1]),
     matched.captures[2] === nothing ? "" : lowercase(matched.captures[2]))
end

function _parse_qe_cards(card_text::AbstractString, nat::Int, ntyp::Int)
    lines = [strip(_strip_qe_comment(line)) for line in split(card_text, '\n')]
    filter!(!isempty, lines)
    cards = Dict{String,Any}()
    index = 1
    while index <= length(lines)
        header = _parse_card_header(lines[index])
        header === nothing && throw(ArgumentError("malformed QE card: $(lines[index])"))
        name, qualifier = header
        haskey(cards, name) && throw(ArgumentError("duplicate QE card $name"))
        if name == "atomic_species"
            index + ntyp <= length(lines) || throw(ArgumentError(
                "malformed ATOMIC_SPECIES count"))
            rows = split.(lines[index + 1:index + ntyp])
            all(length(row) == 3 for row in rows) || throw(ArgumentError(
                "malformed ATOMIC_SPECIES row"))
            cards[name] = (qualifier=qualifier, rows=rows)
            index += ntyp + 1
        elseif name == "cell_parameters"
            index + 3 <= length(lines) || throw(ArgumentError(
                "malformed CELL_PARAMETERS count"))
            rows = split.(lines[index + 1:index + 3])
            all(length(row) == 3 for row in rows) || throw(ArgumentError(
                "malformed CELL_PARAMETERS row"))
            cards[name] = (qualifier=qualifier, rows=rows)
            index += 4
        elseif name == "atomic_positions"
            index + nat <= length(lines) || throw(ArgumentError(
                "nat is inconsistent with ATOMIC_POSITIONS count"))
            rows = split.(lines[index + 1:index + nat])
            all(length(row) == 4 for row in rows) || throw(ArgumentError(
                "nat is inconsistent with ATOMIC_POSITIONS rows"))
            cards[name] = (qualifier=qualifier, rows=rows)
            index += nat + 1
        elseif name == "k_points"
            qualifier == "automatic" || throw(ArgumentError(
                "unsupported K_POINTS qualifier $qualifier"))
            index < length(lines) || throw(ArgumentError("malformed K_POINTS automatic"))
            row = split(lines[index + 1])
            length(row) == 6 || throw(ArgumentError("malformed K_POINTS automatic"))
            cards[name] = (qualifier=qualifier, rows=[row])
            index += 2
        else
            throw(ArgumentError("unsupported QE card $name"))
        end
    end
    required = Set(["atomic_species", "cell_parameters", "atomic_positions", "k_points"])
    Set(keys(cards)) == required || throw(ArgumentError(
        "QE input is missing one or more required cards"))
    cards
end

function _parse_float_token(token::AbstractString, field::AbstractString)
    try
        parse(Float64, replace(token, r"[dD]" => "E"))
    catch
        throw(ArgumentError("malformed numeric value in $field: $token"))
    end
end

function _qe_input(text::AbstractString, base_directory::AbstractString)
    namelists, card_text = _parse_namelists(text)
    control = namelists["control"]
    system = namelists["system"]
    electrons = namelists["electrons"]

    calculation = _normalize_qe_string(
        _required_qe(control, "calculation", "CONTROL"), "calculation")
    calculation == "scf" || throw(ArgumentError(
        "unsupported calculation=$calculation; expected scf"))
    ibrav = _as_int(_required_qe(system, "ibrav", "SYSTEM"), "ibrav")
    ibrav == 0 || throw(ArgumentError("unsupported ibrav=$ibrav; expected 0"))
    nat = _as_int(_required_qe(system, "nat", "SYSTEM"), "nat")
    ntyp = _as_int(_required_qe(system, "ntyp", "SYSTEM"), "ntyp")
    nat > 0 || throw(ArgumentError("nat must be positive"))
    ntyp > 0 || throw(ArgumentError("ntyp must be positive"))
    occupations_qe = _normalize_qe_string(
        get(system, "occupations", "fixed"), "occupations")
    occupations_qe == "fixed" || throw(ArgumentError(
        "unsupported occupations=$occupations_qe"))
    nspin = _as_int(get(system, "nspin", 1), "nspin")
    nspin == 1 || throw(ArgumentError("unsupported nspin=$nspin"))
    charge = _as_float(get(system, "tot_charge", 0), "tot_charge")
    iszero(charge) || throw(ArgumentError("unsupported tot_charge=$charge"))
    input_dft = uppercase(_normalize_qe_string(
        _required_qe(system, "input_dft", "SYSTEM"), "input_dft"))
    xc = input_dft == "LDA" ? :lda : input_dft == "PBE" ? :pbe : throw(
        ArgumentError("unsupported input_dft=$input_dft"))
    ecut_ry = _as_float(_required_qe(system, "ecutwfc", "SYSTEM"), "ecutwfc")
    ecut_ry > 0 || throw(ArgumentError("ecutwfc must be positive"))
    if haskey(system, "ecutrho")
        ecutrho = _as_float(system["ecutrho"], "ecutrho")
        ecutrho >= 4ecut_ry || throw(ArgumentError(
            "ecutrho is insufficient for norm-conserving density"))
    end

    cards = _parse_qe_cards(card_text, nat, ntyp)
    species_rows = cards["atomic_species"].rows
    species_mass = Dict{Symbol,Float64}()
    pseudo_name = Dict{Symbol,String}()
    for row in species_rows
        element = Symbol(row[1])
        haskey(species_mass, element) && throw(ArgumentError(
            "duplicate ATOMIC_SPECIES element $element"))
        mass = _parse_float_token(row[2], "ATOMIC_SPECIES mass")
        mass > 0 || throw(ArgumentError("ATOMIC_SPECIES mass must be positive"))
        species_mass[element] = mass * AMU_TO_ELECTRON_MASS
        pseudo_name[element] = row[3]
    end
    length(species_mass) == ntyp || throw(ArgumentError("ntyp is inconsistent"))

    cell_card = cards["cell_parameters"]
    cell_rows = [_parse_float_token.(row, Ref("CELL_PARAMETERS"))
                 for row in cell_card.rows]
    lattice = reduce(hcat, cell_rows)
    cell_unit = isempty(cell_card.qualifier) ? "alat" : cell_card.qualifier
    if cell_unit == "angstrom"
        lattice .*= ANGSTROM_TO_BOHR
    elseif cell_unit == "alat"
        scale = if haskey(system, "celldm(1)")
            _as_float(system["celldm(1)"], "celldm(1)")
        elseif haskey(system, "a")
            _as_float(system["a"], "A") * ANGSTROM_TO_BOHR
        else
            throw(ArgumentError("CELL_PARAMETERS alat requires celldm(1) or A"))
        end
        lattice .*= scale
    elseif cell_unit != "bohr"
        throw(ArgumentError("unsupported CELL_PARAMETERS unit $cell_unit"))
    end

    position_card = cards["atomic_positions"]
    elements = Symbol[]
    coordinates = zeros(Float64, 3, nat)
    masses = zeros(Float64, nat)
    for (atom, row) in enumerate(position_card.rows)
        element = Symbol(row[1])
        haskey(species_mass, element) || throw(ArgumentError(
            "ATOMIC_POSITIONS species $(lowercase(row[1])) is not declared"))
        push!(elements, element)
        coordinates[:, atom] .= _parse_float_token.(row[2:4], Ref("ATOMIC_POSITIONS"))
        masses[atom] = species_mass[element]
    end
    position_unit = isempty(position_card.qualifier) ? "alat" : position_card.qualifier
    fractional = if position_unit == "crystal"
        coordinates
    elseif position_unit == "bohr"
        lattice \ coordinates
    elseif position_unit == "angstrom"
        lattice \ (coordinates .* ANGSTROM_TO_BOHR)
    elseif position_unit == "alat"
        scale = haskey(system, "celldm(1)") ?
            _as_float(system["celldm(1)"], "celldm(1)") :
            _as_float(_required_qe(system, "a", "SYSTEM"), "A") * ANGSTROM_TO_BOHR
        lattice \ (coordinates .* scale)
    else
        throw(ArgumentError("unsupported ATOMIC_POSITIONS unit $position_unit"))
    end

    pseudo_dir_value = get(control, "pseudo_dir", ".")
    pseudo_dir_value isa AbstractString || throw(ArgumentError("pseudo_dir must be a string"))
    pseudo_dir = isabspath(pseudo_dir_value) ? pseudo_dir_value :
                 normpath(joinpath(base_directory, pseudo_dir_value))
    pseudos = Dict{Symbol,String}()
    for element in keys(species_mass)
        candidate = pseudo_name[element]
        pseudos[element] = isabspath(candidate) ? candidate : joinpath(pseudo_dir, candidate)
        isfile(pseudos[element]) || throw(ArgumentError(
            "pseudopotential file is missing for $element: $(pseudos[element])"))
    end

    krow = cards["k_points"].rows[1]
    kvalues = try
        parse.(Int, krow)
    catch
        throw(ArgumentError("malformed K_POINTS automatic integers"))
    end
    kgrid = (kvalues[1], kvalues[2], kvalues[3])
    all(>(0), kgrid) || throw(ArgumentError("K_POINTS dimensions must be positive"))
    all(iszero, kvalues[4:6]) || throw(ArgumentError(
        "shifted K_POINTS automatic meshes are outside the V0 input subset"))

    crystal = Crystal(lattice, elements, fractional; masses)
    model = KSModel(crystal; pseudopotentials=pseudos, xc, charge, spin=:unpolarized)
    basis = PlaneWaveBasis(model; Ecut=ecut_ry / 2, kgrid)
    conv_thr = _as_float(get(electrons, "conv_thr", 2e-10), "conv_thr") / 2
    maxiter = _as_int(get(electrons, "electron_maxstep", 100), "electron_maxstep")
    options = SCFOptions(energy_tolerance=conv_thr, density_tolerance=1e-8,
                         maxiter=maxiter, extra_bands=4)
    QEInput(model, basis, options)
end

function read_qe_input(io::IO)
    _qe_input(read(io, String), pwd())
end

function read_qe_input(path::AbstractString)
    isfile(path) || throw(ArgumentError("QE input path does not exist: $path"))
    _qe_input(read(path, String), dirname(abspath(path)))
end

run_qe_input(input::QEInput) = ground_state(input.basis; options=input.options)
run_qe_input(path::AbstractString) = run_qe_input(read_qe_input(path))
run_qe_input(io::IO) = run_qe_input(read_qe_input(io))
