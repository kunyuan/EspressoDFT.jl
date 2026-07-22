struct UPFData
    path::String
    element::Symbol
    z_valence::Float64
    functional::String
    pseudo_type::String
    relativistic::String
    core_correction::Bool
    total_psenergy::Float64
    radial_grid::Vector{Float64}
    radial_weights::Vector{Float64}
    local_potential_ry::Vector{Float64}
    projector_l::Vector{Int}
    projector_cutoff::Vector{Int}
    projectors::Vector{Vector{Float64}}
    dij_ry::Matrix{Float64}
    atomic_density_radial::Vector{Float64}
    core_density::Vector{Float64}
    raw::String
end

function _upf_numbers(text::AbstractString, field::AbstractString)
    try
        parse.(Float64, replace.(split(strip(text)), r"[dD]" => "E"))
    catch
        throw(ArgumentError("malformed numeric data in UPF $field"))
    end
end

function _upf_field(raw::AbstractString, tag::AbstractString; required::Bool=true)
    escaped = replace(tag, "." => "\\.")
    matched = match(Regex("<" * escaped * "(?:\\s[^>]*)?>(.*?)</" * escaped * ">", "s"), raw)
    if matched === nothing
        required && throw(ArgumentError("UPF field $tag is missing"))
        return nothing
    end
    matched.captures[1]
end

function _xml_attribute(tag::AbstractString, name::AbstractString)
    matched = match(Regex("\\b" * name * "\\s*=\\s*[\"']([^\"']*)[\"']", "i"), tag)
    matched === nothing ? nothing : matched.captures[1]
end

function _parse_bool_attribute(value, field::AbstractString)
    value === nothing && return false
    normalized = uppercase(strip(value))
    normalized in ("T", "TRUE", ".TRUE.") && return true
    normalized in ("F", "FALSE", ".FALSE.") && return false
    throw(ArgumentError("invalid UPF $field attribute: $value"))
end

function _read_upf(path::AbstractString)
    isfile(path) || throw(ArgumentError("pseudopotential file does not exist: $path"))
    raw = read(path, String)
    root = match(r"<UPF\s+[^>]*>"i, raw)
    root === nothing && throw(ArgumentError("pseudopotential is not a UPF file: $path"))
    version = _xml_attribute(root.match, "version")
    shown_version = something(version, "missing")
    version == "2.0.1" || throw(ArgumentError(
        "unsupported UPF version $shown_version; expected 2.0.1"))

    header = match(r"<PP_HEADER\s+[^>]*\/?>"is, raw)
    header === nothing && throw(ArgumentError("UPF file has no PP_HEADER: $path"))
    tag = header.match
    element_text = _xml_attribute(tag, "element")
    element_text === nothing && throw(ArgumentError("UPF element metadata is missing: $path"))
    pseudo_type = uppercase(strip(something(_xml_attribute(tag, "pseudo_type"), "")))
    pseudo_type == "NC" || throw(ArgumentError(
        "unsupported pseudopotential type $pseudo_type; only norm-conserving NC is supported"))
    _parse_bool_attribute(_xml_attribute(tag, "is_ultrasoft"), "is_ultrasoft") &&
        throw(ArgumentError("unsupported USPP pseudopotential"))
    _parse_bool_attribute(_xml_attribute(tag, "is_paw"), "is_paw") &&
        throw(ArgumentError("unsupported PAW pseudopotential"))
    _parse_bool_attribute(_xml_attribute(tag, "has_so"), "has_so") &&
        throw(ArgumentError("unsupported spin-orbit pseudopotential"))
    relativistic = lowercase(strip(something(_xml_attribute(tag, "relativistic"), "")))
    relativistic in ("scalar", "scalar-relativistic") || throw(ArgumentError(
        "unsupported relativistic mode $relativistic; expected scalar"))
    z_text = _xml_attribute(tag, "z_valence")
    z_text === nothing && throw(ArgumentError("UPF z_valence metadata is missing"))
    z_valence = try
        parse(Float64, replace(strip(z_text), r"[dD]" => "E"))
    catch
        throw(ArgumentError("invalid UPF z_valence: $z_text"))
    end
    z_valence > 0 || throw(ArgumentError("UPF z_valence must be positive"))
    functional = uppercase(strip(something(_xml_attribute(tag, "functional"), "")))
    isempty(functional) && throw(ArgumentError("UPF functional metadata is missing"))
    core = _parse_bool_attribute(_xml_attribute(tag, "core_correction"), "core_correction")
    total_psenergy = parse(Float64, replace(strip(something(
        _xml_attribute(tag, "total_psenergy"), "0")), r"[dD]" => "E"))
    mesh_size = parse(Int, strip(something(_xml_attribute(tag, "mesh_size"), "0")))
    nprojectors = parse(Int, strip(something(_xml_attribute(tag, "number_of_proj"), "0")))

    radial_grid = _upf_numbers(_upf_field(raw, "PP_R"), "PP_R")
    radial_weights = _upf_numbers(_upf_field(raw, "PP_RAB"), "PP_RAB")
    local_potential = _upf_numbers(_upf_field(raw, "PP_LOCAL"), "PP_LOCAL")
    rhoatom = _upf_numbers(_upf_field(raw, "PP_RHOATOM"), "PP_RHOATOM")
    all(length(values) == mesh_size for values in
        (radial_grid, radial_weights, local_potential, rhoatom)) ||
        throw(ArgumentError("UPF radial field size does not match mesh_size"))

    projector_l = Int[]
    projector_cutoff = Int[]
    projectors = Vector{Float64}[]
    for beta in eachmatch(r"<PP_BETA\.(\d+)([^>]*)>(.*?)</PP_BETA\.\1>"s, raw)
        attributes = beta.captures[2]
        angular = _xml_attribute(attributes, "angular_momentum")
        angular === nothing && throw(ArgumentError("UPF projector angular_momentum is missing"))
        cutoff_text = something(_xml_attribute(attributes, "cutoff_radius_index"),
                                string(mesh_size))
        values = _upf_numbers(beta.captures[3], "PP_BETA.$(beta.captures[1])")
        length(values) == mesh_size || throw(ArgumentError("UPF projector size mismatch"))
        push!(projector_l, parse(Int, strip(angular)))
        push!(projector_cutoff, parse(Int, strip(cutoff_text)))
        push!(projectors, values)
    end
    length(projectors) == nprojectors || throw(ArgumentError(
        "UPF number_of_proj does not match PP_BETA fields"))
    dij_values = _upf_numbers(_upf_field(raw, "PP_DIJ"), "PP_DIJ")
    length(dij_values) == nprojectors^2 || throw(ArgumentError("UPF PP_DIJ size mismatch"))
    dij = reshape(dij_values, nprojectors, nprojectors)
    isapprox(dij, dij'; atol=1e-12, rtol=1e-12) || throw(ArgumentError(
        "UPF PP_DIJ must be symmetric for scalar norm-conserving input"))
    core_density = core ? _upf_numbers(_upf_field(raw, "PP_NLCC"), "PP_NLCC") :
                          zeros(mesh_size)
    length(core_density) == mesh_size || throw(ArgumentError("UPF PP_NLCC size mismatch"))

    UPFData(abspath(path), Symbol(strip(element_text)), z_valence, functional,
            pseudo_type, relativistic, core, total_psenergy,
            radial_grid, radial_weights, local_potential, projector_l,
            projector_cutoff, projectors, dij, rhoatom, core_density, raw)
end

function _functional_matches(upf::UPFData, xc::Symbol)
    f = upf.functional
    xc == :pbe && return occursin("PBE", f)
    xc == :lda && return occursin("LDA", f) || occursin("SLA", f) ||
                         occursin("PZ", f) || occursin("PW", f)
    false
end
