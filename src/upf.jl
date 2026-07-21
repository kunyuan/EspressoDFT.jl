struct UPFData
    path::String
    element::Symbol
    z_valence::Float64
    functional::String
    pseudo_type::String
    relativistic::String
    core_correction::Bool
    raw::String
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
    UPFData(abspath(path), Symbol(strip(element_text)), z_valence, functional,
            pseudo_type, relativistic, core, raw)
end

function _functional_matches(upf::UPFData, xc::Symbol)
    f = upf.functional
    xc == :pbe && return occursin("PBE", f)
    xc == :lda && return occursin("LDA", f) || occursin("SLA", f) ||
                         occursin("PZ", f) || occursin("PW", f)
    false
end
