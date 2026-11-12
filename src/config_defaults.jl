# =============================================================================
# config_defaults.jl — THE default mechanism of RANSACScoringDoneRight:
# configs/default.toml, read through `_config_default` (as LocalOptimizationDoneRight's
# and VisualGeometryToolkitEstimation's). The lookup is VisualGeometryToolkitCore's
# `toml_default`, which throws on a missing table or key by design. Parsed at
# precompile by Core's `parse_defaults`, which also records the file as a
# precompilation dependency, so an edit forces a recompile (a Revise session
# needs a restart to see one). Included FIRST: the constants and keyword
# defaults of every later file read it.
# =============================================================================

"The defaults of record (configs/default.toml)."
const DEFAULTS_PATH = normpath(joinpath(@__DIR__, "..", "configs", "default.toml"))
const _DEFAULTS_OF_RECORD = parse_defaults(DEFAULTS_PATH)

# RETIRED keys: a key renamed by an owner's ruling is refused, naming its successor, wherever it appears — the
# keyword and the TOML key alike (this package keeps no list of retired names elsewhere).
const _RETIRED_TAIL_KEY = "`threshold_quantile` is retired: it is `pot_threshold_quantile` (the quantile level of the " *
                          "complement's residuals at which the POT threshold u is set; Coles 2001, §4.3)"
const _REFUSED_RENAMED_KEYS = (("tail_rules", "threshold_quantile") => _RETIRED_TAIL_KEY,)

"""
    _check_retired_keys(d::AbstractDict) -> d

Refuse a defaults table that spells a retired key (`ArgumentError` naming its successor).
"""
function _check_retired_keys(d::AbstractDict)
    for ((table, key), msg) in _REFUSED_RENAMED_KEYS
        t = get(d, table, nothing)
        t isa AbstractDict && haskey(t, key) && throw(ArgumentError("[$table] $msg"))
    end
    return d
end
_check_retired_keys(_DEFAULTS_OF_RECORD)

"""
    _config_default(table, key)

The default `key` of `[table]` (a dotted path, `"inlier_potentials.depth"`) in
configs/default.toml. Throws naming a missing table or key.
"""
_config_default(table::AbstractString, key::AbstractString) = toml_default(_DEFAULTS_OF_RECORD, table, key)
