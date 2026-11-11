# =============================================================================
# config_defaults.jl — THE default mechanism of RANSACScoringDoneRight:
# configs/default.toml, read through `_config_default` (as LocalOptimizationDoneRight's
# and VisualGeometryToolkitEstimation's). The lookup is VisualGeometryToolkitCore's
# `toml_default`, which throws on a missing table or key by design. Parsed at
# precompile; `include_dependency` makes an edit force a recompile (a Revise
# session needs a restart to see one). Included FIRST: the constants and keyword
# defaults of every later file read it.
# =============================================================================

"The defaults of record (configs/default.toml)."
const DEFAULTS_PATH = normpath(joinpath(@__DIR__, "..", "configs", "default.toml"))
include_dependency(DEFAULTS_PATH)
const _DEFAULTS_OF_RECORD = parse_defaults(DEFAULTS_PATH)

"""
    _config_default(table, key)

The default `key` of `[table]` (a dotted path, `"inlier_potentials.depth"`) in
configs/default.toml. Throws naming a missing table or key.
"""
_config_default(table::AbstractString, key::AbstractString) = toml_default(_DEFAULTS_OF_RECORD, table, key)
