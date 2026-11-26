# =============================================================================
# config_defaults.jl — THE default mechanism of RANSACScoringDoneRight:
# configs/default.toml, read through `_config_default` (as LocalOptimizationDoneRight's
# and VisualGeometryToolkitEstimation's) and its versioned converter `RSDR_CONFIG_VERSIONS`.
# The lookup is VisualGeometryToolkitCore's `toml_default`, which throws on a missing
# table or key by design. Parsed at precompile by Core's `parse_defaults`, which also records the file as a
# precompilation dependency, so an edit forces a recompile (a Revise session
# needs a restart to see one). Included FIRST: the constants and keyword
# defaults of every later file read it.
# =============================================================================

"The defaults of record (configs/default.toml)."
const DEFAULTS_PATH = normpath(joinpath(@__DIR__, "..", "configs", "default.toml"))

"""
    RSDR_CONFIG_VERSIONS

RANSACScoringDoneRight's configuration converter (VisualGeometryToolkitCore's `ConfigVersions`): `configs/default.toml` is version 2. Version 1
(0.8 and earlier, no `config_version` key) is the same tables; its one rename is the owner's (Coles 2001, §4.3):
`[tail_rules] threshold_quantile` is `pot_threshold_quantile`, the quantile level of the complement's residuals at
which the peaks-over-threshold threshold u is set. A current file that spells the old key is refused, naming the new
one. Read a stored file at the current version with
`VisualGeometryToolkitCore.migrate_config(TOML.parsefile(path), RSDR_CONFIG_VERSIONS)`.
"""
const RSDR_CONFIG_VERSIONS = ConfigVersions("RANSACScoringDoneRight",
    ConfigMigration(; tables = ["tail_rules", "scale_marginal", "inlier_potentials"],
        renames = ["tail_rules.threshold_quantile" => "tail_rules.pot_threshold_quantile"]);
    tables = ["tail_rules", "scale_marginal", "inlier_potentials"])

const _DEFAULTS_OF_RECORD = parse_defaults(DEFAULTS_PATH; versions = RSDR_CONFIG_VERSIONS)

# The retired keyword (owner's ruling): TailRules' `threshold_quantile` is refused, naming its successor. Its TOML key
# is not refused but migrated, by the converter above.
const _RETIRED_TAIL_KEY = "`threshold_quantile` is retired: it is `pot_threshold_quantile` (the quantile level of the " *
                          "complement's residuals at which the POT threshold u is set; Coles 2001, §4.3)"

"""
    _config_default(table, key)

The default `key` of `[table]` (a dotted path, `"inlier_potentials.depth"`) in
configs/default.toml. Throws naming a missing table or key.
"""
_config_default(table::AbstractString, key::AbstractString) = toml_default(_DEFAULTS_OF_RECORD, table, key)
