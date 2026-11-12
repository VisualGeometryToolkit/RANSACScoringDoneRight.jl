# =============================================================================
# native_widths.jl — the scorer fact a grid evaluation reads: the width axis a
# scoring's grid is native on (VisualGeometryToolkitCore's scoring-contract verb
# `width_axis(scoring)`). The scale multiple → native maps (`to_native`,
# `native_map`) and the native-axis conversions are VisualGeometryToolkitCore's.
# =============================================================================

# The width axis of each grid family of this package:
#
# | Scoring | Native axis |
# |---|---|
# | `StudentTUniformScoring` (Student-t marginal; scoring.jl) | `OutlierHalfwidth` |
# | `LossSpec` (the mixture marginal of a redescending loss) | `OutlierHalfwidth` |
# | every `LossWidthScoring` (RANSAC, MSAC, MLESAC, MAGSAC++, the losses) | `LossWidth` |
# | `ScaleMarginal` (the full-data scale marginals; grid of IG prior centres) | `LossWidth` |
# | `SimFitPP` (grid of initial loss-width guesses) | `LossWidth` |
#
# (RANSAC and MSAC are VisualGeometryToolkitCore's scorers, which state their own.)
width_axis(::LossSpec) = OutlierHalfwidth()
width_axis(::Union{MlesacScoring, MagsacScoring, WelschCostScoring, SmoothTLSScoring, GemanScoring}) = LossWidth()
width_axis(::ScaleMarginal) = LossWidth()
width_axis(::SimFitPP) = LossWidth()

"""
    ALL_SCORING_TAGS

The scoring families' [`TagRegistry`](@extref VisualGeometryToolkitCore.TagRegistry):
VisualGeometryToolkitCore's `SCORING_TAGS` (`"ransac"`, `"msac"`, `"student_t_uniform"`,
VisualGeometryToolkitCore's `StudentTUniformScoring`) merged with this
package's — `"mlesac"`, `"magsac"`, `"welsch_cost"`, `"smooth_tls"`, `"geman"`
(the loss-width families) and `"sim_fit_pp"` ([`SimFitPP`](@extref VisualGeometryToolkitEstimation.SimFitPP), VisualGeometryToolkitEstimation's SIMFIT++). The retired tags `"gau"`, `"welsch"` and
`"simfitpp"` name their successors. [`loss_width_scoring`](@ref) and every config
loader read it; `tag_of(ALL_SCORING_TAGS, x)` is the writer side. It holds Core's
`SCORING_TAGS` and is named apart from it: Core owns that name.
"""
const ALL_SCORING_TAGS = merge(VgtkCore.SCORING_TAGS, TagRegistry(:scoring, "Scoring",
    "mlesac" => MlesacScoring, "magsac" => MagsacScoring, "welsch_cost" => WelschCostScoring,
    "smooth_tls" => SmoothTLSScoring, "geman" => GemanScoring,
    "sim_fit_pp" => SimFitPP;
    retired = ("gau" => "mlesac", "welsch" => "welsch_cost", "simfitpp" => "sim_fit_pp")))

# The scorers' TOML tables are VisualGeometryToolkitEstimation's: a loss-width family
# (built by Core's `@threshold_scoring_constructors`) at its `threshold`, required,
# read by `scoring_from_toml(table; problem, tags = ALL_SCORING_TAGS)` and written by
# `config_dict(scoring; tags = ALL_SCORING_TAGS)`, Estimation's methods for every
# `AbstractThresholdScoring` (this package's own two are deleted, 2026-11-12).
