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
    SCORING_TAGS

The scoring families' [`TagRegistry`](@extref VisualGeometryToolkitCore.TagRegistry):
VisualGeometryToolkitCore's `SCORING_TAGS` (`"ransac"`, `"msac"`) merged with this
package's — `"mlesac"`, `"magsac"`, `"welsch_cost"`, `"smooth_tls"`, `"geman"`
(the loss-width families), `"student_t_uniform"` ([`StudentTUniformScoring`](@ref))
and `"sim_fit_pp"` ([`SimFitPP`](@ref)). The retired tags `"gau"`, `"welsch"` and
`"simfitpp"` name their successors. [`loss_width_scoring`](@ref) and every config
loader read it; `tag_of(SCORING_TAGS, x)` is the writer side.
"""
const SCORING_TAGS = merge(VgtkCore.SCORING_TAGS, TagRegistry(:scoring, "Scoring",
    "mlesac" => MlesacScoring, "magsac" => MagsacScoring, "welsch_cost" => WelschCostScoring,
    "smooth_tls" => SmoothTLSScoring, "geman" => GemanScoring,
    "student_t_uniform" => StudentTUniformScoring, "sim_fit_pp" => SimFitPP;
    retired = ("gau" => "mlesac", "welsch" => "welsch_cost", "simfitpp" => "sim_fit_pp")))
