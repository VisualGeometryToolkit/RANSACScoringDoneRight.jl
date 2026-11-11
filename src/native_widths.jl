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
VisualGeometryToolkitCore's `SCORING_TAGS` (`"ransac"`, `"msac"`) merged with this
package's — `"mlesac"`, `"magsac"`, `"welsch_cost"`, `"smooth_tls"`, `"geman"`
(the loss-width families), `"student_t_uniform"` ([`StudentTUniformScoring`](@ref))
and `"sim_fit_pp"` ([`SimFitPP`](@extref VisualGeometryToolkitEstimation.SimFitPP), VisualGeometryToolkitEstimation's SIMFIT++). The retired tags `"gau"`, `"welsch"` and
`"simfitpp"` name their successors. [`loss_width_scoring`](@ref) and every config
loader read it; `tag_of(ALL_SCORING_TAGS, x)` is the writer side. It holds Core's
`SCORING_TAGS` and is named apart from it: Core owns that name.
"""
const ALL_SCORING_TAGS = merge(VgtkCore.SCORING_TAGS, TagRegistry(:scoring, "Scoring",
    "mlesac" => MlesacScoring, "magsac" => MagsacScoring, "welsch_cost" => WelschCostScoring,
    "smooth_tls" => SmoothTLSScoring, "geman" => GemanScoring,
    "student_t_uniform" => StudentTUniformScoring, "sim_fit_pp" => SimFitPP;
    retired = ("gau" => "mlesac", "welsch" => "welsch_cost", "simfitpp" => "sim_fit_pp")))

# The scorers' TOML tables (VisualGeometryToolkitEstimation's `scoring_from_toml`, the local optimization's schema):
# a loss-width family at its `threshold`, the Student-t scorer at its `outlier_halfwidth`, each key required, read
# through `ALL_SCORING_TAGS` (`scoring_from_toml(table; problem, tags = ALL_SCORING_TAGS)`).
const _LossWidthFamily = Union{MlesacScoring, MagsacScoring, WelschCostScoring, SmoothTLSScoring, GemanScoring}
function VgtkEstimation.scoring_from_toml(::Type{T}, d::AbstractDict; problem::Type, section::AbstractString) where {T <: _LossWidthFamily}
    VgtkCore.check_keys(d, ("type", "threshold"), section)
    haskey(d, "threshold") || throw(ArgumentError("[$section] needs `threshold` (a scorer's width is its config's)"))
    return T(codimension(problem); threshold = Float64(d["threshold"]))
end
function VgtkEstimation.scoring_from_toml(::Type{<:StudentTUniformScoring}, d::AbstractDict; problem::Type, section::AbstractString)
    VgtkCore.check_keys(d, ("type", "outlier_halfwidth"), section)
    haskey(d, "outlier_halfwidth") || throw(ArgumentError("[$section] needs `outlier_halfwidth` (a scorer's width is its config's)"))
    return StudentTUniformScoring(problem, Float64(d["outlier_halfwidth"]))
end
VgtkEstimation.config_dict(sc::Union{MlesacScoring{<:Any, Float64}, MagsacScoring{<:Any, Float64}, WelschCostScoring{<:Any, Float64},
                                     SmoothTLSScoring{<:Any, Float64}, GemanScoring{<:Any, Float64}}) =
    Dict{String, Any}("type" => VgtkCore.tag_of(ALL_SCORING_TAGS, sc), "threshold" => sc.threshold)
VgtkEstimation.config_dict(sc::StudentTUniformScoring{P, DG, L, FixedHalfwidth}) where {P, DG, L} =
    Dict{String, Any}("type" => VgtkCore.tag_of(ALL_SCORING_TAGS, sc), "outlier_halfwidth" => outlier_halfwidth(sc))
