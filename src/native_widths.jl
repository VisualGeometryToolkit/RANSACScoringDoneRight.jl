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
