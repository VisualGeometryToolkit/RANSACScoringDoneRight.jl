# =============================================================================
# src/palette.jl — the colours and marks of the paper's scores (as
# LocalOptimizationDoneRight accv2026's src/palette.jl), for every figure the
# paper draws: VisualGeometryToolkitEval's house palette for the baselines and
# the supplied-width score, never re-typed, and on top of it one hue per
# outlier model, a darker shade for the mixing parameter integrated out, and
# open marks with dashed lines for soft labels — hue = outlier model, shade =
# label prior, fill/line = inference.
# =============================================================================
using VisualGeometryToolkitEval.Render: default_palette

const _HOUSE = Dict(default_palette())

"Score key => colour hex (the keys of SCORE_LABEL, plus the baselines)."
const SCORE_COLORS = Dict(
    "stuU"     => _HOUSE["Jeffreys"],     # the supplied-width profile-marginal score
    "stuUev"   => "b51d3a",               # component I: the red family, darker
    "stuUevB"  => "7a0f24",
    "stuGP"    => "911eb4",               # component II: purple
    "stuGPB"   => "5b1270",
    "RANSAC"   => _HOUSE["RANSAC"],
    "MSAC"     => _HOUSE["MSAC"],
    "GaU"      => _HOUSE["GaU"],
    "MAGSAC"   => _HOUSE["MAGSAC"],
    "SIMFIT"   => "000075",               # navy, as accv2026
    "Oracle"   => _HOUSE["Oracle"],
)

"Inference => (mark, dashed): hard labels filled and solid, soft labels open and dashed."
const INFERENCE_STYLE = Dict("hard" => ("*", false), "soft" => ("o", true))
