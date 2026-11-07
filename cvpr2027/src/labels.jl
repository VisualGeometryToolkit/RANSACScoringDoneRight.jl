# =============================================================================
# src/labels.jl — THE paper's score vocabulary (single source, as
# LocalOptimizationDoneRight accv2026's src/labels.jl): one key per score, its
# label as the paper's preamble macro (paper/main.tex, "THE score names"), so a
# renamed score changes in one place and every generated table follows. A
# score is named for its mixture: Stu-t (the Gaussian inlier, scale integrated
# out) and its outlier component; -β the mixing parameter integrated out;
# soft- the labels summed out (the table's column group says which).
# =============================================================================

"""
    SCORE_LABEL

Score key => its label (a preamble macro). The keys are
`(estimator kind, outlier model, label prior)` of configs/synthetic.toml;
`"fixed"` and `"calibrated"` are the supplied-width references.
"""
const SCORE_LABEL = Dict(
    ("fixed", "uniform", "half")          => raw"Known $a$",
    ("calibrated", "uniform", "half")     => raw"Cal.\ $a$",
    ("calibrated", "uniform", "marginal") => raw"Cal.\ $a$, $\beta$",
    ("pot", "uniform", "half")            => raw"\stuUev",
    ("pot", "uniform", "marginal")        => raw"\stuUevB",
    ("pot", "gpd", "half")                => raw"\stuGP",
    ("pot", "gpd", "marginal")            => raw"\stuGPB",
    ("soft", "uniform", "half")           => raw"\stuUev",
    ("soft", "uniform", "marginal")       => raw"\stuUevB",
    ("soft", "gpd", "half")               => raw"\stuGP",
    ("soft", "gpd", "marginal")           => raw"\stuGPB",
    ("block_max", "uniform", "half")      => "Block max",
    ("block_min", "uniform", "half")      => "Block min",
)

"""
    score_label(kind, model, prior) -> String

The label of a score; a missing key is an error (no locally spelled label).
"""
score_label(kind::AbstractString, model::AbstractString, prior::AbstractString) =
    get(() -> throw(ArgumentError("no label for score ($kind, $model, $prior): add it to SCORE_LABEL")),
        SCORE_LABEL, (kind, model, prior))

"""
    TIMING_LABEL

Timed scorer name (neurips2026's study names its baselines; configs/
score_timing.toml names ours) => its label. Every row of the timing table is
named here.
"""
const TIMING_LABEL = Dict(
    "Ours (exact)"                   => raw"\stuU",
    "RANSAC (exact)"                 => raw"\ransac",
    "MSAC (exact)"                   => raw"\msac",
    "GaU (exact)"                    => raw"\gau",
    "MAGSAC++ (exact)"               => raw"\magsacpp",
    "O1, gamma 1/2"                  => raw"\stuUev",
    "O1, gamma marg."                => raw"\stuUevB",
    "O2 per prefix, gamma marg."     => raw"\stuGPB, refitted per prefix",
    "O2 per hypothesis, gamma marg." => raw"\stuGPB, fitted per hypothesis",
)

timing_label(name::AbstractString) =
    get(() -> throw(ArgumentError("no label for timed scorer $(repr(name)): add it to TIMING_LABEL")),
        TIMING_LABEL, name)
