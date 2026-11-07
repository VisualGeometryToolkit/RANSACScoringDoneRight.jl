# =============================================================================
# src/CVPR2027.jl — the shared prelude every driver includes after
# `@quickactivate "CVPR2027"`: `include(srcdir("CVPR2027.jl"))`. It binds the
# project's data trees (VisualGeometryToolkitEval Experiments: data_root() for the
# read-only inputs, results_root() for the results) and brings in the stage
# runner, the run key and the provenance writer, and includes the project code
# the drivers share: the one-candidate simulations (simulation.jl), which score
# with RANSACScoringDoneRight's StudentTUniformScoring. No data constant lives here (those are
# configs/).
# =============================================================================
if !@isdefined(CVPR2027_PRELUDE_LOADED)

using DrWatson
using TOML: TOML
using StableRNGs: StableRNG
using VisualGeometryToolkitEval.Experiments: Stage, run_stages, require_input, fresher, run_key,
    set_project_root!, data_root, results_root, write_run_records, write_provenance, run_complete

# data/exp_raw (inputs) and data/exp_pro (results) under this project, unless
# VGTK_DATA_ROOT / VGTK_RESULTS_ROOT bind them elsewhere (data/README.md).
set_project_root!(projectdir(); create_results = true)

include(srcdir("labels.jl"))
include(srcdir("palette.jl"))
include(srcdir("simulation.jl"))

const CVPR2027_PRELUDE_LOADED = true
end # include guard
