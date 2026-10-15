# =============================================================================
# src/NeurIPS2026.jl — the shared prelude every driver includes after
# `@quickactivate "NeurIPS2026"`: `include(srcdir("NeurIPS2026.jl"))`. It binds
# the project's data trees (VisualGeometryEval Experiments: data_root() for the
# read-only inputs, results_root() for the results), brings in the stage
# runner, the run key and the provenance writer, and includes the project code
# the drivers share: the panel renderers (panels.jl) and the asset stages that
# verify a regenerated paper asset against the submitted one before it may
# replace it (assets.jl), and the statistics behind the prose numbers
# (numbers.jl). No data constant lives here (those are configs/).
# =============================================================================
if !@isdefined(NEURIPS2026_PRELUDE_LOADED)

using DrWatson
using TOML: TOML
using JLD2: JLD2
using DataFrames: DataFrames, DataFrame
using VisualGeometryEval.Experiments: Stage, run_stages, require_input, fresher, run_key,
    set_project_root!, data_root, results_root, write_run_records, write_provenance, run_complete

# data/exp_raw (inputs) and data/exp_pro (results) under this project, unless
# CAU_DATA_ROOT / CAU_RESULTS_ROOT bind them elsewhere (data/README.md).
set_project_root!(projectdir(); create_results = true)

include(srcdir("panels.jl"))
include(srcdir("assets.jl"))
include(srcdir("numbers.jl"))

const NEURIPS2026_PRELUDE_LOADED = true
end # include guard
