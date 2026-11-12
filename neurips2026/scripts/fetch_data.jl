#!/usr/bin/env julia
# =============================================================================
# scripts/fetch_data.jl — fetch the paper's published data (Zenodo, as the lazy
# Julia artifacts of neurips2026/Artifacts.toml) and link it where the drivers
# read it. The new-machine setup step (REPRODUCE.md, "On a new machine"); a
# no-op where the data are already bound.
#
#   julia --project=. scripts/fetch_data.jl              # the required artifacts
#   julia --project=. scripts/fetch_data.jl --all        # every catalogued artifact
#   julia --project=. scripts/fetch_data.jl --only=heb_raw
#   julia --project=. scripts/fetch_data.jl --list       # the catalog, nothing fetched
#
# The command line, the fetch and the report are VisualGeometryToolkitEval
# DataArtifacts' `fetch_data!` (every paper project's one fetch driver; this
# script's own copy is deleted, 2026-11-12): each artifact is downloaded once
# into the Julia depot (sha256 and tree hash verified) and linked file by file
# into its mount root, an existing binding (VGTK_DATA_ROOT / VGTK_RESULTS_ROOT,
# or the author's data/exp_raw, data/exp_pro links) winning and never written to.
# =============================================================================
using DrWatson
@quickactivate "NeurIPS2026"
include(srcdir("NeurIPS2026.jl"))
include(srcdir("data_artifacts.jl"))

fetch_data!(project_catalog())
