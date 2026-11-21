# =============================================================================
# The NeurIPS 2026 project's tests. Run from the project (a DrWatson project,
# not a package — no Pkg.test):
#
#   julia --project=. test/runtests.jl
#
#   * the paper-project conventions (VisualGeometryToolkitEval PaperProjects): the
#     layout, the data policy, the REPRODUCE.md asset index against what
#     paper/main.tex reads, configs of record, provenance and the data catalog;
#   * the panel renderers reproduce every submitted figure fragment byte for
#     byte from the stored runs of record, and the prose numbers check against
#     their data (both skipped, loudly, where the stored runs are not bound).
# =============================================================================
using Test
using DrWatson
@quickactivate "NeurIPS2026"
using VisualGeometryToolkitEval.PaperProjects: paper_project_violations

include(joinpath(@__DIR__, "test_paper_project.jl"))
include(joinpath(@__DIR__, "test_figures.jl"))
