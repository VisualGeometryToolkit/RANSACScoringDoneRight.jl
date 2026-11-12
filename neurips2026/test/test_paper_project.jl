# =============================================================================
# test_paper_project.jl — the stack's paper-project conventions
# (VisualGeometryToolkitEval `PaperProjects.paper_project_violations`): the layout,
# the data policy (D1–D4), the REPRODUCE.md asset index against what
# paper/main.tex reads, configs of record, provenance and the data catalog.
# Opt-outs with their reasons sit in the files they excuse
# (`# paper-project: allow <RULE> <reason>`); no violation is deferred: any
# violation fails.
# =============================================================================
using Test
using VisualGeometryToolkitEval.PaperProjects: paper_project_violations

@testset "paper-project conventions" begin
    v = paper_project_violations(normpath(joinpath(@__DIR__, "..")); papers = ["paper/main.tex"])
    foreach(x -> @info("paper-project violation", x), v)
    @test isempty(v)
end
