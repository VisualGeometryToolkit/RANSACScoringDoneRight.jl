# =============================================================================
# CVPR2027's tests. The first is the paper-project conventions (VisualGeometryToolkitEval
# PaperProjects): layout, data policy, the REPRODUCE.md index, configs of
# record, provenance and the data catalog. Add the project's own tests below.
#
#   julia --project=. test/runtests.jl
# =============================================================================
using Test
using DrWatson
@quickactivate "CVPR2027"
using VisualGeometryToolkitEval.PaperProjects: paper_project_violations
include(srcdir("CVPR2027.jl"))

@testset "CVPR2027" begin
    @testset "paper-project conventions" begin
        v = paper_project_violations(projectdir(); papers = ["paper/main.tex"])
        foreach(x -> @info(sprint(show, x)), v)
        @test isempty(v)
    end
    include(joinpath(@__DIR__, "simulation_tests.jl"))
    include(joinpath(@__DIR__, "subtree_tests.jl"))
end
