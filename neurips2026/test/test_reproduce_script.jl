# =============================================================================
# test_reproduce_script.jl — the runner bootstrap is VisualGeometryToolkitEval's,
# generated (Reproduce.write_reproduce_script), never a hand copy.
# =============================================================================
using Test
using VisualGeometryToolkitEval.Reproduce: reproduce_script

@testset "scripts/reproduce/reproduce.sh is Eval's generated bootstrap" begin
    @test read(joinpath(@__DIR__, "..", "scripts", "reproduce", "reproduce.sh"), String) ==
          reproduce_script(; julia_project = "neurips2026")
end
