# quality_naming_tests.jl — the stack's iterative-estimator vocabulary
# (VisualGeometryToolkitCore's CLAUDE.md), checked by VisualGeometryToolkitCore's
# `check_naming` on this package: every name it binds, and every line of its
# stack text. The rules and the word lists live in the checker, not here.

@testitem "quality_naming" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using RANSACScoringDoneRight
    # The paper project neurips2026/ is out of this gate: it reproduces with its own pinned environment and keeps that
    # release's vocabulary; its migration is the planned paper re-run (LocalOptimizationDoneRight's accv2026 rule).
    found = filter(v -> !startswith(v.file, "neurips2026/"), VgtkCore.check_naming(RANSACScoringDoneRight))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end

@testitem "quality: no duplication" begin
    # One implementation per concept (VisualGeometryToolkitCore's check_duplication): names
    # an upstream stack package owns, restated function bodies, copied files and retired
    # patterns; the other stack repositories in VGTK_STACK_PATHS are compared too.
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using RANSACScoringDoneRight
    VgtkCore.test_duplication(RANSACScoringDoneRight)
end
