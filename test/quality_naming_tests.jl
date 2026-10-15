# quality_naming_tests.jl — the stack's iterative-estimator vocabulary
# (VisualGeometryCore's CLAUDE.md), checked by VisualGeometryCore's
# `check_naming` on this package: every name it binds, and every line of its
# stack text. The rules and the word lists live in the checker, not here.

@testitem "quality_naming" begin
    using Test
    using VisualGeometryCore: test_naming
    using RansacScoringDoneRight
    test_naming(RansacScoringDoneRight)
end
