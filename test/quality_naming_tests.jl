# quality_naming_tests.jl — the stack's iterative-estimator vocabulary
# (VisualGeometryToolkitCore's CLAUDE.md), checked by VisualGeometryToolkitCore's
# `check_naming` on this package: every name it binds, and every line of its
# stack text. The rules and the word lists live in the checker, not here.

@testitem "quality_naming" begin
    using Test
    using VisualGeometryToolkitCore: test_naming
    using RANSACScoringDoneRight
    test_naming(RANSACScoringDoneRight)
end
