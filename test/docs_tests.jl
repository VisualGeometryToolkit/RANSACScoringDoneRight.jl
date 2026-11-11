# docs_tests.jl — the docstring gates the Documenter build cannot run. Its
# `checkdocs = :public` only sees docstrings that exist, so a public name with
# none, a docstring detached from its definition by a blank or comment line
# (Julia 1.12 makes it a bare string) and a signature line that drifted from
# the methods (a removed keyword) all get through it. The General registration review
# asks for every public name to be documented (#173032).
#
# The gates are VisualGeometryToolkitCore's one `test_docs` (its Test
# extension); this package's copy of them (docs_gates.jl) is deleted
# (2026-11-11).

@testitem "docs" begin
    using Test, RANSACScoringDoneRight
    using VisualGeometryToolkitCore: test_docs

    # No exception and no backlog: every public name is documented. The first
    # exception takes VisualGeometryToolkitCore's test/docs_tests.jl form, each
    # entry with its reason.
    test_docs(RANSACScoringDoneRight)
end
