# docs_tests.jl — the docstring gates the Documenter build cannot run. Its
# `checkdocs = :public` only sees docstrings that exist, so a public name with
# none, a docstring detached from its definition by a blank or comment line
# (Julia 1.12 makes it a bare string) and a signature line that drifted from
# the methods (a removed keyword) all pass it. The General registration review
# asks for every public name to be documented (#173032).
#
# docs_gates.jl is the julia-docs-pass copy the stack's suites also carry; it
# is not a `*_tests.jl` file, so ReTestItems does not run it.

@testitem "docs" begin
    using Test, RANSACScoringDoneRight
    include(joinpath(@__DIR__, "docs_gates.jl"))
    M = RANSACScoringDoneRight

    # No exception and no backlog: every public name is documented. The first
    # exception brings back VisualGeometryToolkitCore's lists (its
    # test/docs_tests.jl), each entry with its reason, printed by the item.
    @test isempty(undocumented_public(M))
    @test isempty(detached_docstrings(joinpath(pkgdir(M), "src")))
    @test isempty(signature_mismatches(M))
end
