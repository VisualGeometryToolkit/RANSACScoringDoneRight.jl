# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code;
# the quick start once used residuals it never defined, and a General review
# asked the stack for this test (#173031, #174496, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package by path, the rest of the
# install line from the registries — with none of the test run's load path.

@testitem "readme" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    import RANSACScoringDoneRight
    # The README's example, run as a reader runs it (VisualGeometryToolkitCore's
    # test_readme_example): one block shown on the landing page, its install line
    # adding exactly what it loads, run in an environment built from that line.
    VgtkCore.test_readme_example(pkgdir(RANSACScoringDoneRight); checks = """
    # The sweep selects the 80 inliers it was given, give or take the
    # outliers that land inside the inlier band, and their scale is the
    # noise scale 0.01.
    @test 75 <= readme.k <= 90
    @test 0.008 < sqrt(readme.s2) < 0.013
    @test isfinite(readme.S)
    """)
end
