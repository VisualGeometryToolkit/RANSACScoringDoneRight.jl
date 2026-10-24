# readme_tests.jl — every ```julia block of the README runs as written. A
# reader's first contact with the package is that code; the quick start once
# used residuals it never defined (General registration review, 2026-10).
# The installation step is a `pkg>` block (```julia-repl), so it is not run:
# the test environment already has every package the README adds.

@testitem "readme" begin
    using Test
    readme = read(joinpath(pkgdir(RANSACScoringDoneRight), "README.md"), String)
    blocks = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, readme)]
    @test !isempty(blocks)
    sandbox = Module(:ReadmeSandbox)
    for (i, code) in enumerate(blocks)
        @test (include_string(sandbox, code, "README.md (julia block $i)"); true)
    end
    # The quick start selects the 80 inliers it builds, give or take the
    # outliers that land inside the inlier band.
    @test 75 <= Core.eval(sandbox, :k) <= 90
end
