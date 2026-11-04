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
    using Test, RANSACScoringDoneRight
    root = pkgdir(RANSACScoringDoneRight)
    julia_blocks(text) = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, text)]
    readme = read(joinpath(root, "README.md"), String)
    index = read(joinpath(root, "docs", "src", "index.md"), String)
    example = julia_blocks(readme)
    landing = julia_blocks(index)
    @test length(example) == 1
    @test landing == example                         # one example, shown twice

    # The install line adds exactly the packages the examples load.
    install = [m[1] for m in eachmatch(r"^pkg> add (.+)$"m, readme)]
    @test length(install) == 1
    packages = split(only(install))
    loaded = Set{String}()
    for code in example, m in eachmatch(r"^(?:using|import)\s+([^:\n]+)"m, code)
        for part in split(m[1], ',')
            push!(loaded, first(split(strip(part), r"[.\s]")))
        end
    end
    stdlib(name) = isdir(joinpath(Sys.STDLIB, name))
    @test Set(filter(!stdlib, loaded)) == Set(packages)
    @test occursin("pkg> add $(only(install))", index)

    env = mktempdir()
    write(joinpath(env, "readme_example.jl"), only(example))
    registered = filter(!=("RANSACScoringDoneRight"), String.(packages))
    script = joinpath(env, "run.jl")
    write(script, """
    using Pkg
    Pkg.develop(path = $(repr(root)); io = devnull)
    Pkg.add($(repr(registered)); io = devnull)
    readme = Module(:ReadmeSandbox)
    # Named for the README, so an error in the example points at it.
    include_string(readme, read($(repr(joinpath(env, "readme_example.jl"))), String), "README.md (example)")
    using Test
    @testset "README example" begin
        # The sweep selects the 80 inliers it was given, give or take the
        # outliers that land inside the inlier band, and their scale is the
        # noise scale 0.01.
        @test 75 <= readme.k <= 90
        @test 0.008 < sqrt(readme.s2) < 0.013
        @test isfinite(readme.S)
    end
    """)
    # Base.julia_cmd() carries the test run's flags (--depwarn=error, coverage).
    # The load path is the project and the standard library only: no default
    # environment (@v1.12) can supply a package the install line does not add.
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$env $script`
    proc = withenv("JULIA_LOAD_PATH" => "@:@stdlib") do   # --project overrides JULIA_PROJECT
        run(ignorestatus(cmd))
    end
    @test proc.exitcode == 0
end
