# =============================================================================
# sample_boundaries_tests.jl — ONE owner of minimal-sample boundaries:
# VisualGeometryToolkitEval `Pools.sample_starts` (certified against the pool's
# `total_samples` stamp). A "new sample where the `sample_indices` column
# changes" rule merges repeated draws — neurips2026/sample_budget_miscal.jl's
# own `sample_ids` did, until 2026-10-08 — so no code here compares adjacent
# sample columns: the package source and the research experiments.
# =============================================================================

@testitem "sample_boundaries" begin
    using Test
    using RANSACScoringDoneRight
    adjacent = r"(sample_indices|sidx)[^\n]*\b[jk]\s*[-+]\s*1\b"
    root = pkgdir(RANSACScoringDoneRight)
    offenders = String[]
    dirs = filter(isdir, [joinpath(root, d) for d in ("src", "neurips2026")])
    for dir in dirs, (d, _, fs) in walkdir(dir), f in fs
        endswith(f, ".jl") || continue
        for (i, line) in enumerate(eachline(joinpath(d, f)))
            occursin(adjacent, first(split(line, '#'; limit = 2))) && push!(offenders, "$(relpath(joinpath(d, f), root)):$i")
        end
    end
    @test isempty(offenders) || (@info "ad hoc sample boundaries" offenders; false)
end
