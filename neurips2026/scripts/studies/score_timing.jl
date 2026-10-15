#!/usr/bin/env julia
# =============================================================================
# neurips2026/scripts/studies/score_timing.jl — per-hypothesis CPU scoring cost,
# using ONLY the package's own scorers.
#
# THIS FILE CONTAINS NO SCORING CODE. Every form ships in
# src/scoring_eval_common.jl (`best_pool_model` at a loss width) and, for a
# scorer at its width, VisualGeometryCore's generic `best_pool_model` over
# `score!`; this driver only calls them.
#
#   EXACT (closed form, evaluated per residual)
#     best_pool_model(scoring::StudentTUniformScoring, pair, R; ws, scores)  ours — sort-and-sweep
#     best_pool_model(RansacScoring(), pair, R, loss_width)        inlier count
#     best_pool_model(MsacScoring(), pair, R, loss_width)          truncated quadratic
#     best_pool_model(MlesacScoring(), pair, R, loss_width)        softplus (log1p∘exp)
#     best_pool_model(MagsacScoring(), pair, R, loss_width)        two incomplete gammas
#   (`R` is the pair's Float32 residual matrix, from the scene's residuals file)
#   (the four loss-width families are one loop over `residual_quality`; until 2026-09-22
#   they were score_ransac/score_msac(pair, loss_width²)/score_gau/score_magsac and
#   score_marginal, each restating its formula)
#
# NOT REPORTED: approximate forms. The LogBinned histogram aggregation
#   (`grid_evaluate` with `LogBinned()`) is Shekhovtsov 2025 Sec 6.1 Eq. 17, whose stated
#   purpose is amortising one binning scan across a LOSS-WIDTH GRID (its worked
#   example is T=200). It is a BATCH evaluator -- it needs all M hypotheses of a
#   pair before scoring any -- so no online RANSAC can run it, and timing
#   against it would not measure what a deployment pays.
#
# PROTOCOL: 20 pairs, 3 reps, minimum, single thread, per-pair workspaces —
# the published protocol, so these numbers supersede the 2026-07-24 table.
# =============================================================================

using JLD2, Printf
using RansacScoringDoneRight
using VisualGeometryCore.Pools: num_points, num_models
using VisualGeometryEval: cli_option
using VisualGeometryEval.Pools: load_pool, load_pool_residuals, residuals_path, residual_matrix
import VisualGeometryCore as VGC
const RSDR = RansacScoringDoneRight

# The data root: ENV["CAU_DATA_ROOT"] (VisualGeometryEval's data-root variable).
# Needed only for the --pool/--out defaults; give both to run without it.
data_root() = get(ENV, "CAU_DATA_ROOT") do
    error("set CAU_DATA_ROOT to the data root, or give --pool and --out")
end

# The marginal scorer's native parameter is the outlier half-width, NOT a loss
# width. This driver feeds it the same NUMBER as the loss-width methods' loss
# width on purpose — it is a timing harness and the cost is what is being
# measured, not the selection — so the binding is named `outlier_halfwidth` and
# the deliberate reuse is stated rather than hidden behind a `loss_width` at an
# outlier-half-width call site.
#
# Per-pair workspace: deployed sizing, so `sweep!` takes the radix fast path
# (RVG's `_set_n!` resizes to n for exactly this reason).
function marginal_workspaces(pairs, loss_width::Float64)
    outlier_halfwidth = loss_width   # same magnitude, different quantity; timing only
    sc = RSDR.StudentTUniformScoring(1, outlier_halfwidth; codimension = 1)   # the recorded protocol's
    wsc = Dict{Int,Any}()
    for p in pairs
        n = num_points(p)
        get!(() -> (VGC.scoring_workspace(sc, n, 1),
                    Vector{Float64}(undef, n)), wsc, n)
    end
    return sc, wsc
end

# The scorers timed: (name, f(pair, R)). MAGSAC++'s DOF-and-cutoff constants
# are module constants (derived once at load), so its timing is the loop's
# n·M `gamma_inc` pairs.
function timed_scorers(pairs, loss_width::Float64)
    sc, wsc = marginal_workspaces(pairs, loss_width)
    ours(p, R) = (let (w, b) = wsc[num_points(p)]; VGC.best_pool_model(sc, p, R; ws = w, scores = b) end)
    return [
        ("Ours (exact)",     ours),
        ("RANSAC (exact)",   (p, R) -> VGC.best_pool_model(RSDR.RansacScoring(), p, R, loss_width)),
        ("MSAC (exact)",     (p, R) -> VGC.best_pool_model(RSDR.MsacScoring(), p, R, loss_width)),
        ("GaU (exact)",      (p, R) -> VGC.best_pool_model(RSDR.MlesacScoring(), p, R, loss_width)),
        ("MAGSAC++ (exact)", (p, R) -> VGC.best_pool_model(RSDR.MagsacScoring(), p, R, loss_width)),
    ]
end

"""
    time_scorers(subs, loss_width; reps) -> Vector{NamedTuple}

µs per hypothesis of every timed scorer on each `(regime, pairs, mats)` of
`subs`: the minimum over `reps` repetitions, compilation excluded.
"""
function time_scorers(subs, loss_width::Float64; reps::Int)
    results = NamedTuple[]
    for (regime, sub, mats) in subs
        nhyp = sum(num_models(p) for p in sub)
        navg = round(Int, sum(num_points(p) for p in sub) / length(sub))
        println("=== $regime: $(length(sub)) pairs, n̄=$navg points, $nhyp hypotheses ==="); flush(stdout)
        @printf("  %-24s %12s\n", "score", "µs/hyp")
        for (name, f) in timed_scorers(sub, loss_width)
            f(sub[1], mats[1])                    # warmup — compilation excluded
            best = Inf
            for _ in 1:reps
                t0 = time_ns()
                for (p, R) in zip(sub, mats); f(p, R); end
                best = min(best, (time_ns() - t0) / 1e3 / nhyp)
            end
            @printf("  %-24s %12.2f\n", name, best); flush(stdout)
            push!(results, (regime = regime, score = name, us_per_hyp = best,
                            n_avg = navg, n_hyp = nhyp, n_pairs = length(sub)))
        end
        println()
    end
    return results
end

function main(args = ARGS)
    pool_dir   = cli_option(args, "pool", nothing)
    scene      = cli_option(args, "scene", "british_museum")
    npairs     = parse(Int, cli_option(args, "npairs", "20"))
    loss_width = parse(Float64, cli_option(args, "loss-width", "0.00164"))
    reps       = parse(Int, cli_option(args, "reps", "3"))
    out        = cli_option(args, "out", nothing)

    pool_dir === nothing && (pool_dir = joinpath(data_root(), "accv2026", "data", "PT-E-SIFT-4K"))
    out === nothing && (out = joinpath(data_root(), "accv2026", "results", "score_timing"))

    Threads.nthreads() == 1 || error("single-thread BY PROTOCOL — use `julia -t 1`")
    mkpath(out)

    println("loading $scene …"); flush(stdout)
    path = joinpath(pool_dir, "$scene.jld2")
    pool = load_pool(path)
    res  = load_pool_residuals(residuals_path(path), pool)
    pairs = pool.pairs
    println("loaded $(length(pairs)) pairs"); flush(stdout)
    k   = min(npairs, length(pairs))
    byn = sortperm(pairs; by = num_points, rev = true)
    sub(ix) = (pairs[ix], [residual_matrix(res, i) for i in ix])
    subs = [("typical", sub(1:k)...), ("dense", sub(byn[1:k])...)]

    results = time_scorers(subs, loss_width; reps)

    dst = joinpath(out, "score_timing_exact_$(scene).jld2")
    JLD2.jldsave(dst; scene, loss_width, reps, nthreads = Threads.nthreads(), results)
    println("saved -> ", dst); flush(stdout)
    return 0
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
