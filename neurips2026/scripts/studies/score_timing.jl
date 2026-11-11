#!/usr/bin/env julia
# =============================================================================
# neurips2026/scripts/studies/score_timing.jl — per-hypothesis CPU scoring cost,
# using ONLY the package's own scorers.
#
# THIS FILE CONTAINS NO SCORING CODE. Every scorer is timed through
# VisualGeometryToolkitCore's `best_pool_model(scoring, pair, R; ws, scores)` over
# `score!` of a scorer at its width (the loss-width families at a loss width,
# `with_width(F(1), LossWidth(w))`); this driver only calls it.
#
#   EXACT (closed form, evaluated per residual)
#     StudentTUniformScoring        ours — sort-and-sweep
#     RansacScoring at w            inlier count            (VisualGeometryToolkitCore's score!)
#     MsacScoring at w              truncated quadratic     (VisualGeometryToolkitCore's score!)
#     MlesacScoring at w            softplus (log1p∘exp)    (this package's score!)
#     MagsacScoring at w            two incomplete gammas   (this package's score!)
#   (`R` is the pair's Float32 residual matrix, from the scene's residuals file)
#   (RANSAC's and MSAC's `score!` also builds the consensus order, which the
#   pool argmax does not read: since RANSACScoringDoneRight 0.5 their rows time
#   that too, where the width-free 4-argument `best_pool_model` timed the sum alone)
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
using RANSACScoringDoneRight
using VisualGeometryToolkitCore.Pools: num_points, num_models
using VisualGeometryToolkitEval: cli_option
using VisualGeometryToolkitEval.Experiments: data_root
using VisualGeometryToolkitEval.Pools: load_pool, load_pool_residuals, residuals_path, residual_matrix
import VisualGeometryToolkitCore as VGC
const RSDR = RANSACScoringDoneRight

# The data root is VisualGeometryToolkitEval's `data_root()` (ENV["VGTK_DATA_ROOT"],
# else the project's data/exp_raw). Needed only for the --pool/--out defaults;
# give both to run without it.

# The marginal scorer's native parameter is the outlier half-width, NOT a loss
# width. This driver feeds it the same NUMBER as the loss-width methods' loss
# width on purpose — it is a timing harness and the cost is what is being
# measured, not the selection — so the binding is named `outlier_halfwidth` and
# the deliberate reuse is stated rather than hidden behind a `loss_width` at an
# outlier-half-width call site.
#
# Per-pair workspace: deployed sizing, so `sweep!` takes the radix fast path
# (RVG's `_set_n!` resizes to n for exactly this reason); every scorer gets one,
# so no timed call allocates its scratch.
function timed_selection(sc, pairs)
    wsc = Dict{Int,Any}()
    for p in pairs
        n = num_points(p)
        get!(() -> (VGC.scoring_workspace(sc, n, 1), Vector{Float64}(undef, n)), wsc, n)
    end
    return (p, R) -> (let (w, b) = wsc[num_points(p)]; VGC.best_pool_model(sc, p, R; ws = w, scores = b) end)
end

# The scorers timed: (name, f(pair, R)). MAGSAC++'s DOF-and-cutoff constants
# are module constants (derived once at load), so its timing is the loop's
# n·M `gamma_inc` pairs.
function timed_scorers(pairs, loss_width::Float64)
    outlier_halfwidth = loss_width   # same magnitude, different quantity; timing only
    ours = VGC.StudentTUniformScoring(1, outlier_halfwidth; codimension = 1, scale_prior = VGC.JeffreysPrior(),
                                      label_prior = VGC.FlatLabelPrior())   # the recorded protocol's
    at(F) = VGC.with_width(F(1), VGC.LossWidth(loss_width))
    return [
        ("Ours (exact)",     timed_selection(ours, pairs)),
        ("RANSAC (exact)",   timed_selection(at(VGC.RansacScoring), pairs)),
        ("MSAC (exact)",     timed_selection(at(VGC.MsacScoring), pairs)),
        ("GaU (exact)",      timed_selection(at(RSDR.MlesacScoring), pairs)),
        ("MAGSAC++ (exact)", timed_selection(at(RSDR.MagsacScoring), pairs)),
    ]
end

"""
    time_scorers(subs, scorers; reps) -> Vector{NamedTuple}

µs per hypothesis of every timed scorer on each `(regime, pairs, mats)` of
`subs`: the minimum over `reps` repetitions, compilation excluded.
`scorers(pairs)` returns the `(name, f(pair, R))` list to time (this study's
is `timed_scorers(pairs, loss_width)`; another study may hand it its own list
and reuse this harness).
"""
function time_scorers(subs, scorers; reps::Int)
    results = NamedTuple[]
    for (regime, sub, mats) in subs
        nhyp = sum(num_models(p) for p in sub)
        navg = round(Int, sum(num_points(p) for p in sub) / length(sub))
        println("=== $regime: $(length(sub)) pairs, n̄=$navg points, $nhyp hypotheses ==="); flush(stdout)
        @printf("  %-24s %12s\n", "score", "µs/hyp")
        for (name, f) in scorers(sub)
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

"""
    timing_subsets(path, npairs) -> [(regime, pairs, mats)]

The two density regimes of the protocol: the first `npairs` pairs of the pool
at `path` ("typical") and its `npairs` pairs with the most points ("dense"),
with their residual matrices — read for those pairs only, so a 95 GB
residuals file costs what the timed pairs hold.
"""
function timing_subsets(path::AbstractString, npairs::Int)
    pool = load_pool(path)
    pairs = pool.pairs
    println("loaded $(length(pairs)) pairs"); flush(stdout)
    k   = min(npairs, length(pairs))
    byn = sortperm(pairs; by = num_points, rev = true)
    typical, dense = collect(1:k), byn[1:k]
    res = load_pool_residuals(residuals_path(path), pool; pairs = sort(unique(vcat(typical, dense))))
    sub(ix) = (pairs[ix], [residual_matrix(res, i) for i in ix])
    return [("typical", sub(typical)...), ("dense", sub(dense)...)]
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
    subs = timing_subsets(path, npairs)
    results = time_scorers(subs, sub -> timed_scorers(sub, loss_width); reps)

    dst = joinpath(out, "score_timing_exact_$(scene).jld2")
    JLD2.jldsave(dst; scene, loss_width, reps, nthreads = Threads.nthreads(), results)
    println("saved -> ", dst); flush(stdout)
    return 0
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
