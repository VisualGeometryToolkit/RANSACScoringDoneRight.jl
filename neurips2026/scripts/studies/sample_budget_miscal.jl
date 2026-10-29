#!/usr/bin/env julia
# =============================================================================
# neurips2026/scripts/studies/sample_budget_miscal.jl — SCORING-ONLY sample-budget sweep,
# calibrated vs miscalibrated.
#
# THE QUESTION
#   How does each SCORING FUNCTION's model selection improve as the hypothesis
#   budget B grows, and how much of that is destroyed by a mis-estimated noise
#   scale?
#
# WHY IT LIVES IN THIS PACKAGE
#   There is NO local optimization here. The sweep exercises only the selection
#   rule — score every stored hypothesis, keep the running best, read off its
#   pose error — which is exactly this package's subject matter. The existing
#   ACCV budget figure (accv2026, LocalOptimizationDoneRight's budget engine) runs each arm
#   at its product-grid optimum WITH a refiner attached, so scoring and
#   refinement robustness are entangled there. Stripping LO isolates the
#   scoring half of the miscalibration claim.
#
# MISCALIBRATION
#   `corrupt_native(nat, ζ) = ζ·nat` (VisualGeometryToolkitEval.Tuning's, the one
#   miscalibration map of the stack). It is principled for BOTH scorer families because both
#   natives are linear in the noise scale ŝ:
#       loss-width family       : loss_width = scale_multiple·ŝ
#       outlier-half-width family: outlier_halfwidth = ŝ·√(π/2)·exp(scale_multiple²/2)
#                                  (the exp factor depends only on scale_multiple)
#   so a scale mis-estimated by ζ multiplies either native by exactly ζ. We run
#   ζ = 1 (calibrated) and ζ = 32 (miscalibrated).
#
# HYPERPARAMETERS ARE NOT RE-TUNED
#   Every native is read from the deployed coarse→fine CV sidecar
#   (`coarse_ranges_<protocol>.jld2`). The script HARD-ERRORS if any record it
#   consumes carries `boundary_hit` — consuming a clipped operating point
#   violates the interior-argmin invariant.
#
# BUDGET AXIS
#   Because the pool stores every hypothesis's score inputs, all budgets come
#   from ONE scan: score all M models once, then a running argmax over prefixes
#   gives the selection at every B simultaneously. B is in MINIMAL SAMPLES —
#   the unit a RANSAC trial draws and its termination criterion counts. One
#   sample yields several solver solutions occupying contiguous columns, and a
#   repeated draw repeats them, so the boundaries are VisualGeometryToolkitEval's
#   `sample_starts` (the one reconstruction, certified against the pool's
#   `total_samples` stamp); a pool the record cannot certify (PT-E-SIFT-4K_pf3:
#   duplicated E roots) reads the replay-derived starts of `--pool-samples`
#   (accv2026 scripts/tools/derive_pool_samples.jl).
#
# USAGE
#   julia --project=neurips2026 -t auto neurips2026/scripts/studies/sample_budget_miscal.jl \
#       --pool=$CAU_DATA_ROOT/accv2026/data/PT-E-SIFT-4K_pf3 \
#       --ranges=$CAU_DATA_ROOT/accv2026/data/PT-E-SIFT-4K_pf3/coarse_ranges_hierarchical.jld2 \
#       [--pool-samples=<dir of <scene>.samples.jld2>] \
#       --out=$CAU_DATA_ROOT/accv2026/results/sample_budget_miscal \
#       --splits=<VisualGeometryToolkitEval>/configs/splits.toml --split=phototourism \
#       [--zetas=1,32] [--criterion=median] \
#       [--budgets=100,500,1000,...,4000]   # MINIMAL SAMPLES
#       [--max-pairs=N]                     # smoke: first N pairs per scene
#
#   Each scene is the light pool `<scene>.jld2` plus its residuals
#   `<scene>.residuals.jld2`, read with VisualGeometryToolkitEval.Pools.
#
#   The scenes are the TEST side of the registered split `--split` of the
#   stack's one scene-split registry (`--splits`, VisualGeometryToolkitEval
#   configs/splits.toml): the validation side tuned the natives consumed here,
#   so reading budget curves off it would be tuning on the test set.
# =============================================================================

using JLD2, Printf, Statistics, TOML
using RANSACScoringDoneRight
using RANSACScoringDoneRight: alloc_scoring_workspace
# `best_pool_model`: VisualGeometryToolkitCore's generic (a scorer at its width) plus this
# package's 4-argument method (a width-free loss-width family at a loss width).
using VisualGeometryToolkitCore: width_axis, JeffreysPrior, codimension, best_pool_model
using VisualGeometryToolkitCore.Pools: PoolPair, num_points, num_models, pool_imc
using VisualGeometryToolkitCore.Synthetic: mean_average_accuracy
using VisualGeometryToolkitEval: cli_option
using VisualGeometryToolkitEval.Stats: maa_threshold
using VisualGeometryToolkitEval.Pools: scene_loaders, residual_matrix, sample_starts, sample_ids
using VisualGeometryToolkitEval.Tuning: corrupt_native
using VisualGeometryToolkitEstimation: EssentialProblem

# -----------------------------------------------------------------------------
# Scorer roster — the CV sidecar's method keys (without the `_<criterion>`
# suffix). Each key's scoring and native axis are RSDR's: `scoring_for` builds
# the scoring, VisualGeometryToolkitCore's `width_axis` says which native the sidecar must have tuned
# (asserted against the sidecar's `param_name` at load time).
# -----------------------------------------------------------------------------
const ROSTER = (:ransac, :msac, :gau, :magsac, :jeffreys)
# NOTE: Welsch / smooth-TLS / Geman-McClure are RSDR loss-width families but
# have NO tuned native in the 2026-06-11 PT-E-4K probe sidecar. Add them here
# once a probe covering them exists; `load_natives` will then pick them up and
# the sweep needs no other change. Likewise IG-MLE / IG-robust, which
# additionally need a fitted scale prior.

# The scoring of roster key `key` at native `native`: the Jeffreys marginal
# (outlier half-width) for `:jeffreys`, else the loss-width family of that name
# (`loss_width_scoring` throws for an unknown one; the width is a call argument).
# The roster/sidecar key `:gau` is the Gaussian-uniform family, whose
# `loss_width_scoring` tag is `:mlesac` (renamed at the K+ contracts).
family_tag(key::Symbol) = key === :gau ? :mlesac : key
scoring_for(key::Symbol, problem_type::Type, native::Float64) =
    key === :jeffreys ? StudentTUniformScoring(problem_type, native; prior = JeffreysPrior()) :
                        loss_width_scoring(family_tag(key))

# -----------------------------------------------------------------------------
# Per-model score vectors.
#
# The package's `best_pool_model` returns only the arg-best over the whole
# pool; a budget sweep needs the score of EVERY model so the running best can
# be read at each prefix. These are the package's own per-model scores:
# `pool_model_score` for a loss-width family, `score_residuals!` for the
# marginal — no formula is restated here.
#
# `check_roster_against_package` below asserts argmax(model_scores(...)) equals
# the package's own `best_pool_model`, so any future divergence fails loudly
# instead of silently changing what the curves mean.
# -----------------------------------------------------------------------------
function model_scores!(out::Vector{Float64}, problem_type::Type, key::Symbol,
                       pair::PoolPair, R::AbstractMatrix{Float32}, native::Float64)
    n = num_points(pair); M = num_models(pair)
    length(out) == M || resize!(out, M)
    scoring = scoring_for(key, problem_type, native)
    if scoring isa StudentTUniformScoring
        w = alloc_scoring_workspace(n, codimension(problem_type), scoring)
        q = Vector{Float64}(undef, n)
        @inbounds for j in 1:M
            s, _ = score_residuals!(w, q, scoring, view(R, 1:n, j), n;
                                    sample_idx = view(pair.sample_indices, :, j))
            out[j] = isfinite(s) ? s : -Inf
        end
    else
        for j in 1:M
            out[j] = pool_model_score(scoring, pair, R, j, native)
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Running best over prefixes → pose error at every budget.
#
# imc pose error of a hypothesis is RSDR's `pool_imc` over the PRECOMPUTED
# stored errors (`errors_r`, `errors_t`) — no pose decomposition happens here.
# -----------------------------------------------------------------------------
imc_errors(pair::PoolPair) = [pool_imc(pair, j) for j in 1:num_models(pair)]


"""
Selection error of the running-best-by-score model at each SAMPLE budget.

`budgets[b]` counts minimal samples; the recorded entry is the running best
over every model belonging to samples `1:budgets[b]`. Budgets beyond the pool's
sample count carry the final running best.
"""
function errors_at_budgets(scores::Vector{Float64}, imc::Vector{Float64},
                           ids::Vector{Int}, budgets::Vector{Int})
    out = fill(NaN, length(budgets))
    best_j = 0; best_s = -Inf; bi = 1
    @inbounds for j in eachindex(scores)
        # Close out every budget that the preceding samples have completed.
        while bi <= length(budgets) && budgets[bi] < ids[j]
            out[bi] = best_j == 0 ? NaN : imc[best_j]
            bi += 1
        end
        sc = scores[j]
        sc > best_s && (best_s = sc; best_j = j)
    end
    while bi <= length(budgets)
        out[bi] = best_j == 0 ? NaN : imc[best_j]
        bi += 1
    end
    return out
end

# -----------------------------------------------------------------------------
# CV sidecar → tuned natives (with the interior-argmin gate)
# -----------------------------------------------------------------------------
function load_natives(ranges_path::AbstractString, criterion::AbstractString)
    isfile(ranges_path) || error("CV sidecar not found: $ranges_path")
    m = JLD2.load(ranges_path)["ranges"][:methods]
    nat = Dict{Symbol,Float64}()
    for k in ROSTER
        key = Symbol(string(k, "_", criterion))
        haskey(m, key) || continue
        rec = m[key]
        # INTERIOR-ARGMIN GATE — a clipped window means the CV never bracketed
        # the optimum; consuming it would silently report a boundary value.
        get(rec, :boundary_hit, false) &&
            error("$k: CV record $key carries boundary_hit — the window " *
                  "clipped the optimum. Re-run the probe with an extended window; " *
                  "do NOT consume a clipped operating point.")
        pn = get(rec, :param_name, :?)
        axis = width_axis(scoring_for(k, EssentialProblem, 1.0))
        width_axis(pn) == axis ||
            error("$k: sidecar tuned :$pn but the scoring is native on $axis")
        nat[k] = Float64(rec[:x_star])
    end
    isempty(nat) && error("no roster method found in $ranges_path for criterion=$criterion")
    return nat
end

# -----------------------------------------------------------------------------
# CONTRACT: our per-model scores must agree with the package's pool scorers.
# -----------------------------------------------------------------------------
function check_roster_against_package(pair::PoolPair, R::AbstractMatrix{Float32},
                                      nat::Dict{Symbol,Float64})
    buf = Float64[]
    for k in ROSTER
        haskey(nat, k) || continue
        v = nat[k]
        model_scores!(buf, EssentialProblem, k, pair, R, v)
        ours = argmax(buf)
        theirs = k === :jeffreys ?
                 best_pool_model(scoring_for(k, EssentialProblem, v), pair, R) :
                 best_pool_model(loss_width_scoring(family_tag(k)), pair, R, v)
        theirs == 0 && continue                      # no model: nothing to compare
        ours == theirs || error("CONTRACT VIOLATION: $k per-model argmax " *
                                "$ours ≠ package best_pool_model $theirs — the experiment's " *
                                "score has diverged from RSDR's pool scorers")
    end
    return true
end

# -----------------------------------------------------------------------------
# One scene's curves
# -----------------------------------------------------------------------------
"""
    scene_curves(pairs, mats, starts, nat, combos, budgets) -> Matrix{Vector{Float64}}

The pose error at every budget of every `(key, ζ)` combo (columns) for every
pair (rows): pair `i` scored on its residual matrix `mats[i]`. Scoring is
embarrassingly parallel over pairs: each pair owns its own score buffer and
workspace, and writes only its own row.
"""
function scene_curves(pairs::AbstractVector{<:PoolPair}, mats::AbstractVector{<:AbstractMatrix{Float32}},
                      starts::AbstractVector{<:AbstractVector{<:Integer}},
                      nat::Dict{Symbol,Float64}, combos, budgets::Vector{Int})
    P = length(pairs)
    length(mats) == P || throw(DimensionMismatch("$(length(mats)) residual matrices for $P pairs"))
    curves = Array{Vector{Float64}}(undef, P, length(combos))
    Threads.@threads for i in 1:P
        pair = pairs[i]; R = mats[i]
        imc  = imc_errors(pair)
        ids  = sample_ids(starts[i], num_models(pair))
        # Task-local buffer. Deliberately NOT a per-threadid() pool:
        # threadid() is not stable across yields under dynamic scheduling.
        buf  = Float64[]
        for (ci, (k, ζ)) in enumerate(combos)
            model_scores!(buf, EssentialProblem, k, pair, R, corrupt_native(nat[k], ζ))
            curves[i, ci] = errors_at_budgets(buf, imc, ids, budgets)
        end
    end
    return curves
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function main(args = ARGS)
    pool_dir  = cli_option(args, "pool")
    ranges    = cli_option(args, "ranges")
    outdir    = cli_option(args, "out")
    criterion = cli_option(args, "criterion", "median")
    ζs        = parse.(Float64, split(cli_option(args, "zetas", "1,32"), ','))
    # Budget axis in MINIMAL SAMPLES — the unit a RANSAC trial draws.
    budgets   = parse.(Int, split(cli_option(args, "budgets",
                       "100,500,1000,1500,2000,2500,3000,3500,4000"), ','))
    # Smoke runs only: score the first `--max-pairs` pairs of each scene.
    max_pairs = parse(Int, cli_option(args, "max-pairs", "0"))
    # Replay-derived sample starts (`<scene>.samples.jld2`), for a pool whose
    # record does not certify its own boundaries; empty = from the record.
    samples_dir = cli_option(args, "pool-samples", "")

    splits     = cli_option(args, "splits")   # VisualGeometryToolkitEval configs/splits.toml
    split_name = cli_option(args, "split")    # a [splits.<name>] of --splits
    scenes = let reg = get(TOML.parsefile(splits), "splits", Dict{String,Any}())
        haskey(reg, split_name) || error("$splits has no [splits.$split_name] (has: $(sort(collect(keys(reg)))))")
        sort(String.(reg[split_name]["test_scenes"]))
    end

    nat = load_natives(ranges, criterion)
    @info "tuned natives (criterion=$criterion)" nat
    @info "roster" methods = [k for k in ROSTER if haskey(nat, k)] ζs scenes = length(scenes)
    flush(stderr)

    mkpath(outdir)
    contract_checked = false
    results = Dict{Tuple{Symbol,Float64},Vector{Vector{Float64}}}()  # (method, ζ) -> per-pair curves

    # (key, ζ) product, materialised once — the threaded loop writes by INDEX
    # into a preallocated array, so no shared container is mutated concurrently.
    combos = [(k, ζ) for k in ROSTER if haskey(nat, k) for ζ in ζs]

    present = filter(s -> isfile(joinpath(pool_dir, s * ".jld2")) ||
                          (@warn "missing scene, skipped" s; false), scenes)
    for (scene, load) in zip(present, scene_loaders(pool_dir, present; residuals = true))
        # Free the PREVIOUS scene before allocating the next: two scenes
        # resident at once is what OOM-killed job 4896 at 300 GB.
        GC.gc(true)
        pool, res = load()
        pool.problem === :E || error("$scene: a $(pool.problem) pool; this sweep scores E pools")
        P = max_pairs > 0 ? min(max_pairs, length(pool.pairs)) : length(pool.pairs)
        P == 0 && continue
        starts = sample_starts(pool; derived = isempty(samples_dir) ? nothing :
                                              joinpath(samples_dir, "$scene.samples.jld2"))[1:P]
        pairs = pool.pairs[1:P]
        mats  = [residual_matrix(res, i) for i in 1:P]

        # The package-agreement contract is established ONCE, on the first
        # pair, BEFORE any threading.
        if !contract_checked
            p1 = first(pairs); ns = length(first(starts))
            @info "budget axis (SAMPLES)" budgets n_samples = ns n_models = num_models(p1)
            ns >= budgets[end] ||
                @warn "pool holds fewer samples than the largest budget — the top points repeat the full-pool best" n_samples = ns
            check_roster_against_package(p1, first(mats), nat)
            contract_checked = true
            @info "contract OK — per-model scores agree with the package pool scorers"
            flush(stderr)
        end

        curves = scene_curves(pairs, mats, starts, nat, combos, budgets)
        for (ci, key) in enumerate(combos)
            append!(get!(results, key, Vector{Vector{Float64}}()), view(curves, :, ci))
        end
        pool = res = pairs = mats = starts = nothing
        # Report peak RSS after every scene. VmHWM is the high-water mark, so
        # the LAST line of a successful run is the number to size
        # --cpus-per-task from (peak_GB / 8).
        hwm = try
            # paper-project: allow C2 not data: the kernel's per-process status file, read for the peak RSS it reports
            parse(Int, match(r"VmHWM:\s+(\d+)", read("/proc/self/status", String))[1]) / 1024^2
        catch; NaN end
        @info "scene done" scene npairs = P peak_rss_GB = round(hwm; digits = 1)
        flush(stderr)
    end

    # --- aggregate: median error and mAA@10° per (method, ζ, B) ---------------
    rows = String["method\tzeta\tsamples\tmedian_err_deg\tmaa10\tn_pairs"]
    agg = Dict{Tuple{Symbol,Float64},Any}()
    for ((k, ζ), curves) in results
        Mb = length(budgets)
        med = [median(filter(!isnan, [c[b] for c in curves])) for b in 1:Mb]
        maa = [mean_average_accuracy([c[b] for c in curves]; T = maa_threshold()) for b in 1:Mb]
        agg[(k, ζ)] = (; median = med, maa10 = maa, n = length(curves))
        for b in 1:Mb
            push!(rows, @sprintf("%s\t%g\t%d\t%.6f\t%.6f\t%d",
                                 String(k), ζ, budgets[b], med[b], maa[b], length(curves)))
        end
    end

    tsv = joinpath(outdir, "scoring_budget_miscal.tsv")
    write(tsv, join(rows, "\n") * "\n")
    jld = joinpath(outdir, "scoring_budget_miscal.jld2")
    JLD2.jldsave(jld; budgets, budget_unit = "samples", zetas = ζs, criterion, natives = nat,
                 roster = [String(k) for k in ROSTER if haskey(nat, k)],
                 aggregate = agg, pool_dir, pool_samples = samples_dir, ranges_path = ranges, splits, split = split_name, scenes,
                 max_pairs)
    @info "wrote" tsv jld
    flush(stderr)
    return 0
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
