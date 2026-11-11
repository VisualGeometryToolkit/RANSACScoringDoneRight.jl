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
#   from ONE scan: VisualGeometryToolkitEval's sample-budget engine
#   (`evaluate_sample_budget`, every column selection only: the Jeffreys marginal a
#   `SampleBudgetMethodSpec`, each loss-width family a `LossWidthSpec` under
#   `NoLocalOptimization`) folds every model into a running best and records its
#   pose error at every B. The statistic of record is that engine's `model_imc`
#   of the selected model (until RANSACScoringDoneRight 0.5 this study recorded
#   the pool's stored `pool_imc` from its own loop). B is in MINIMAL SAMPLES —
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
#       --pool=$VGTK_DATA_ROOT/accv2026/data/PT-E-SIFT-4K_pf3 \
#       --ranges=$VGTK_DATA_ROOT/accv2026/data/PT-E-SIFT-4K_pf3/coarse_ranges_hierarchical.jld2 \
#       [--pool-samples=<dir of <scene>.samples.jld2>] \
#       --out=$VGTK_DATA_ROOT/accv2026/results/sample_budget_miscal \
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
using RANSACScoringDoneRight: ALL_SCORING_TAGS
using VisualGeometryToolkitCore: width_axis, JeffreysPrior, FlatLabelPrior, StudentTUniformScoring, retired_tags
using VisualGeometryToolkitCore.Pools: PoolPair, num_models
using VisualGeometryToolkitCore.Synthetic: mean_average_accuracy
using VisualGeometryToolkitEval: cli_option
using VisualGeometryToolkitEval.Stats: maa_threshold
using VisualGeometryToolkitEval.Pools: scene_loaders, residual_matrix, sample_starts
using VisualGeometryToolkitEval.Tuning: corrupt_native
using VisualGeometryToolkitEval.Evaluation: SampleBudgetConfig, SampleBudgetMethodSpec, LossWidthSpec,
    evaluate_sample_budget
using VisualGeometryToolkitEstimation: EssentialProblem, NoLocalOptimization

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

# The scoring tag of a roster/sidecar key: the sidecar predates the tag rule
# and spells the Gaussian-uniform family by its retired tag `gau`, which
# RANSACScoringDoneRight's `ALL_SCORING_TAGS` records as renamed `mlesac` — read from
# the registry, never restated here.
family_tag(key::Symbol) = get(Dict(retired_tags(ALL_SCORING_TAGS)), String(key), String(key))

# The width-free scoring of a loss-width roster key (`loss_width_scoring` throws
# for an unknown one), and the Jeffreys marginal at its outlier half-width.
loss_width_family(key::Symbol, problem_type::Type) = loss_width_scoring(family_tag(key), problem_type)
jeffreys_scoring(problem_type::Type, outlier_halfwidth::Float64) =
    StudentTUniformScoring(problem_type, outlier_halfwidth; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
scoring_for(key::Symbol, problem_type::Type, native::Float64) =
    key === :jeffreys ? jeffreys_scoring(problem_type, native) : loss_width_family(key, problem_type)

# (The study's own per-model score vectors, running best over sample prefixes
#  and contract check against the package's pool scorers are deleted: the
#  sample-budget engine of VisualGeometryToolkitEval is that loop, over the
#  scorers' own `score!`.)

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
# One scene's curves
# -----------------------------------------------------------------------------
# The engine column of combo `(key, ζ)`.
combo_name(k::Symbol, ζ::Float64) = Symbol(k, "_zeta", ζ)

"""
    scene_curves(pairs, mats, starts, nat, combos, budgets) -> Matrix{Vector{Float64}}

The pose error at every budget of every `(key, ζ)` combo (columns) for every
pair (rows): pair `i` scored on its residual matrix `mats[i]`, by
VisualGeometryToolkitEval's `evaluate_sample_budget` with no local optimization —
the Jeffreys combos as `SampleBudgetMethodSpec`s at their corrupted outlier
half-widths, every loss-width combo a `LossWidthSpec` at its corrupted loss
width. The checkpoints are every multiple of the budgets' common divisor up to
the largest budget, read at `budgets`; each is the running best's `model_imc`.
The engine threads over pairs.
"""
function scene_curves(pairs::AbstractVector{<:PoolPair}, mats::AbstractVector{<:AbstractMatrix{Float32}},
                      starts::AbstractVector{<:AbstractVector{<:Integer}},
                      nat::Dict{Symbol,Float64}, combos, budgets::Vector{Int})
    methods = [SampleBudgetMethodSpec(combo_name(k, ζ), jeffreys_scoring(EssentialProblem, corrupt_native(nat[k], ζ)))
               for (k, ζ) in combos if k === :jeffreys]
    isempty(methods) && error("the sample-budget engine needs a scoring method column: the sidecar has no :jeffreys native")
    specs = [LossWidthSpec(combo_name(k, ζ), loss_width_family(k, EssentialProblem), corrupt_native(nat[k], ζ))
             for (k, ζ) in combos if k !== :jeffreys]
    cap = maximum(budgets)
    cfg = SampleBudgetConfig(; sample_cap = cap, batch_size = reduce(gcd, budgets), lo_every = cap,
                             lo_strategy = NoLocalOptimization(), budget_unit = :samples)
    res = evaluate_sample_budget(EssentialProblem, methods, pairs, mats; loss_width_specs = specs, cfg,
                                 sample_starts = starts)
    at = [findfirst(==(b), res.sample_counts) for b in budgets]
    curves = Array{Vector{Float64}}(undef, length(pairs), length(combos))
    for (ci, (k, ζ)) in enumerate(combos)
        m = findfirst(==(combo_name(k, ζ)), res.methods)
        for i in eachindex(pairs)
            curves[i, ci] = Float64.(res.err_batch[i, m, at])
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
    axis_reported = false
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

        if !axis_reported
            p1 = first(pairs); ns = length(first(starts))
            @info "budget axis (SAMPLES)" budgets n_samples = ns n_models = num_models(p1)
            ns >= budgets[end] ||
                @warn "pool holds fewer samples than the largest budget — the top points repeat the full-pool best" n_samples = ns
            axis_reported = true
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
        maa = [mean_average_accuracy([c[b] for c in curves]; max_error = maa_threshold()) for b in 1:Mb]
        agg[(k, ζ)] = (; median = med, maa10 = maa, n = length(curves))
        for b in 1:Mb
            push!(rows, @sprintf("%s\t%g\t%d\t%.6f\t%.6f\t%d",
                                 String(k), ζ, budgets[b], med[b], maa[b], length(curves)))
        end
    end

    tsv = joinpath(outdir, "scoring_budget_miscal.tsv")
    write(tsv, join(rows, "\n") * "\n")
    jld = joinpath(outdir, "scoring_budget_miscal.jld2")
    JLD2.jldsave(jld; budgets, budget_unit = "samples", statistic = "model_imc", zetas = ζs, criterion, natives = nat,
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
