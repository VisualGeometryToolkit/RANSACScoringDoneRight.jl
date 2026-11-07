#!/usr/bin/env julia
# provenance: every run writes _config.toml and _provenance.toml (write_run_records) beside its results.
# =============================================================================
# scripts/synthetic.jl — the one-candidate simulations of Sec. 4.1
# (\label{tab:synthetic}): every estimator of the outlier half-width and both
# outlier models on the same draws, the calibrated Gaussian–uniform baseline,
# and the unit-invariance check (configs/synthetic.toml).
#
# reproduce: reads configs/synthetic.toml
# reproduce: writes paper/figures/tab_synthetic.tex
# reproduce: writes paper/figures/synthetic_numbers.tex
#
# STAGES
#   simulate     every case × trial: the draws once, then each estimator under
#                each of its outlier models, and every width of the
#                calibration grid → <results>/synthetic/synthetic-<run key>/trials.tsv
#   invariance   the same draws ×scale select the same prefix → invariance.toml
#   summary      means, standard deviations, the calibrated width per case
#                → summary.tsv beside them
#   table        the table the paper reads → paper/figures/tab_synthetic.tex
#   macros       every number the prose quotes, as a lookup
#                \\syn{case}{prior}{estimator-model}{stat} → paper/figures/synthetic_numbers.tex
#
#   julia --project=. scripts/synthetic.jl [--plan] [--stage=NAME] [--force]
# =============================================================================
using DrWatson
@quickactivate "CVPR2027"
include(srcdir("CVPR2027.jl"))

using Statistics: mean, std, median
using Printf: @sprintf
using VisualGeometryToolkitEval.Render: read_tsv, write_tsv

const CFG_PATH = projectdir("configs", "synthetic.toml")
const CFG = TOML.parsefile(CFG_PATH)
const KEY = run_key(CFG)                      # [render] is excluded
const OUTDIR = joinpath(results_root(), "synthetic", "synthetic-" * KEY)
const TRIALS = joinpath(OUTDIR, "trials.tsv")
const INVARIANCE = joinpath(OUTDIR, "invariance.toml")
const SUMMARY = joinpath(OUTDIR, "summary.tsv")
const TABLE = projectdir("paper", "figures", CFG["render"]["table"])
const MACROS = projectdir("paper", "figures", CFG["render"]["macros"])

const SIM = CFG["simulate"]
const POT = CFG["pot"]
const BETA = Tuple(Float64.(CFG["label_prior"]["beta"]))

# The outlier components an estimator is scored with: component II needs a
# fitted shape, so only the peaks-over-threshold and soft-label estimators have it.
models_of(est) = [Symbol(m) for m in SIM["outlier_models"] if m == "uniform" || est["kind"] in ("pot", "soft")]

priors() = SIM["label_priors"]

# The package's scorer for the hard-label estimators (n_θ = 0: one fixed
# hypothesis, nothing fitted), at a supplied width or with an outlier model.
hard_scoring(outlier, prior) = StudentTUniformScoring(0, outlier; label_prior = label_prior_of(prior, BETA))

# One estimator under one outlier component and label prior → (k, tp, shape).
function evaluate(t::Trial, est, model, prior, cache::WorkspaceCache)
    kind = est["kind"]
    kind == "soft" && return soft_labels(t; model, prior, beta = BETA, soft = est, pot = POT)
    if kind == "fixed"
        return (hard_labels(t, hard_scoring(est["halfwidth_multiple"] * t.outlier_halfwidth, prior), cache)..., NaN)
    elseif kind == "pot"
        k, tp = hard_labels(t, hard_scoring(tail_model(est, POT; model), prior), cache)
        return (k, tp, complement_shape(t, est, k))
    elseif kind in ("block_max", "block_min")
        s = hard_scoring(block_term(t, kind == "block_max" ? :max : :min, est["blocks"]), prior)
        return (hard_labels(t, s, cache)..., NaN)
    end
    throw(ArgumentError("estimator kind $(repr(kind))"))
end

# The shape reported for a peaks-over-threshold estimator: the fit to the
# complement of the partition the score selected (a diagnostic only).
complement_shape(t::Trial, est, k) =
    length(t.z) - k >= 2 ? tail_fit(tail_rules(est, POT), view(t.z, k+1:length(t.z))).shape : NaN

calibration_grid() = (c = CFG["calibration"];
                      exp.(range(log(c["outlier_halfwidth_min"]), log(c["outlier_halfwidth_max"]); length = c["n"])))

# One stream per case (seed + case index): every estimator sees the same draws.
case_rng(i) = StableRNG(SIM["seed"] + i)

# paper-project: allow C2 the column schema of trials.tsv, not a list of data names
const COLUMNS = ["case", "trial", "true_halfwidth", "prior", "estimator", "model", "outlier_halfwidth", "k", "tp", "f1", "shape"]

function simulate()
    grid = calibration_grid()
    cache = WorkspaceCache()
    rows = Vector{Any}[]
    for (ci, case) in enumerate(CFG["cases"])
        rng = case_rng(ci)
        n_in = case["n_in"]
        for trial in 1:SIM["trials"]
            t = Trial(rng, case)
            H = t.outlier_halfwidth
            for prior in priors()
                for est in CFG["estimators"], model in models_of(est)
                    k, tp, shape = evaluate(t, est, model, prior, cache)
                    push!(rows, [case["key"], trial, H, prior, est["key"], model, "", k, tp,
                                 partition_metrics(k, tp, n_in).f1, shape])
                end
                for w in grid
                    k, tp = hard_labels(t, hard_scoring(w, prior), cache)
                    push!(rows, [case["key"], trial, H, prior, "calibration", "uniform", w, k, tp,
                                 partition_metrics(k, tp, n_in).f1, NaN])
                end
            end
        end
    end
    write_tsv(TRIALS, COLUMNS, rows)
    write_run_records(OUTDIR, CFG; driver = "synthetic.jl", config_source = CFG_PATH, key = KEY)
    return nothing
end

function invariance()
    inv = CFG["invariance"]
    ci = findfirst(c -> c["key"] == inv["case"], CFG["cases"])
    case = CFG["cases"][ci]
    c = inv["residual_multiplier"]
    rng = StableRNG(SIM["seed"] + 1000 + ci)
    cache = WorkspaceCache()
    ests = filter(e -> e["key"] in inv["estimators"], CFG["estimators"])
    same = Dict(string(e["key"], "/", p, "/", m) => 0 for e in ests for p in priors() for m in models_of(e))
    for _ in 1:inv["trials"]
        t = Trial(rng, case)
        tc = Trial(c .* t.r, t.inlier, c * t.outlier_halfwidth, c .* t.z, t.block_rank)
        for e in ests, p in priors(), m in models_of(e)
            same[string(e["key"], "/", p, "/", m)] += evaluate(t, e, m, p, cache)[1] == evaluate(tc, e, m, p, cache)[1]
        end
    end
    open(INVARIANCE, "w") do io
        TOML.print(io, Dict("trials" => inv["trials"], "residual_multiplier" => c,
                            "estimators" => inv["estimators"], "identical" => same))
    end
    return nothing
end

nanmedian(x) = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))
num(v) = isempty(v) ? NaN : parse(Float64, v)

# paper-project: allow C2 the column schema of summary.tsv
const SUMMARY_COLUMNS = ["case", "prior", "estimator", "model", "outlier_halfwidth", "k_mean", "k_std", "f1_mean",
                         "f1_std", "precision_mean", "recall_mean", "shape_median"]

function summary()
    require_input(TRIALS; what = "the simulated trials", provenance = "stage simulate of this file")
    # The trials grouped once by (case, prior, estimator, model, width).
    groups = Dict{NTuple{5, String}, Vector{Dict{String, String}}}()
    for r in read_tsv(TRIALS)
        push!(get!(groups, (r["case"], r["prior"], r["estimator"], r["model"], r["outlier_halfwidth"]), Dict{String, String}[]), r)
    end
    stat(rs, c) = [num(r[c]) for r in rs]
    out = Vector{Any}[]
    line!(key, est, model, width, rs, n_in) = push!(out, [key, est..., model, width,
        mean(stat(rs, "k")), std(stat(rs, "k")), mean(stat(rs, "f1")), std(stat(rs, "f1")),
        mean(stat(rs, "tp") ./ stat(rs, "k")), mean(stat(rs, "tp")) / n_in, nanmedian(stat(rs, "shape"))])
    crit = CFG["calibration"]["criterion"]
    for case in CFG["cases"], prior in priors()
        n_in = case["n_in"]
        for est in CFG["estimators"], model in models_of(est)
            line!(case["key"], (prior, est["key"]), string(model), "",
                  groups[(case["key"], prior, est["key"], string(model), "")], n_in)
        end
        # The calibrated baseline: the grid width with the best mean criterion.
        cal = filter(((g, _),) -> g[1] == case["key"] && g[2] == prior && g[3] == "calibration", groups)
        best = argmax(g -> mean(stat(cal[g], crit)), collect(keys(cal)))
        line!(case["key"], (prior, "calibrated"), "uniform", best[5], cal[best], n_in)
    end
    write_tsv(SUMMARY, SUMMARY_COLUMNS, out)
    return nothing
end

read_summary() = read_tsv(SUMMARY)

# A column's header: the score's label (src/labels.jl), from its estimator's kind.
estimator_kind(key) = key == "calibrated" ? "calibrated" : only(e["kind"] for e in CFG["estimators"] if e["key"] == key)
column_label(c) = score_label(estimator_kind(c["estimator"]), c["model"], c["prior"])

# The table the paper reads: per case, the selected prefix length and the F1
# of the partition (mean ± sd), for the columns of [render].
function table()
    rows = read_summary()
    r = CFG["render"]
    get_row(case, est, model, prior) = only(filter(x -> x["case"] == case && x["estimator"] == est &&
                                                     x["model"] == model && x["prior"] == prior, rows))
    fmt(x, d) = @sprintf("%.*f", d, parse(Float64, x))
    cols = r["columns"]
    mkpath(dirname(TABLE))
    open(TABLE, "w") do io
        println(io, "% generated by scripts/synthetic.jl (run ", KEY, "); do not edit")
        println(io, "\\begin{tabular}{l", "c"^length(cols), "}")
        println(io, "\\toprule")
        # The group header: one \\multicolumn per run of equal groups, ruled below.
        groups = Tuple{String,Int}[]
        for c in cols
            g = get(c, "group", "")
            !isempty(groups) && groups[end][1] == g ? (groups[end] = (g, groups[end][2] + 1)) : push!(groups, (g, 1))
        end
        if any(!isempty(first(g)) for g in groups)
            println(io, " & ", join(("\\multicolumn{$n}{c}{$g}" for (g, n) in groups), " & "), " \\\\")
            lo = 2
            println(io, join(("\\cmidrule(lr){$(lo + sum(last.(groups[1:j-1]); init = 0))-$(lo + sum(last.(groups[1:j])) - 1)}" for j in eachindex(groups)), ""))
        end
        println(io, "Outliers & ", join((column_label(c) for c in cols), " & "), " \\\\")
        println(io, "\\midrule")
        for case in CFG["cases"]
            cells = map(cols) do c
                x = get_row(case["key"], c["estimator"], c["model"], c["prior"])
                string("\$", fmt(x["k_mean"], r["digits"]), "{\\pm}", fmt(x["k_std"], r["digits"]),
                       "\$\\,/\\,", fmt(x["f1_mean"], r["f1_digits"]))
            end
            println(io, r["case_tex"][case["key"]], " & ", join(cells, " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return nothing
end

# The numbers the prose quotes: one control sequence per summary statistic,
# read through \syn{case}{prior}{estimator-model}{stat} (defined here), so no
# number is typed into the manuscript.
const MACRO_STATS = Dict("k" => ("k_mean", "digits"), "ksd" => ("k_std", "digits"),   # paper-project: allow C2 the statistics' column names in summary.tsv
                         "f1" => ("f1_mean", "f1_digits"), "prec" => ("precision_mean", "f1_digits"),
                         "rec" => ("recall_mean", "f1_digits"), "shape" => ("shape_median", "shape_digits"),
                         "halfwidth" => ("outlier_halfwidth", "halfwidth_digits"))

function macros()
    rows = read_summary()
    r = CFG["render"]
    inv = TOML.parsefile(INVARIANCE)
    mkpath(dirname(MACROS))
    open(MACROS, "w") do io
        println(io, "% generated by scripts/synthetic.jl (run ", KEY, "); do not edit")
        println(io, "\\providecommand{\\syn}[4]{\\csname syn@#1@#2@#3@#4\\endcsname}")
        for x in rows, (stat, (col, dig)) in sort(collect(MACRO_STATS))
            v = isempty(x[col]) ? NaN : parse(Float64, x[col])
            isfinite(v) || continue
            name = join(("syn", x["case"], x["prior"], x["estimator"] * "-" * x["model"], stat), '@')
            println(io, "\\expandafter\\def\\csname ", name, "\\endcsname{", replace(@sprintf("%.*f", r[dig], v), r"^-" => "\\ensuremath{-}"), "}")
        end
        derived_macros(io, rows, r)
        println(io, "\\newcommand{\\SynTrials}{", SIM["trials"], "}")
        println(io, "\\newcommand{\\SynGridN}{", CFG["calibration"]["n"], "}")
        println(io, "\\newcommand{\\SynInvTrials}{", inv["trials"], "}")
        println(io, "\\newcommand{\\SynInvScale}{", round(Int, inv["residual_multiplier"]), "}")
        println(io, "\\newcommand{\\SynInvIdentical}{", minimum(values(inv["identical"])), "}")
    end
    return nothing
end

# The numbers the prose derives from the summary and the config, never typed:
# the generative model's constants per case, the γ–width equivalence's
# predicted calibrated widths (eq. gamma-width), and the bounds the text states.
function derived_macros(io, rows, r)
    f1(case, prior, est, model) = parse(Float64, only(x for x in rows if x["case"] == case && x["prior"] == prior &&
                                                       x["estimator"] == est && x["model"] == model)["f1_mean"])
    ksd(case, prior, est, model) = parse(Float64, only(x for x in rows if x["case"] == case && x["prior"] == prior &&
                                                        x["estimator"] == est && x["model"] == model)["k_std"])
    def(name, v) = println(io, "\\expandafter\\def\\csname ", name, "\\endcsname{", v, "}")
    fmt(v, d) = replace(@sprintf("%.*f", d, v), r"^-" => "\\ensuremath{-}")
    keys_ = [c["key"] for c in CFG["cases"]]
    for c in CFG["cases"]
        k = c["key"]
        def("syncase@$k@nin", c["n_in"]); def("syncase@$k@nout", c["n_out"]); def("syncase@$k@N", c["n_in"] + c["n_out"])
        haskey(c, "outlier_halfwidth") && def("syncase@$k@halfwidth", fmt(c["outlier_halfwidth"], 0))
        haskey(c, "outlier_halfwidth_min") && def("syncase@$k@halfwidthmin", fmt(c["outlier_halfwidth_min"], 0))
        haskey(c, "outlier_halfwidth_max") && def("syncase@$k@halfwidthmax", fmt(c["outlier_halfwidth_max"], 0))
        def("syncase@$k@fraction", fmt(c["n_in"] / (c["n_in"] + c["n_out"]), 2))
        def("syncase@$k@percent", fmt(100 * c["n_in"] / (c["n_in"] + c["n_out"]), 0))
        # The width at which γ = ½ selects as the true inlier fraction would at the
        # supplied width (eq. gamma-width, d_g = 1): the odds ratio times the
        # density-matched width (the support for uniform outliers, half of it for
        # a linearly decaying density, whose density at the model is twice).
        if haskey(c, "outlier_halfwidth")
            γ = c["n_in"] / (c["n_in"] + c["n_out"])
            matched = c["outliers"] == "linear" ? c["outlier_halfwidth"] / 2 : c["outlier_halfwidth"]
            def("syncase@$k@equivwidth", fmt(matched * γ / (1 - γ), 0))
        end
    end
    println(io, "\\providecommand{\\syncase}[2]{\\csname syncase@#1@#2\\endcsname}")
    # "the extreme-value width performs as the known width does": the largest F1 gap.
    def("SynMaxGapKnownEV", fmt(maximum(abs(f1(k, "half", "pot90", "uniform") - f1(k, "half", "fixed", "uniform")) for k in keys_), r["f1_digits"]))
    # "moving η from 0.8 to 0.95 changes the F1 score by at most …"
    def("SynMaxGapEta", fmt(maximum(maximum(f1(k, "half", e, "uniform") for e in ("pot80", "pot90", "pot95")) -
                                    minimum(f1(k, "half", e, "uniform") for e in ("pot80", "pot90", "pot95")) for k in keys_), r["f1_digits"]))
    # the growth of sd(k*) under component II where inliers are few or overlap heavily
    ratios = [ksd(k, "half", "pot90", "gpd") / ksd(k, "half", "pot90", "uniform") for k in ("uniform_3pct", "uniform_A10")]
    def("SynSdRatioMin", fmt(minimum(ratios), 1)); def("SynSdRatioMax", fmt(maximum(ratios), 1))
    def("SynNumCases", length(CFG["cases"]))
    # Under hard labels, how often profiling γ selects another partition than
    # marginalizing it (paired: same trial, estimator and outlier model).
    sel = Dict{Tuple, Dict{String, String}}()
    for t in read_tsv(TRIALS)
        t["estimator"] in ("calibration", "soft90") && continue
        t["prior"] in ("profile", "marginal") || continue
        get!(sel, (t["case"], t["trial"], t["estimator"], t["model"]), Dict{String, String}())[t["prior"]] = t["k"]
    end
    pairs = [v for v in values(sel) if length(v) == 2]
    def("SynProfilePairs", length(pairs)); def("SynProfileSame", count(v -> v["profile"] == v["marginal"], pairs))
    return nothing
end
const STAGES = [
    Stage(:simulate, "every case × trial × estimator × outlier model, and the calibration grid, keyed by the run key",
          () -> run_complete(OUTDIR; sidecar = "trials.tsv"), simulate),
    Stage(:invariance, "the same draws ×scale select the same prefix", () -> isfile(INVARIANCE), invariance),
    Stage(:summary, "means, sds and the calibrated width per case", () -> fresher(SUMMARY, TRIALS), summary),
    Stage(:table, "the table the paper reads", () -> fresher(TABLE, SUMMARY, CFG_PATH), table),
    Stage(:macros, "the numbers the prose quotes", () -> fresher(MACROS, SUMMARY, INVARIANCE, CFG_PATH), macros),
]

run_stages("synthetic", STAGES)
