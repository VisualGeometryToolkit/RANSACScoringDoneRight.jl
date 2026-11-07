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
const SWEEP_KW = (; kmin = SIM["kmin"], kmax_margin = SIM["kmax_margin"], pot = CFG["pot"],
                   beta = Tuple(Float64.(CFG["label_prior"]["beta"])))

# The outlier models an estimator is scored with: the extended model needs a
# fitted shape, so only the POT estimators have it.
models_of(est) = [Symbol(m) for m in SIM["outlier_models"] if m == "uniform" || est["kind"] in ("pot", "soft")]

# One estimator under one outlier model and label prior: the hard-label sweep
# over partitions, or the soft-label mixture (labels summed out).
function evaluate(d, inl, keys, A, est, model, prior)
    est["kind"] == "soft" && return soft_mixture(d, inl; model, prior, beta = SWEEP_KW.beta, soft = est, pot = SWEEP_KW.pot)
    return sweep(d, inl, est; model, prior, keys, true_halfwidth = A, SWEEP_KW...)
end

priors() = Symbol.(SIM["label_priors"])

calibration_grid() = (c = CFG["calibration"];
                      exp.(range(log(c["a_min"]), log(c["a_max"]); length = c["n"])))

# One stream per case (seed + case index): every estimator sees the same draws.
case_rng(i) = StableRNG(SIM["seed"] + i)

# paper-project: allow C2 the column schema of trials.tsv, not a list of data names
const COLUMNS = ["case", "trial", "A", "prior", "estimator", "model", "a", "k", "tp", "f1", "xi"]

function simulate()
    mkpath(OUTDIR)
    grid = calibration_grid()
    open(TRIALS, "w") do io
        println(io, join(COLUMNS, '\t'))
        row(cols...) = println(io, join(cols, '\t'))
        for (ci, case) in enumerate(CFG["cases"])
            rng = case_rng(ci)
            for t in 1:SIM["trials"]
                d, inl, A = draw_case(rng, case)
                keys = rand(rng, length(d))
                run(est, model, prior) = sweep(d, inl, est; model, prior, keys, true_halfwidth = A, SWEEP_KW...)
                for prior in priors()
                    for est in CFG["estimators"], model in models_of(est)
                        k, tp, ξ = evaluate(d, inl, keys, A, est, model, prior)
                        row(case["key"], t, A, prior, est["key"], model, "", k, tp,
                            partition_metrics(k, tp, case["n_in"]).f1, ξ)
                    end
                    for a in grid
                        k, tp, _ = run(Dict("kind" => "fixed_abs", "a" => a), :uniform, prior)
                        row(case["key"], t, A, prior, "calibration", "uniform", a, k, tp,
                            partition_metrics(k, tp, case["n_in"]).f1, NaN)
                    end
                end
            end
        end
    end
    write_run_records(OUTDIR, CFG; driver = "synthetic.jl", config_source = CFG_PATH, key = KEY)
    return nothing
end

function invariance()
    inv = CFG["invariance"]
    ci = findfirst(c -> c["key"] == inv["case"], CFG["cases"])
    case = CFG["cases"][ci]
    est = only(filter(e -> e["key"] == inv["estimator"], CFG["estimators"]))
    rng = StableRNG(SIM["seed"] + 1000 + ci)
    same = Dict(string(p, "/", m) => 0 for p in priors() for m in models_of(est))
    for _ in 1:inv["trials"]
        d, inl, A = draw_case(rng, case)
        keys = rand(rng, length(d))
        for p in priors(), m in models_of(est)
            k1 = sweep(d, inl, est; model = m, prior = p, keys, true_halfwidth = A, SWEEP_KW...)[1]
            k2 = sweep(d .* inv["scale"], inl, est; model = m, prior = p, keys,
                       true_halfwidth = A * inv["scale"], SWEEP_KW...)[1]
            same[string(p, "/", m)] += k1 == k2
        end
    end
    open(INVARIANCE, "w") do io
        TOML.print(io, Dict("trials" => inv["trials"], "scale" => inv["scale"],
                            "estimator" => inv["estimator"], "identical" => same))
    end
    return nothing
end

function read_trials()
    lines = readlines(TRIALS)
    hdr = split(lines[1], '\t')
    rows = [Dict(zip(hdr, split(l, '\t'))) for l in lines[2:end]]
    num(r, c) = (v = r[c]; isempty(v) ? NaN : parse(Float64, v))
    return [(case = r["case"], trial = parse(Int, r["trial"]), prior = r["prior"], estimator = r["estimator"],
             model = r["model"], a = num(r, "a"), k = num(r, "k"), tp = num(r, "tp"),
             f1 = num(r, "f1"), xi = num(r, "xi")) for r in rows]
end

nanmedian(x) = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))

function summary()
    require_input(TRIALS; what = "the simulated trials", provenance = "stage simulate of this file")
    rows = read_trials()
    out = String[join(["case", "prior", "estimator", "model", "a", "k_mean", "k_std", "f1_mean", "f1_std",
                       "precision_mean", "recall_mean", "xi_median"], '\t')]
    for case in CFG["cases"], prior in string.(priors())
        n_in = case["n_in"]
        cr = filter(r -> r.case == case["key"] && r.prior == prior, rows)
        line(est, model, a, rs) = push!(out, join([case["key"], prior, est, model, a,
            mean(r.k for r in rs), std([r.k for r in rs]), mean(r.f1 for r in rs), std([r.f1 for r in rs]),
            mean(r.tp / r.k for r in rs), mean(r.tp / n_in for r in rs), nanmedian([r.xi for r in rs])], '\t'))
        for est in CFG["estimators"], model in models_of(est)
            line(est["key"], string(model), "", filter(r -> r.estimator == est["key"] && r.model == string(model), cr))
        end
        # The calibrated baseline: the grid width with the best mean criterion.
        cal = filter(r -> r.estimator == "calibration", cr)
        crit = Symbol(CFG["calibration"]["criterion"])
        widths = sort(unique(r.a for r in cal))
        score(a) = mean(getfield(r, crit) for r in cal if r.a == a)
        abest = widths[argmax(map(score, widths))]
        line("calibrated", "uniform", abest, filter(r -> r.a == abest, cal))
    end
    write(SUMMARY, join(out, '\n') * '\n')
    return nothing
end

function read_summary()
    lines = readlines(SUMMARY)
    hdr = split(lines[1], '\t')
    return [Dict(zip(hdr, split(l, '\t'))) for l in lines[2:end]]
end

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
        println(io, "Outliers & ", join((c["header"] for c in cols), " & "), " \\\\")
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
                         "rec" => ("recall_mean", "f1_digits"), "xi" => ("xi_median", "xi_digits"),
                         "a" => ("a", "a_digits"))

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
        println(io, "\\newcommand{\\SynTrials}{", SIM["trials"], "}")
        println(io, "\\newcommand{\\SynGridN}{", CFG["calibration"]["n"], "}")
        println(io, "\\newcommand{\\SynInvTrials}{", inv["trials"], "}")
        println(io, "\\newcommand{\\SynInvScale}{", round(Int, inv["scale"]), "}")
        println(io, "\\newcommand{\\SynInvIdentical}{", minimum(values(inv["identical"])), "}")
    end
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
