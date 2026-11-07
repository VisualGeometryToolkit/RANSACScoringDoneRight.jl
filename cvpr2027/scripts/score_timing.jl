#!/usr/bin/env julia
# provenance: every run writes _config.toml and _provenance.toml (write_run_records) beside its results.
# =============================================================================
# scripts/score_timing.jl — the CPU cost per hypothesis of the extreme-value
# scorers (\label{tab:timing}), next to neurips2026's baselines, on the same
# protocol and with the same harness: this driver includes
# neurips2026/scripts/studies/score_timing.jl and hands its `time_scorers` the
# baseline list plus the scorers of configs/score_timing.toml. It contains no
# timing or scoring code of its own: every scorer is RANSACScoringDoneRight's,
# timed through VisualGeometryToolkitCore's `best_pool_model`.
#
# reproduce: reads configs/score_timing.toml
# reproduce: writes paper/figures/tab_timing.tex
# reproduce: writes paper/figures/timing_numbers.tex
#
# STAGES
#   time    µs per hypothesis of every scorer, both density regimes
#           → <results>/score_timing/score_timing-<run key>/timing.tsv
#   table   the table the paper reads → paper/figures/tab_timing.tex
#   macros  the timings the prose quotes, as \\scoretiming{score}{regime}{stat}
#           → paper/figures/timing_numbers.tex
#
#   julia -t 1 --project=. scripts/score_timing.jl [--plan] [--stage=NAME] [--force]
# =============================================================================
using DrWatson
@quickactivate "CVPR2027"
include(srcdir("CVPR2027.jl"))
# The harness and the baselines of the NeurIPS study (its `main` runs only as a script).
include(normpath(projectdir(), "..", "neurips2026", "scripts", "studies", "score_timing.jl"))

using Printf: @sprintf
using VisualGeometryToolkitEval.Render: read_tsv, write_tsv

const CFG_PATH = projectdir("configs", "score_timing.toml")
const CFG = TOML.parsefile(CFG_PATH)
const KEY = run_key(CFG)                       # [render] is excluded
const OUTDIR = joinpath(results_root(), "score_timing", "score_timing-" * KEY)
const TIMING = joinpath(OUTDIR, "timing.tsv")
const TABLE = projectdir("paper", "figures", CFG["render"]["table"])
const MACROS = projectdir("paper", "figures", CFG["render"]["macros"])

const BETA_PRIOR = Tuple(Float64.(CFG["label_prior"]["beta"]))

function outlier_of(name::AbstractString)
    rules = tail_rules(CFG["tail"], CFG["tail"])
    name == "ev_halfwidth" && return ExtremeValueHalfwidth(rules)
    name == "gpd_prefix" && return GeneralizedParetoOutliers{:prefix}(rules)
    name == "gpd_hypothesis" && return GeneralizedParetoOutliers{:hypothesis}(rules)
    throw(ArgumentError("outlier $(repr(name)): ev_halfwidth, gpd_prefix or gpd_hypothesis"))
end

# The NeurIPS baselines at the shared operating point, then the extreme-value
# scorers (model DOF 1, codimension 1: the recorded protocol's scorer shape);
# `reference` selects the exact Θ(N²) references or the rest.
function our_scorers(pairs; reference::Bool)
    rows = filter(r -> get(r, "reference", false) == reference, CFG["scorers"])
    return map(rows) do row
        s = StudentTUniformScoring(1, outlier_of(row["outlier"]); codimension = 1,
                                   label_prior = label_prior_of(row["label_prior"], BETA_PRIOR))
        (row["name"], timed_selection(s, pairs))
    end
end
all_scorers(pairs) = vcat(timed_scorers(pairs, CFG["protocol"]["loss_width"]), our_scorers(pairs; reference = false))

function time_all()
    Threads.nthreads() == 1 || error("single-thread BY PROTOCOL — use `julia -t 1`")
    p = CFG["pool"]
    path = joinpath(data_root(), p["dir"], p["scene"] * ".jld2")
    require_input(path; what = "the PT-E pool", provenance = "configs/score_timing.toml [pool]")
    subs = timing_subsets(path, p["npairs"])
    results = time_scorers(subs, all_scorers; reps = CFG["protocol"]["reps"])
    # The exact references on the first reference_npairs pairs of each regime.
    m = p["reference_npairs"]
    small = [(g, prs[1:min(m, end)], mats[1:min(m, end)]) for (g, prs, mats) in subs]
    append!(results, time_scorers(small, prs -> our_scorers(prs; reference = true); reps = CFG["protocol"]["reps"]))
    # paper-project: allow C2 the column schema of timing.tsv
    write_tsv(TIMING, ("regime", "score", "us_per_hyp", "n_avg", "n_hyp", "n_pairs"),
              [(r.regime, r.score, r.us_per_hyp, r.n_avg, r.n_hyp, r.n_pairs) for r in results])
    write_run_records(OUTDIR, CFG; driver = "score_timing.jl", config_source = CFG_PATH, key = KEY,
                      inputs = [path])
    return nothing
end

# The table the paper reads: one row per scorer, µs per hypothesis in each regime.
function table()
    rows = read_tsv(TIMING)
    regimes = unique(r["regime"] for r in rows)
    scores = unique(r["score"] for r in rows)
    navg = Dict(r["regime"] => r["n_avg"] for r in rows)
    us(s, g) = parse(Float64, only(filter(r -> r["score"] == s && r["regime"] == g, rows))["us_per_hyp"])
    d = CFG["render"]["digits"]
    mkpath(dirname(TABLE))
    open(TABLE, "w") do io
        println(io, "% generated by scripts/score_timing.jl (run ", KEY, "); do not edit")
        println(io, "\\begin{tabular}{l", "r"^length(regimes), "}")
        println(io, "\\toprule")
        println(io, "Score & ", join(("$g (\$\\bar n=$(navg[g])\$)" for g in regimes), " & "), " \\\\")
        println(io, "\\midrule")
        for s in scores
            npairs = only(unique(r["n_pairs"] for r in rows if r["score"] == s))
            note = npairs == string(CFG["pool"]["npairs"]) ? "" : "\$^\\dagger\$"
            println(io, timing_label(s), note, " & ",
                    join((@sprintf("%.*f", d, us(s, g)) for g in regimes), " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return nothing
end

# A scorer's key in the macros: its name as letters and digits.
timing_key(name) = lowercase(replace(name, r"[^A-Za-z0-9]+" => ""))

# The timings the prose quotes: µs per hypothesis, and the ratio to the
# supplied-width profile-marginal score, per scorer and regime; the regimes'
# mean point counts and the reference rows' pair count.
function macros()
    rows = read_tsv(TIMING)
    d = CFG["render"]["digits"]
    base = Dict(r["regime"] => parse(Float64, r["us_per_hyp"]) for r in rows if r["score"] == "Ours (exact)")
    mkpath(dirname(MACROS))
    open(MACROS, "w") do io
        println(io, "% generated by scripts/score_timing.jl (run ", KEY, "); do not edit")
        println(io, "\\providecommand{\\scoretiming}[3]{\\csname timing@#1@#2@#3\\endcsname}")
        for r in rows
            us = parse(Float64, r["us_per_hyp"])
            key = "timing@" * timing_key(r["score"]) * "@" * r["regime"]
            println(io, "\\expandafter\\def\\csname ", key, "@us\\endcsname{", @sprintf("%.*f", d, us), "}")
            println(io, "\\expandafter\\def\\csname ", key, "@ratio\\endcsname{", @sprintf("%.1f", us / base[r["regime"]]), "}")
        end
        for g in unique(r["regime"] for r in rows)
            println(io, "\\expandafter\\def\\csname timing@regime@", g, "@navg\\endcsname{",
                    first(r["n_avg"] for r in rows if r["regime"] == g), "}")
        end
        println(io, "\\newcommand{\\TimingPairs}{", CFG["pool"]["npairs"], "}")
        println(io, "\\newcommand{\\TimingReferencePairs}{", CFG["pool"]["reference_npairs"], "}")
    end
    return nothing
end

const STAGES = [
    Stage(:time, "µs per hypothesis of every scorer, keyed by the run key",
          () -> run_complete(OUTDIR; sidecar = "timing.tsv"), time_all),
    Stage(:table, "the table the paper reads", () -> fresher(TABLE, TIMING, CFG_PATH), table),
    Stage(:macros, "the timings the prose quotes", () -> fresher(MACROS, TIMING, CFG_PATH), macros),
]

run_stages("score_timing", STAGES)
