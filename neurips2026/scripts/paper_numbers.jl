#!/usr/bin/env julia
# provenance: every run writes _config.toml and _provenance.toml (write_run_records) beside numbers.tsv.
# =============================================================================
# scripts/paper_numbers.jl — the numbers Sec. 4 states about Fig. 1 and Fig. 2
# (configs/paper_numbers.toml), recomputed from the stored runs of record and
# checked against the values main.tex states. The prose states most of them
# as literals, and this driver is what ties each one to its data; a number the
# config gives `macros` is GENERATED: written to paper/figures/paper_numbers.tex,
# which main.tex reads (REPRODUCE.md, "Numbers in the prose").
#
# reproduce: reads configs/paper_numbers.toml
# reproduce: writes paper/figures/paper_numbers.tex
#
# STAGES
#   compute  every listed statistic, unrounded and rounded as stated
#            → <results>/numbers/paper_numbers-<run key>/numbers.tsv
#   macros   the generated numbers as \newcommand lines → paper_numbers.tex
#            beside numbers.tsv
#   verify   the macro file against paper/figures (verify.tsv)
#   deploy   into paper/figures — only where that changes nothing, unless
#            [deploy] replace_submitted
#   report   each number against the value the prose states
#
#   julia --project=. scripts/paper_numbers.jl [--plan] [--stage=NAME] [--force]
# =============================================================================
using DrWatson
@quickactivate "NeurIPS2026"
include(srcdir("NeurIPS2026.jl"))

const CFG_PATH = projectdir("configs", "paper_numbers.toml")
const CFG = TOML.parsefile(CFG_PATH)
const KEY = run_key(CFG)
const OUTDIR = joinpath(results_root(), "numbers", "paper_numbers-" * KEY)
const TABLE = joinpath(OUTDIR, "numbers.tsv")
const ASSETS = [CFG["render"]["macros"]]

function compute()
    rows = [check_prose_number(CFG, e) for e in CFG["numbers"]]
    mkpath(OUTDIR)
    open(TABLE, "w") do io
        println(io, join(("id", "computed", "rounded", "stated", "relation", "ok"), '\t'))
        for r in rows
            println(io, join((r.id, join(r.computed, ","), join(r.rounded, ","), join(r.stated, ","), r.relation, r.ok), '\t'))
        end
    end
    inputs = unique([joinpath(stored_dir(CFG), run, "source.jld2") for run in values(CFG["runs"])])
    write_run_records(OUTDIR, CFG; driver = "paper_numbers.jl", config_source = CFG_PATH, key = KEY, inputs)
    return nothing
end

macros() = write(joinpath(OUTDIR, only(ASSETS)), prose_macros(CFG, [check_prose_number(CFG, e) for e in CFG["numbers"]]))
verify() = (verify_assets(OUTDIR, ASSETS); nothing)
deploy() = deploy_assets(OUTDIR, ASSETS, CFG)

function report()
    lines = readlines(TABLE)[2:end]
    for l in lines
        id, computed, rounded, stated, relation, ok = split(l, '\t')
        println("   ", rpad(id, 30), ok == "true" ? "ok       " : "DIFFERS  ",
                "stated ", stated, relation == "lt" ? " (upper bound)" : "", ", computed ", rounded, " (", computed, ")")
    end
    nbad = count(l -> endswith(l, "\tfalse"), lines)
    println("   $(length(lines) - nbad) of $(length(lines)) prose numbers match their data")
    return nothing
end

const STAGES = [
    Stage(:compute, "the prose numbers from the stored runs, keyed by the run key",
          () -> run_complete(OUTDIR; sidecar = "numbers.tsv"), compute),
    Stage(:macros, "the generated numbers as a macro file", () -> fresher(joinpath(OUTDIR, only(ASSETS)), TABLE), macros),
    Stage(:verify, "the macro file against the paper's copy", () -> fresher(joinpath(OUTDIR, VERIFY_FILE), joinpath(OUTDIR, only(ASSETS))), verify),
    Stage(:deploy, "into paper/figures where nothing changes (or signed off)", () -> assets_deployed(OUTDIR, ASSETS), deploy),
    Stage(:report, "each against the value main.tex states (prints; writes nothing)", () -> false, report),
]

run_stages("paper_numbers", STAGES)
