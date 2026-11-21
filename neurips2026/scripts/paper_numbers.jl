#!/usr/bin/env julia
# provenance: every product writes _config.toml and _provenance.toml (produce) beside its output.
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
#            → <results>/numbers/paper_numbers/<key>/numbers.tsv
#   macros   the generated numbers as VisualGeometryToolkitEval number macros
#            (write_numbers), keyed by the compute product
#            → <results>/numbers/paper_macros/<key>/paper_numbers.tex
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
const ASSETS = [CFG["render"]["macros"]]
const RECORDS = (; driver = "paper_numbers.jl", config_source = CFG_PATH)
# the statistics, keyed by the config and the code; the macro file, keyed by them (its upstream) and the [render] table
# (under its own name: a whole config's key excludes [render])
numbers() = Product("numbers", "paper_numbers", CFG; output = "numbers.tsv")
macros_product() = Product("numbers", "paper_macros", Dict("macros" => CFG["render"]); output = only(ASSETS),
                           upstream = ("numbers" => numbers(),))
inputs() = [input_path(CFG, id) for id in sort(collect(keys(CFG["inputs"])))]   # resolved when a stage runs (--plan needs no data)

const COLUMNS = ("id", "computed", "rounded", "stated", "relation", "ok")
_cell(x) = x isa AbstractVector ? join(x, ",") : x
compute(dir) = (write_tsv(joinpath(dir, "numbers.tsv"), COLUMNS,
                          ([_cell(getfield(r, Symbol(c))) for c in COLUMNS] for r in (check_prose_number(CFG, e) for e in CFG["numbers"])));
                nothing)

# the upstream table's rows, as check_prose_number returns them (the macros read the numbers they are keyed by)
_floats(s) = parse.(Float64, split(s, ","))
read_numbers() = [(id = r["id"], computed = _floats(r["computed"]), rounded = _floats(r["rounded"]), stated = _floats(r["stated"]),
                   relation = r["relation"], ok = r["ok"] == "true") for r in read_tsv(joinpath(produced_dir(numbers()), "numbers.tsv"))]

function macros(dir)
    up = numbers()
    write_numbers(joinpath(dir, only(ASSETS)),
                  prose_numbers(CFG, read_numbers(); run_key = up.key, commit = string(read_provenance(product_dir(up))["gitcommit"])))
    return nothing
end

function report()
    rows = read_numbers()
    for r in rows
        println("   ", rpad(r.id, 30), r.ok ? "ok       " : "DIFFERS  ", "stated ", join(r.stated, ","),
                r.relation == "lt" ? " (upper bound)" : "", ", computed ", join(r.rounded, ","), " (", join(r.computed, ","), ")")
    end
    println("   $(count(r -> r.ok, rows)) of $(length(rows)) prose numbers match their data")
    return nothing
end

const STAGES = [
    Stage(:compute, "the prose numbers from the stored runs, keyed by the config and the code";
          items = () -> [numbers()], product = identity, run = p -> produce(compute, p; RECORDS..., inputs = inputs())),
    Stage(:macros, "the generated numbers as a macro file, keyed by the numbers";
          items = () -> [macros_product()], product = identity, run = p -> produce(macros, p; RECORDS...)),
    asset_stages(macros_product, CFG, ASSETS)...,
    Stage(:report, "each against the value main.tex states (prints; writes nothing)"; check = report),
]

run_stages("paper_numbers", STAGES)
