# =============================================================================
# test_studies.jl — the scoring studies of scripts/studies/ are drivers of the
# paper-project conventions: each reads its one config of record, declares
# every input it reads under the data root as an [inputs.<id>] table, and
# writes only into its own keyed run directory under results_root() (Eval's
# keyed_cell_dir of its config: the run key folds the inputs' sha256 in).
# `--plan` resolves all of it without the data and writes nothing.
#
# Fail-first (2026-11-12): score_timing and sample_budget_miscal took their
# pool, ranges and output directories on the command line, defaulting into
# accv2026's results tree (data_root()/accv2026/results), and f32_radix_probe
# loaded a pool at include time and saved into `DATA_ROOT`, a name it never
# defined (an UndefVarError after the whole benchmark had run).
# =============================================================================
using Test
using TOML: TOML
using VisualGeometryToolkitEval.Experiments: run_isolated, run_key, keyed_cell_dir, data_root, results_root

const _STUDIES = [
    # driver under scripts/studies, config of record, the keyed_cell_dir kind
    ("score_timing.jl", "score_timing", "timing"),
    ("sample_budget_miscal.jl", "sample_budget_miscal", "curves"),
    ("f32_radix_probe.jl", "f32_radix_probe", "sort"),
]

@testset "scoring studies: one config, declared inputs, own keyed results" begin
    data, results = mktempdir(), mktempdir()
    env = Dict("VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => results)
    for (driver, name, kind) in _STUDIES
        @testset "$name" begin
            path = joinpath(@__DIR__, "..", "scripts", "studies", driver)
            m = run_isolated(path; args = ["--plan"], env)
            cfg = TOML.parsefile(joinpath(@__DIR__, "..", "configs", name * ".toml"))
            @test m.CFG == cfg
            @test m.KEY == run_key(cfg)
            withenv(env...) do
                @test m.OUTDIR == keyed_cell_dir(cfg, name, kind)
                @test startswith(m.OUTDIR, results_root())
            end
            @test !startswith(m.OUTDIR, data)
            ins = get(cfg, "inputs", Dict{String,Any}())
            @test !isempty(ins)
            for (id, t) in ins
                @test all(k -> haskey(t, k), ("tier", "path", "provenance")) || (@info "input lacks tier/path/provenance" name id; false)
                @test !isabspath(t["path"])
            end
        end
    end
    @test isempty(readdir(data))          # --plan wrote nothing into the input tree
end
