# =============================================================================
# test_paper_project.jl — the stack's paper-project conventions
# (VisualGeometryToolkitEval `PaperProjects.paper_project_violations`): the layout,
# the data policy (D1–D4), the REPRODUCE.md asset index against what
# paper/main.tex reads, configs of record, provenance and the data catalog.
# Opt-outs with their reasons sit in the files they excuse
# (`# paper-project: allow <RULE> <reason>`); deferred violations are listed
# here, each with its fix. An entry that stops matching fails as STALE, so the
# list only shrinks, and any NEW violation fails.
# =============================================================================
using Test
using VisualGeometryToolkitEval.PaperProjects: paper_project_violations
using VisualGeometryToolkitEval.Experiments: DataPolicyAllow

# The scoring studies of scripts/studies/ (write-ups in notes/; not paper
# assets: the paper reads none of their outputs) predate the conventions: they
# take their inputs, outputs and grids on the command line (the .sbatch
# launchers of scripts/slurm/ carry the invocations behind the write-ups) and
# write unkeyed JLD2s. Fix, per study: a config of record under configs/
# (pool, ranges, scenes, ζ, budgets, repetitions), the output directory keyed
# by its run key under results_root(), write_run_records beside it, and
# run_stages. Until then they are listed here.
const _STUDY_NO_CONFIG = "reads no config of record"
const _STUDY_NO_PROVENANCE = "writes no provenance record"
const _STUDY_NO_KEY = "keys no output directory by the run key"
const PAPER_PROJECT_ALLOW = [
    DataPolicyAllow("C1", "scripts/studies/f32_radix_probe.jl", _STUDY_NO_CONFIG),
    DataPolicyAllow("P1", "scripts/studies/f32_radix_probe.jl", _STUDY_NO_PROVENANCE),
    DataPolicyAllow("P2", "scripts/studies/f32_radix_probe.jl", _STUDY_NO_KEY),
    # the probe's problem sizes and repetition count (→ its config)
    DataPolicyAllow("C2", "scripts/studies/f32_radix_probe.jl", "(1:256)"),
    DataPolicyAllow("C2", "scripts/studies/f32_radix_probe.jl", "(1:20)"),
    DataPolicyAllow("P1", "scripts/studies/sample_budget_miscal.jl", _STUDY_NO_PROVENANCE),
    DataPolicyAllow("P2", "scripts/studies/sample_budget_miscal.jl", _STUDY_NO_KEY),
    # not data: the kernel's per-process status file, read for the peak RSS it reports
    DataPolicyAllow("C2", "scripts/studies/sample_budget_miscal.jl", "\"/proc/self/status\""),
    DataPolicyAllow("C1", "scripts/studies/score_timing.jl", _STUDY_NO_CONFIG),
    DataPolicyAllow("P1", "scripts/studies/score_timing.jl", _STUDY_NO_PROVENANCE),
    DataPolicyAllow("P2", "scripts/studies/score_timing.jl", _STUDY_NO_KEY),
]

@testset "paper-project conventions" begin
    v = paper_project_violations(normpath(joinpath(@__DIR__, "..")); papers = ["paper/main.tex"],
                                 allow = PAPER_PROJECT_ALLOW)
    foreach(x -> @info("paper-project violation", x), v)
    @test isempty(v)
end
