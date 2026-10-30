# Experiments

## Reproducing the paper

The paper's figures are drawn by drivers in the research project
`neurips2026/`, which has its own environment:

```bash
cd neurips2026
julia --project=. -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl"); Pkg.instantiate()'
julia --project=. scripts/fetch_data.jl          # a machine without the data (neurips2026/data/README.md)
julia --project=. scripts/fig_profile.jl --plan
julia --project=. test/runtests.jl
make -C paper
```

`neurips2026/REPRODUCE.md` maps every figure of the paper to its driver, its
config of record and its data. The figures regenerate byte for byte from the
six stored evaluation runs of record. The evaluation that produced those runs
is documented there but not yet ported to the current stack.

## Scoring studies

The repository ships two scoring studies beside the package: a driver per study
in `neurips2026/scripts/studies/` and its write-up in `neurips2026/notes/`, checked in together so that a
result and the code that produced it travel together. Both exercise the
selection rule alone, with no local optimization: every scorer scores the same
stored minimal-sample hypotheses of the PhotoTourism essential-matrix pool
(PT-E-SIFT-4K, RootSIFT), and every scorer is called from the package itself;
the drivers contain no scoring code.

These are not the paper's Profile and Sensitivity experiments, whose figure
drivers are `neurips2026/scripts/fig_profile.jl` and `fig_sensitivity.jl`
(see the [Overview](../overview.md#Where-the-paper's-experiments-live) and
`neurips2026/REPRODUCE.md`).

| driver | write-up | question |
|:---|:---|:---|
| `neurips2026/scripts/studies/sample_budget_miscal.jl` (+ `scripts/slurm/*.sbatch`) | `neurips2026/notes/sample_budget_miscal.md` | how selection improves with the hypothesis budget, calibrated and under a 32× miscalibrated scale |
| `neurips2026/scripts/studies/score_timing.jl` (+ `scripts/slurm/*.sbatch`) | `neurips2026/notes/score_timing.md` | the per-hypothesis CPU cost of each scorer |
| `neurips2026/scripts/studies/f32_radix_probe.jl` | — | a decision probe: does a Float32 sort key speed up the sweep's radix sort at these problem sizes? |

## Environment

`neurips2026/` is its own Julia project, not a dependency of the package: the
drivers need pool IO (VisualGeometryToolkitEval) and the concrete problem types
(VisualGeometryToolkitEstimation), which the package deliberately does not depend on.

```bash
cd neurips2026 && julia --project=. -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl"); Pkg.instantiate()'
```

The pools are read with VisualGeometryToolkitEval's `load_pool` and
`load_pool_residuals`. The data root is `ENV["CAU_DATA_ROOT"]` (no default: set
it, or give the drivers `--pool` and `--out`); pools are read from
`<root>/accv2026/data/PT-E-SIFT-4K/` and heavy outputs land under
`<root>/accv2026/results/<experiment>/`. The `.sbatch` files are example SLURM
launchers: they carry the exact driver invocation behind the published numbers,
with the resource requests of the cluster they ran on — adapt the partition,
CPUs and memory to yours. Submit them from the repository root.

## Sample budget under scale miscalibration

Each scorer's operating point is read from the deployed cross-validation
sidecar, tuned at calibration (ζ = 1) and not re-tuned; miscalibration
multiplies every scorer's native width by the same ζ, which is principled for
both families because both natives are linear in the noise scale. The budget is
counted in minimal samples, from 100 to 4 000, on 8 800 pairs of four held-out
test scenes.

```bash
sbatch neurips2026/scripts/slurm/sample_budget_miscal.sbatch
```

The write-up reports mAA@10° and median pose error at every budget for the
Jeffreys score, MSAC, GaU, RANSAC and MAGSAC++. Calibrated, the five are within
0.013 mAA of one another at 4 000 samples; at ζ = 32 the Jeffreys score retains
83 % of its calibrated mAA and keeps improving with the budget, while the four
baselines retain 16–23 % and get worse as the budget grows.

## Per-hypothesis scoring cost

Single-threaded, on the stored residuals of one PhotoTourism scene, every scorer
at one shared operating point; the figure is the minimum over three
repetitions, per hypothesis scored.

```bash
sbatch neurips2026/scripts/slurm/score_timing.sbatch
```

The sort-and-sweep costs more than RANSAC and MSAC and less than GaU and
closed-form MAGSAC++; its cost grows linearly with the number of points, since
both its radix sort and its sweep are linear. The write-up's MAGSAC++
lookup-table row was measured with a tabulated scorer that has since been
removed from the package; the current driver times the exact forms only.
