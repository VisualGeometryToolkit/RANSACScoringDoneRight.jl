# Experiments

## Reproducing the paper

The paper's figures are drawn by drivers in the research project
`neurips2026/`, which has its own environment:

```bash
cd neurips2026
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/fetch_data.jl          # a machine without the data (neurips2026/data/README.md)
julia --project=. scripts/fig_profile.jl --plan
julia --project=. test/runtests.jl
make -C paper
```

`neurips2026/REPRODUCE.md` maps every figure of the paper to its driver, its
config of record and its data. The figures regenerate byte for byte from the
six stored evaluation runs of record. The evaluation that produced those runs
is documented there but not yet ported to the current stack.

## Scoring studies (retired)

Two scoring studies and a decision probe used to ship beside the package as
drivers in `neurips2026/scripts/studies/`. They were retired in 0.8.3 and are
**not reproducible from VisualGeometryToolkitEval 0.10 on**: their pool of record (PT-E-SIFT-4K) is a
pool format that VisualGeometryToolkitEval no longer reads. Their write-ups
stay in `neurips2026/notes/` as records of the runs they describe, and the
drivers and configs are in the repository's history up to tag `v0.8.2`. Both
studies exercised the selection rule alone, with no local optimization: every
scorer scored the same stored minimal-sample hypotheses of the PhotoTourism
essential-matrix pool, and every scorer was called from the package itself.

These are not the paper's Profile and Sensitivity experiments, whose figure
drivers are `neurips2026/scripts/fig_profile.jl` and `fig_sensitivity.jl`
(see the [Overview](../overview.md#Where-the-paper's-experiments-live) and
`neurips2026/REPRODUCE.md`).

| write-up | question |
|:---|:---|
| `neurips2026/notes/sample_budget_miscal.md` | how selection improves with the hypothesis budget, calibrated and under a 32× miscalibrated scale |
| `neurips2026/notes/score_timing.md` | the per-hypothesis CPU cost of each scorer |

## Environment

`neurips2026/` is its own Julia project, not a dependency of the package: the
drivers need VisualGeometryToolkitEval (pool IO, the run records), which the
package deliberately does not depend on, and the concrete problem types of
VisualGeometryToolkitEstimation, which the package itself uses only for SIMFIT++.

```bash
cd neurips2026 && julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Sample budget under scale miscalibration (the write-up)

Each scorer's operating point is read from the deployed cross-validation
sidecar, tuned at calibration (ζ = 1) and not re-tuned; miscalibration
multiplies every scorer's native width by the same ζ, which is principled for
both families because both natives are linear in the noise scale. The budget is
counted in minimal samples, from 100 to 4 000, on 8 800 pairs of four held-out
test scenes.

The write-up reports mAA@10° and median pose error at every budget for the
Jeffreys score, MSAC, GaU, RANSAC and MAGSAC++. Calibrated, the five are within
0.013 mAA of one another at 4 000 samples; at ζ = 32 the Jeffreys score retains
83 % of its calibrated mAA and keeps improving with the budget, while the four
baselines retain 16–23 % and get worse as the budget grows.

## Per-hypothesis scoring cost (the write-up)

Single-threaded, on the stored residuals of one PhotoTourism scene, every scorer
at one shared operating point; the figure is the minimum over three
repetitions, per hypothesis scored.

The sort-and-sweep costs more than RANSAC and MSAC and less than GaU and
closed-form MAGSAC++; its cost grows linearly with the number of points, since
both its radix sort and its sweep are linear. The write-up's MAGSAC++
lookup-table row was measured with a tabulated scorer that has since been
removed from the package; the current driver times the exact forms only.
