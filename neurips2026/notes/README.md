# notes/

Write-ups of the scoring studies in `../scripts/studies/` (formerly the
package's top-level `results/`), checked in so a result and the code that
produced it travel together. They are not paper assets: the paper's figures
and their generators are indexed in `../REPRODUCE.md`.

| File | Experiment | Job |
|---|---|---|
| `score_timing.md` | `neurips2026/scripts/studies/score_timing.jl` — per-hypothesis CPU scoring cost | 4839 (reproduced by 4840) |

Each write-up names the driver that produced it and the JLD2 it was read from.
The JLD2 data itself is not committed (it lands under
`$CAU_DATA_ROOT/accv2026/results/score_timing/`), since it is regenerable from
the committed driver and does not belong in the package.

Regenerate any of these with the matching `.sbatch` in `../scripts/slurm/`. Those
launchers carry the exact invocation behind the published numbers; changing
`--npairs` or `--reps` invalidates cross-row comparisons, because each figure is
a per-hypothesis minimum over repetitions and rows are only comparable when
measured in one process.

The data root is `ENV["CAU_DATA_ROOT"]` (no default); heavy outputs land under
`<root>/accv2026/results/<experiment>/`, named after the driver. The `.sbatch`
launchers are examples: adapt their resource requests to your cluster.
