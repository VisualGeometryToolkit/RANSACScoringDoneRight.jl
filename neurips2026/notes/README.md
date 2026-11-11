# notes/

Write-ups of the scoring studies in `../scripts/studies/` (formerly the
package's top-level `results/`), checked in so a result and the code that
produced it travel together. They are not paper assets: the paper's figures
and their generators are indexed in `../REPRODUCE.md`.

| File | Experiment | Job |
|---|---|---|
| `score_timing.md` | `neurips2026/scripts/studies/score_timing.jl` — per-hypothesis CPU scoring cost | 4839 (reproduced by 4840) |

Each write-up names the driver that produced it and the JLD2 it was read from.
The JLD2 data itself is not committed, since it is regenerable from the
committed driver and does not belong in the package.

Regenerate any of these through the paper's stage graph
(`bash ../scripts/reproduce/reproduce_results_dgx.sh --only=<study>`), which
runs the driver on its config of record (`../configs/<study>.toml`: the protocol
of the published numbers and its declared inputs). Changing `npairs` or `reps`
there invalidates cross-row comparisons, because each figure is a
per-hypothesis minimum over repetitions and rows are only comparable when
measured in one process. A run lands in its own keyed directory under
`results_root()` (`<dataset>/<matcher>/<problem>/<study>/<kind>-<run key>/`);
the write-ups' JLD2s predate that and stayed where they were written.
