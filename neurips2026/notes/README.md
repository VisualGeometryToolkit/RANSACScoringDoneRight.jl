# notes/

Write-ups of the package's scoring studies (formerly the package's top-level
`results/`). The studies were retired in 0.8.3 (`../REPRODUCE.md`, "Scoring
studies (retired)"); their drivers and configs of record are in the
repository's history up to tag `v0.8.2`, so each write-up still names the
driver and the data it was produced from. They are not paper assets: the paper's figures
and their generators are indexed in `../REPRODUCE.md`.

| File | Experiment | Job |
|---|---|---|
| `score_timing.md` | `neurips2026/scripts/studies/score_timing.jl` — per-hypothesis CPU scoring cost | 4839 (reproduced by 4840) |

Each write-up names the driver that produced it and the JLD2 it was read from.
The JLD2 data itself is not committed.
