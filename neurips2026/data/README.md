# data/

DrWatson's data directory, bound per machine (VisualGeometryToolkitEval Experiments):

| Tree | Resolved by | Access |
|---|---|---|
| `data_root()` — inputs | `CAU_DATA_ROOT`, else `data/exp_raw` (a link or a directory), else the artifact view of the published data (`configs/data_artifacts.toml`) | read-only: a driver never writes here |
| `results_root()` — results | `CAU_RESULTS_ROOT`, else `data/exp_pro` (created when absent) | add-only: one directory per run key, each with `_config.toml` and `_provenance.toml` |
| `datadir("sims")` | — | smoke runs, never read by a paper asset |

`data/exp_raw`, `data/exp_pro` and `data/sims` are ignored by git. Configs name
inputs relative to `data_root()` and stored results relative to
`results_root()`; no code names a machine path.

On the author's host both are links: `exp_raw` to the immutable input root
(the PhotoTourism, HEB and ETH3D trees the pool configs' `[source]` tables
name) and `exp_pro` to the paper's results root, which holds
`stored/paper-runs-2026-05-06/` (the six evaluated runs behind Fig. 1, Fig. 2
and the Sec. 4 numbers, with `PROVENANCE.toml` and `SHA256SUMS`) and the keyed
run directories of the drivers (`figures/fig_profile-<key>/`, …).

## A machine without the data: the published artifacts

```
julia --project=. scripts/fetch_data.jl          # the required artifact: the stored runs
julia --project=. scripts/fetch_data.jl --all    # also the raw PhotoTourism / HEB / ETH3D inputs (~100 GB)
julia --project=. scripts/fetch_data.jl --list   # the catalog, nothing fetched
```

downloads them once into the Julia depot (sha256 and tree hash verified) and
links them into `data/exp_raw` (an artifact view) and `data/exp_pro`. An
existing binding always wins and is never written to. The stored runs'
Zenodo record is not deposited yet (its URL carries the placeholder
`ZENODO_RECORD_NEURIPS2026`, which the fetch refuses); the raw inputs are the
deposited records shared with accv2026. `NEURIPS2026_ARTIFACTS_TOML=<file>`
points the fetch at another copy of the bindings (same hashes, other URLs).
