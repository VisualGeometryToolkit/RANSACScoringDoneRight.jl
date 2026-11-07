# REPRODUCE — cvpr2027 ("RANSAC Scoring Done Extremely Right")

Every figure, table and number macro the paper reads is produced by a driver
under `scripts/` from its config of record under `configs/`, and indexed
below. The suite (`test/runtests.jl`, `paper_project_violations`) checks the
index against what the paper reads.

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/<driver>.jl --plan     # what is done, what is pending
julia --project=. scripts/<driver>.jl            # run the pending stages
```

## Data

Inputs are read from `data_root()` (read-only: `VGTK_DATA_ROOT`, else
`data/exp_raw`, else the artifact view of the published data, see
`configs/data_artifacts.toml`); results are written under `results_root()`
(`VGTK_RESULTS_ROOT`, else `data/exp_pro`), one directory per run key, each
with its `_config.toml` and `_provenance.toml`.

## Asset index

| label | asset | kind | generator | config | data | provenance |
|---|---|---|---|---|---|---|
| `tab:synthetic` | `paper/figures/tab_synthetic.tex` | generated | `scripts/synthetic.jl` | `configs/synthetic.toml` | synthetic draws ([simulate], [[cases]]) | `_provenance.toml` of the run |
| — | `paper/figures/synthetic_numbers.tex` | generated | `scripts/synthetic.jl` | `configs/synthetic.toml` | synthetic draws ([simulate], [[cases]]) | `_provenance.toml` of the run |
| `tab:timing` | `paper/figures/tab_timing.tex` | generated | `scripts/score_timing.jl` (harness: `../neurips2026/scripts/studies/score_timing.jl`) | `configs/score_timing.toml` | the PT-E format-5 pool of [pool] under `data_root()`: not a published artifact yet — LocalOptimizationDoneRight `accv2026/scripts/pt_e_pools.jl` generates it from the published `pt_raw_validation` / `pt_raw_test` inputs (neurips2026 `configs/data_artifacts.toml`) | `_provenance.toml` of the run |
| — | `paper/figures/timing_numbers.tex` | generated | `scripts/score_timing.jl` | `configs/score_timing.toml` | as `tab:timing` | `_provenance.toml` of the run |
