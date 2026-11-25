# neurips2026 — "RANSAC Scoring Done Right" (NeurIPS 2026)

The paper project of RANSACScoringDoneRight: the manuscript, the drivers that
produce its figures, their configs of record, and the asset index that ties
every figure the paper reads to its generator and data. A
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/) project laid out by
the stack's paper-project conventions (VisualGeometryToolkitEval `PaperProjects`;
LocalOptimizationDoneRight's `accv2026/` is the sibling project), not a
package: the package in `..` never depends on it.

```
cd neurips2026
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/fetch_data.jl               # a machine without the data: the published runs
julia --project=. scripts/fig_profile.jl --plan       # a driver's stages: done / pending
julia --project=. scripts/fig_profile.jl              # run what is pending
julia --project=. test/runtests.jl                    # the conventions + the project's tests
make -C paper                                         # the PDF
```

| Path | Holds |
|---|---|
| `paper/` | the manuscript as posted to arXiv (`main.tex`, `main.bib`, the NeurIPS style), `Makefile`; generated figures under `paper/figures/` |
| `REPRODUCE.md` | the asset index: every figure the paper reads → its generator, config and data, with its verification status; what is not ported, and the reruns it would take |
| `configs/` | the configs of record (TOML): the figures (`fig_profile`, `fig_sensitivity`), the prose-number check (`paper_numbers`), the stage graph (`reproduce`), the pools of record (`*_pools`), the data catalog (`data_artifacts`) |
| `scripts/` | the drivers, stage-idempotent (`--plan`, `--stage=NAME`, `--force`); `scripts/reproduce/` the one-command reproduction per machine (`configs/reproduce.toml`) |
| `src/` | the shared prelude (`NeurIPS2026.jl`): the panel renderers (`panels.jl`), the verify/deploy stages (`assets.jl`), the prose-number statistics (`numbers.jl`), the data catalog binding (`data_artifacts.jl`) |
| `test/` | the suite: `paper_project_violations`, and the renderers reproduce every submitted fragment byte for byte |
| `data/exp_raw`, `data/exp_pro` | this machine's input and results trees (links or directories, never tracked; `data/README.md`) |
| `Artifacts.toml`, `configs/data_artifacts.toml` | the published data (Zenodo) and their catalog |
| `notes/` | write-ups of the retired scoring studies (not paper assets; their drivers are in the history up to `v0.8.2`) |

`Project.toml` devs the package (`..`) and takes the rest of the stack from
the registry. It reproduces with RANSACScoringDoneRight 0.8.x and the
VisualGeometryToolkitCore 0.13 stack: run it in a checkout of the `v0.8.3` tag,
where it is a member of the repository's workspace. From 0.9.0 it is not (it
stays on its released environment until its run of record is archived).
