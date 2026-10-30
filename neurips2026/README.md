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
julia --project=. -e 'using Pkg; Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl"); Pkg.instantiate()'
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
| `configs/` | the configs of record (TOML): the figures (`fig_profile`, `fig_sensitivity`), the prose-number check (`paper_numbers`), the pools of record (`*_pools`), the data catalog (`data_artifacts`) |
| `scripts/` | the drivers, stage-idempotent (`--plan`, `--stage=NAME`, `--force`); `scripts/studies/` the scoring studies (write-ups in `notes/`), `scripts/slurm/` their launchers |
| `src/` | the shared prelude (`NeurIPS2026.jl`): the panel renderers (`panels.jl`), the verify/deploy stages (`assets.jl`), the prose-number statistics (`numbers.jl`), the data catalog binding (`data_artifacts.jl`) |
| `test/` | the suite: `paper_project_violations`, and the renderers reproduce every submitted fragment byte for byte |
| `data/exp_raw`, `data/exp_pro` | this machine's input and results trees (links or directories, never tracked; `data/README.md`) |
| `Artifacts.toml`, `configs/data_artifacts.toml` | the published data (Zenodo) and their catalog |
| `notes/` | write-ups of the scoring studies (not paper assets) |

`Project.toml` devs the package (`..`) and takes the rest of the stack from
the registry. It is a member of the repository's workspace: the package, its
tests, its docs and this project resolve together in one untracked Manifest at
the repository root.
