# cvpr2027 — "RANSAC Scoring Done Extremely Right" (CVPR 2027)

The follow-up to *RANSAC Scoring Done Right* (`../neurips2026`). That paper
integrates the inlier scale σ out of the RANSAC score; one length parameter is
left, the half-width `a` of the uniform outlier density. This project removes
it: an extreme-value (peaks-over-threshold) fit to the upper tail of each
partition's complement estimates the outlier support — as the half-width of
the Gaussian–uniform score, or as a generalized Pareto outlier component of
the mixture itself. With Jeffreys' prior on σ the score has no parameter with
units of length, and its ranking is invariant to the units of the residuals.

The paper project of RANSACScoringDoneRight for that paper: the manuscript,
the drivers that produce its tables and figures, their configs of record, and
the asset index. A [DrWatson](https://juliadynamics.github.io/DrWatson.jl/)
project laid out by the stack's paper-project conventions
(VisualGeometryToolkitEval `PaperProjects`, `new_paper_project`; the sibling
of `neurips2026/`, LocalOptimizationDoneRight's `accv2026/`, BlobBoards'
`bmvc2026/`), not a package: the package in `..` never depends on it.

```
cd cvpr2027
julia --project=. scripts/synthetic.jl --plan     # a driver's stages: done / pending
julia --project=. scripts/synthetic.jl            # run what is pending
julia --project=. test/runtests.jl                # the conventions + the project's tests
make -C paper                                     # the PDF
```

| Path | Holds |
|---|---|
| `paper/` | the manuscript (`main.tex`, `main.bib`, the CVPR style), `Makefile`; generated tables and figures under `paper/figures/` |
| `REPRODUCE.md` | the asset index: every table, figure and macro the paper reads → its generator, config and data |
| `configs/` | the configs of record (TOML): `synthetic` (Sec. 4.1), the data catalog (`data_artifacts`) |
| `scripts/` | the drivers, stage-idempotent (`--plan`, `--stage=NAME`, `--force`) |
| `src/` | the shared prelude (`CVPR2027.jl`) and the extreme-value pieces of the score (`evt.jl`) |
| `test/` | the suite: `paper_project_violations` |
| `data/exp_raw`, `data/exp_pro` | this machine's input and results trees (links or directories, never tracked; `data/README.md`) |
| `notes/` | working notes (`idea.md`: the estimators tried and what they showed) |

`Project.toml` is a member of the repository's workspace: the package, its
tests, its docs, `neurips2026` and this project resolve together in one
untracked Manifest at the repository root.
