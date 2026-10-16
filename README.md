# RansacScoringDoneRight.jl

Code for the NeurIPS 2026 paper "RANSAC Scoring Done Right", in two halves:
the registered package `RansacScoringDoneRight` — scale-free marginal RANSAC
scoring (`StudentTUniformScoring`), its CPU/GPU grid evaluators and the
paper's baseline scorers, an implementation of VisualGeometryCore's scoring and
pool-selection contracts that depends only on VisualGeometryCore — and the
research project `neurips2026/` that holds the paper and produces its figures.
Both, with the package's tests and docs, are one Pkg workspace: they resolve
together, in one `Manifest.toml` at the repository root that is not committed.

Documentation and project page: <https://ransacscoringdoneright.github.io>.

## Installation

```julia
using Pkg
Pkg.add("RansacScoringDoneRight")
```

Requires Julia 1.12. Part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack; built on
[VisualGeometryCore.jl](https://github.com/VisualGeometryToolkit/VisualGeometryCore.jl).

## Contracts implemented

VisualGeometryCore owns the contracts; this package implements them for its
own types and re-exports nothing VisualGeometryCore owns.

- **Scoring contract** (VGC `robust/estimation.jl`). `StudentTUniformScoring`
  implements `scoring_workspace` (its `ScoringWorkspace`), `score!` (the
  sort-and-sweep, Algorithm 1), `inlier_count`, `consensus_order`,
  `inlier_threshold` (the MAP crossover of the consensus set's ŝ² against the
  uniform outlier box), `scale_prior`, `outlier_halfwidth`, `check_workspace`,
  `with_width`/`width_axis` (`OutlierHalfwidth`), and, for
  `StudentTUniformScoring{Predictive}`, `is_predictive` and `scored_model`.
  The consensus scale (`consensus_s2`), the soft inlier weights and the
  Student-t posterior toolkit are derived in VisualGeometryCore from these.
- **Pool-selection contract** (VGC `robust/pool_selection.jl`). Every grid
  family of this package (the marginal, the width-free loss-width families,
  `LossSpec`, `ScaleMarginal`, `SimFitPP`) declares the `:rsdr` staging
  (`selection_staging`, `stage_selection`) and implements `select_on_grid!`
  over its device kernels. `best_pool_model(scoring, pair, R)` is
  VisualGeometryCore's for any scorer at its width; this package adds the
  method for its width-free loss-width families.

Problems implement VisualGeometryCore's `AbstractConstraintProblem` (`data_size`, `codimension`,
`minimal_sample_size`, `constraint_with_jacobians`; optionally `solver_covariance`,
declared by `provides_solver_covariance`, for predictive scoring). The RANSAC
loop and the problems are RobustVisualGeometry's, the local-optimization
strategies RobustVisualGeometry's and LocalOptimizationDoneRight's; a caller
composes them, e.g.
`find_essential(camera1, camera2, cs; scoring = StudentTUniformScoring(EssentialProblem, a), local_optimization = …)`.
RANSAC and MSAC at a threshold (`RansacScoring`, `MsacScoring`) are
VisualGeometryCore's baseline scorers; this package adds their grid methods.

## GPU scoring (KernelAbstractions)

`ig_sweep_ka.jl` / `loss_width_sweep_ka.jl` / `marg_sweep_ka.jl` provide
CPU+CUDA-portable device kernels over a pool staged once with
`stage_ig_shared_from_pairs(pairs, residuals; backend)` — the light pool
records (VisualGeometryCore's `PoolPair`) and their residual matrices. ONE verb
evaluates a grid, dispatched on the scoring:

- `grid_evaluate([S,] loss_width_scoring(:msac, dg), staged, loss_widths; aggregation)`
  — the loss-width baselines (RANSAC / MSAC / GaU (MLESAC) / MAGSAC++ and the
  Welsch / smooth-TLS / Geman-McClure losses), over ONE per-residual scalar,
  `residual_quality(scoring, d², loss_width)`. `aggregation` is `Exact()` (every
  residual — the reference) or `LogBinned(bins_per_decade = 200)` (the fast
  cross-validation path, the default).
- `grid_evaluate(StudentTUniformScoring(problem_type, h; prior), staged, outlier_halfwidths)` —
  the IG/Jeffreys marginal sweep (the grid replaces the scoring's own `h`).
- `grid_evaluate([S,] loss_spec(shape, prior; Δ, …), staged, outlier_halfwidths; model_dof, dg)`
  — the Track-B redescending-loss mixture marginal.
- `grid_evaluate([S,] WelschMarginal(; prior_loss_width, outlier_halfwidth), staged, prior_loss_widths)`
  — the full-data scale marginals (the grid replaces the prior centre).

VisualGeometryCore's `width_axis(scoring)` names the width axis a scoring's
grid is native on (`OutlierHalfwidth` or `LossWidth`; this package states it for
its grid families), `to_native(axis, scale_multiple, scale; d_g)` maps a scale
multiple onto it, and `native_map(axis; d_g)` is that map curried (both
VisualGeometryCore's).

Built on `KernelAbstractions` (runs on `CPU()` or `CUDA.CUDABackend()`); the
Float64 device path is bit-identical to the CPU reference
(`test/gate_marg_mix_ka.jl`, which also runs on CUDA with `--cuda`). Each grid
evaluator frees its own function-local device buffers eagerly, through
`GPUArrays.unsafe_free!`, rather than waiting for Julia's GC.

## SIMFIT++ loss-width estimation

`src/simfitpp.jl` ports Edstedt's SIMFIT++ (CVPRW 2025, github.com/Parskatt/simfitpp)
step for step (the reference calls the loss width its threshold):
`unbiased_median_variance` is the reference's `UnbiasedMed` (median of the
squared residuals below the current loss width, de-biased for the truncation by the χ²
fixed-point quantile iteration — it returns σ², hence the name), and
`simfitpp_loss_width` is `SIMFITPP.__call__`
(random train/validation split, geometry on the train split at the current loss width,
held-out residuals → variance → loss width `sqrt(ppf(alpha)·σ²)`, bounds, running mean,
`ftol`/repeat stopping, final bounds check, cross-pair log-space filtering). The
geometry estimator is a callback pair, so the loop runs on a full RANSAC (the
reference) or on a hypothesis pool: `simfitpp_select(pair, initial_loss_width_px)` plays
the estimator with the MSAC-best pool hypothesis on the train split and returns
the MSAC-best hypothesis over all points at the estimated loss width. `SimFitPP(; ...)`
holds the reference defaults (`alpha = 0.99`, `max_iter = 4`, `ftol = 0.01`,
`train_fraction = 0.5`, `min_loss_width = 0.25`, `max_loss_width = 8`, `dg = 1`).

**Frames.** The reference's bounds (`min_loss_width = 0.25`, `max_loss_width = 8`) are pixel
constants, while a pool stores its data in `NormalizedImageFrame` (image
centre, half diagonal; no single scalar relates the two frames per point).
`pixel_residuals(pair)` is the image-plane Sampson residual of the pool's
models in pixels, computed from the light record (it never reads the stored
residual matrix, whose kind — tangent Sampson for E pools — it therefore does
not depend on), and the pair method of `simfitpp_select` runs on it, so
`initial_loss_width_px`, the bounds and the returned loss width are all pixels.
To work in the pool frame's unit instead, call the matrix method on the pair's
residual matrix with bounds in that unit. `residual_unit_scale(pair)` (the
mean pool-frame-per-pixel scale) is a convention for soft quantities only, e.g.
an initial guess taken from a pool-frame-unit grid.

## Scoring + tuning vocabulary

What lives here is the **vocabulary and the per-pair machinery**: the per-pair
selection over a pool record and its residual matrix (`best_pool_model` for
the width-free loss-width families, `pool_model_score`, `precompute_pair`)
and the KernelAbstractions grids above. A scoring at its width is its own
pool-selection handle. The prior-spec layer that fits `IGPrior` from
validation data (`PriorSpec`, `fit_prior`) is VisualGeometryEval's `Tuning`. The pool record
itself — `PoolPair` and its accessors (`num_points`, `num_models`, `pool_imc`,
`pool_normalization`, …) — is VisualGeometryCore's (`VisualGeometryCore.Pools`);
pool IO is the evaluation package's.

This layer is **problem-agnostic**: concrete problem types
(`HomographyProblem`/`FundMatProblem`/`EssentialProblem`) stay in the consumer
and are given as a leading `problem_type` argument where a scorer needs one
(`best_pool_model(StudentTUniformScoring(problem_type, outlier_halfwidth), pair, R)`, VisualGeometryCore's
per-pair argmax of a scorer at its width).

## Vocabulary

VisualGeometryCore's robust layer is authoritative for the loss words: the
shape math `ρ`/`ψ`/`weight`, `scale_multiple` the dimensionless multiple of the
scale a width sits at, `s2` the noise VARIANCE, `loss_width` a loss's scale
parameter and `sq_loss_width` its square (`scale_multiple² · s2` for a deployed
loss). VGC spells the squared RESIDUAL `s` in `ρ(shape, s, sq_loss_width)`; this
package spells it `sq_r` / `sq_residuals` and never `s`.

On top of that RSDR owns one layer:

| name | quantity |
|------|----------|
| `outlier_halfwidth` | the half-width of the uniform outlier component |
| `decision_threshold` | the residual at which the MAP rule flips |
| `decision_threshold` / `decision_width` (VGC's generic, the mixture's method) | the ONE map between the two — so a grid of outlier half-widths is never called a grid of thresholds |
| `operating_scale_multiple` | the scale multiple a protocol converts at (`loss_width = operating_scale_multiple · ŝ`) |
| `ŝ`, `s_hat_sq` | the scale estimate and its square |
| `d_g` / `dg` | the constraint codimension |
| `inlier_prior` / `inlier_posterior` | the scalar mixing proportion π / the per-point `P(inlier \| r)` |

A name containing `scale` returns a LINEAR scale (`residual_unit_scale`);
anything squared says so (`unbiased_median_variance`, `s_hat_sq`,
`sq_loss_width`) — except `fit_scale`, named for the `fit`
contract, whose docstring states the square first.

## Repository layout

```
src/                   the package: the sort-and-sweep marginal scorer, the grid
                       evaluators (KernelAbstractions), the baseline families, SIMFIT++
test/                  the package's tests (ReTestItems)
docs/                  the documentation site (Documenter)
neurips2026/           the research project (DrWatson layout, devs this package):
  Project.toml         every dependency the drivers need (`@quickactivate "NeurIPS2026"`)
  paper/               the manuscript (arXiv version); generated figures in paper/figures/
  scripts/             the figure drivers (stage-idempotent), the data fetch,
                       studies/ (the scoring studies) and slurm/ (their launchers)
  configs/             configs of record (figures, prose-number check, pools, data catalog)
  src/                 the shared prelude: the ported panel renderers, verify/deploy
  test/                the project's tests (`julia --project=. test/runtests.jl`)
  data/                exp_raw / exp_pro bindings (per machine, not tracked)
  notes/               write-ups of the scoring studies
  REPRODUCE.md         the figure → generator → config → data index, and what is not ported
```

The split rule: what a second paper or a user's RANSAC reuses — the scorers,
their grids, the per-pair selection — belongs in the package; drivers,
figures, configs of record and stored results belong in `neurips2026/`. The
package never depends on the project.

## Reproducing the paper

```bash
cd neurips2026
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/fetch_data.jl          # a machine without the data (neurips2026/data/README.md)
julia --project=. scripts/fig_profile.jl --plan
julia --project=. test/runtests.jl
make -C paper
```

`neurips2026/REPRODUCE.md` maps every figure of the paper to its driver, config
of record and data. The figures regenerate byte for byte from the six stored
evaluation runs of record; the evaluation that produced those runs is
documented there but not yet ported to the current stack.

## Development

The repository ships git hooks in `.githooks/` that keep your local
`Manifest.toml` resolvable: after a pull, checkout or rebase that changes any
workspace member's `Project.toml`, they re-resolve the workspace (resolve only; precompilation
waits for the next `using`). They are opt-in, once per clone:

```bash
git config core.hooksPath .githooks
```

## License

MIT
