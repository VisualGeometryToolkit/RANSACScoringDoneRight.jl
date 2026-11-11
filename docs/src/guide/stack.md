# Working on the stack

```@meta
CurrentModule = RANSACScoringDoneRight
```

This page is for people who change this package or the packages around it. It
lists the VisualGeometryToolkitCore contracts the package implements, how the
GPU grid evaluators and the SIMFIT++ grid are organised, the vocabulary the
package adds to the stack's, and how the repository is laid out. The
[Overview](../overview.md) explains the same pieces from a user's point of view.

## Contracts implemented

VisualGeometryToolkitCore owns the contracts. This package implements them for
its own types and re-exports nothing VisualGeometryToolkitCore owns.

The scoring contract lives in VisualGeometryToolkitCore's
`robust/estimation.jl`. [`StudentTUniformScoring`](@extref VisualGeometryToolkitCore.StudentTUniformScoring) implements
`scoring_workspace` (its [`ScoringWorkspace`](@extref VisualGeometryToolkitCore.ScoringWorkspace)), `score!` (the
sort-and-sweep of Algorithm 1), `inlier_count`, `consensus_order`,
`scale_prior`, `outlier_halfwidth`, `check_workspace` and
`with_width`/`width_axis` (the axis is `OutlierHalfwidth`). Its
`decision_boundary` is the MAP crossover of the consensus set's ŝ² against the
uniform outlier box. `StudentTUniformScoring{Predictive}` also implements
`is_predictive` and `scored_model`. VisualGeometryToolkitCore derives the
consensus scale (`consensus_s2`), the soft inlier weights and the Student-t
posterior toolkit from these methods.

The pool-selection contract lives in VisualGeometryToolkitCore's
`robust/pool_selection.jl`. Every grid family of this package declares the
`:rsdr` staging through `selection_staging` and `stage_selection`, and
implements `select_on_grid!` over its device kernels. The families are the
marginal, the width-free loss-width families, [`LossSpec`](@ref),
`ScaleMarginal` and VisualGeometryToolkitEstimation's `SimFitPP`. VisualGeometryToolkitCore's
`best_pool_model(scoring, pair, R)` serves any scorer at its width; this
package's loss-width families take it at a loss width,
`with_width(F(dg), LossWidth(w))`, through their `score!`.

Problems implement VisualGeometryToolkitCore's `AbstractConstraintProblem`:
`data_size`, `codimension`, `minimal_sample_size` and
`constraint_with_jacobians`. Predictive scoring also needs
`solver_covariance`, declared by `provides_solver_covariance`. The RANSAC loop
and the concrete problems are VisualGeometryToolkitEstimation's. The
local-optimization strategies are VisualGeometryToolkitEstimation's and
LocalOptimizationDoneRight's. A caller composes them, for example

```julia
find_essential(camera1, camera2, cs;
               scoring = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => a);
                                           problem = EssentialProblem),   # priors from the stack defaults
               local_optimization = …)
```

RANSAC and MSAC at a threshold (`RansacScoring`, `MsacScoring`) are
VisualGeometryToolkitCore's baseline scorers. This package adds their grid
methods.

## GPU scoring with KernelAbstractions

The files `ig_sweep_ka.jl`, `loss_width_sweep_ka.jl` and `marg_sweep_ka.jl`
hold device kernels that run on the CPU and on CUDA. They score a pool that is
staged once with `stage_ig_shared_from_pairs(pairs, residuals; backend)` from
the light pool records (VisualGeometryToolkitCore's `PoolPair`) and their
residual matrices. One verb, [`grid_evaluate`](@ref), evaluates a grid and
dispatches on the scoring:

- `grid_evaluate([S,] loss_width_scoring("msac", dg), staged, loss_widths; aggregation)`
  evaluates the loss-width baselines. These are RANSAC, MSAC, GaU (MLESAC) and
  MAGSAC++, plus the Welsch, smooth-TLS and Geman–McClure losses. Each scores
  through one per-residual scalar, `residual_quality(scoring, d², loss_width)`.
  The `aggregation` is [`Exact`](@ref) (every residual, the reference) or
  [`LogBinned`](@ref) (`bins_per_decade = 200`, the fast cross-validation path
  and the default).
- `grid_evaluate(StudentTUniformScoring(problem_type, h; scale_prior, label_prior), staged, outlier_halfwidths)`
  is the IG/Jeffreys marginal sweep. The grid replaces the scoring's own `h`.
- `grid_evaluate([S,] loss_spec(shape, prior; Δ, …), staged, outlier_halfwidths; model_dof, dg)`
  is the Track-B redescending-loss mixture marginal.
- `grid_evaluate([S,] WelschMarginal(; prior_loss_width, outlier_halfwidth), staged, prior_loss_widths)`
  evaluates the full-data scale marginals. The grid replaces the prior centre.

VisualGeometryToolkitCore's `width_axis(scoring)` names the axis a scoring's
grid is native on (`OutlierHalfwidth` or `LossWidth`), and this package states
it for its grid families. VisualGeometryToolkitCore's
`to_native(axis, scale_multiple, scale; d_g)` maps a scale multiple onto that
axis, and `native_map(axis; d_g)` is the same map curried.

The kernels run on `CPU()` or `CUDA.CUDABackend()`. The Float64 device path is
bit-identical to the CPU reference; `test/gate_marg_mix_ka.jl` checks this and
also runs on CUDA with `--cuda`. Each grid evaluator frees its function-local
device buffers eagerly through `GPUArrays.unsafe_free!` instead of waiting for
Julia's garbage collector.

## SIMFIT++

SIMFIT++ (Edstedt, CVPRW 2025, [Parskatt/simfitpp](https://github.com/Parskatt/simfitpp))
is VisualGeometryToolkitEstimation's: its step-for-step port, `SimFitPP`, `simfitpp_loss_width`,
`simfitpp_select`, `unbiased_median_variance` and `with_residual_unit`, moved
there from this package on 2026-11-11 ([`SimFitPP`](@extref VisualGeometryToolkitEstimation.SimFitPP)).
This package gives `SimFitPP` a grid on the pool-selection contract: for every
initial loss width of the grid, `simfitpp_select` re-estimates the loss width on
the pool and selects the MSAC-best hypothesis at the estimate.

The reference's bounds, 0.25 and 8, are pixel constants. A pool stores its
data in `NormalizedImageFrame` (image centre, half diagonal), and no single
scalar relates the two frames point by point. VisualGeometryToolkitCore's `Pools.pixel_residuals(pair)`
is the image-plane Sampson residual of the pool's models in pixels. It is
computed from the light record and never reads the stored residual matrix, so
it does not depend on that matrix's residual kind (tangent Sampson for E
pools). The pair method of Estimation's `simfitpp_select` runs on it, so
`initial_loss_width_px`, the bounds and the returned loss width are all in
pixels. To work in the pool frame's unit instead, call the matrix method on the
pair's residual matrix with bounds in that unit.
VisualGeometryToolkitCore's `Pools.pool_pixel_scale(pair)`, the larger of the
two views' pool-frame-per-pixel scales, carries a pixel quantity into the pool
frame; SIMFIT++'s grid uses it for its initial guess only.

## Scoring and tuning vocabulary

This package holds the vocabulary and the per-pair machinery. That is the
scoring families' tag registry ([`ALL_SCORING_TAGS`](@ref)), the `score!` of its
loss-width families at a loss width (which VisualGeometryToolkitCore's
`best_pool_model` and VisualGeometryToolkitEval's sample-budget engine select
with), [`precompute_pair`](@ref) and the KernelAbstractions grids above. A
scoring at its width is its own pool-selection handle.

Other layers live elsewhere. The prior-spec layer that fits `IGPrior` from
validation data (`PriorSpec`, `fit_prior`) is VisualGeometryToolkitEval's
`Tuning`. The pool record, `PoolPair`, and its accessors (`num_points`,
`num_models`, `pool_imc`, `pool_normalization` and others) belong to
VisualGeometryToolkitCore's `Pools` module. Pool IO belongs to
VisualGeometryToolkitEval.

The layer is problem-agnostic. The concrete problem types
(`HomographyProblem`, `FundMatProblem`, `EssentialProblem`) stay in the
consumer. A scorer that needs one takes it as a leading `problem_type`
argument, as in
`best_pool_model(StudentTUniformScoring(problem_type, outlier_halfwidth; scale_prior, label_prior), pair, R)`.

## Names of quantities

VisualGeometryToolkitCore's robust layer is authoritative for the loss words.
The shape math is `ρ`, `ψ` and `weight`. A `scale_multiple` is the
dimensionless multiple of the scale at which a width sits, `s2` is the noise
variance, `loss_width` is a loss's scale parameter and `sq_loss_width` is its
square (`scale_multiple² · s2` for a deployed loss). VisualGeometryToolkitCore
writes the squared residual as `s` in `ρ(shape, s, sq_loss_width)`. This
package writes it `sq_r` or `sq_distances`, never `s`.

On top of that, this package names the following quantities.

| name | quantity |
|------|----------|
| `outlier_halfwidth` | the half-width of the uniform outlier component |
| `decision_boundary` | the residual at which the MAP rule flips |
| `decision_boundary` / `decision_width` | the one map between the two (VisualGeometryToolkitCore's generic, with the mixture's method), so a grid of outlier half-widths is never called a grid of thresholds |
| `operating_scale_multiple` | the scale multiple at which a protocol converts (`loss_width = operating_scale_multiple · ŝ`) |
| `ŝ`, `s_hat_sq` | the scale estimate and its square |
| `d_g` / `dg` | the constraint codimension |
| `inlier_prior` / `inlier_posterior` | the scalar mixing proportion π and the per-point `P(inlier \| r)` |

A name that contains `scale` returns a linear scale (`pool_pixel_scale`).
A name for a squared quantity says so (`unbiased_median_variance`, `s_hat_sq`,
`sq_loss_width`). The one exception is `fit_scale`, which is named for the
`fit` contract; its docstring states first that it returns the square.

## Repository layout

```
src/                   the package: the sort-and-sweep marginal scorer, the grid
                       evaluators (KernelAbstractions), the baseline families, SIMFIT++'s grid
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

The package, its tests, its docs and the research project are one Pkg
workspace. They resolve together into one `Manifest.toml` at the repository
root, which is not committed.

The split follows one rule. What a second paper or a user's RANSAC reuses (the
scorers, their grids, the per-pair selection) belongs in the package. Drivers,
figures, configs of record and stored results belong in `neurips2026/`. The
package never depends on the project.

## Development

The stack's git hooks keep the local `Manifest.toml` resolvable: after a pull,
checkout or rebase that changes the `Project.toml` of any workspace member,
they re-resolve the workspace. They only resolve; precompilation waits for the
next `using`. They are VisualGeometryToolkitCore's (its `githooks/`, one copy for
every stack repository) and opt-in, once per clone:

```bash
julia --project=. -e 'using VisualGeometryToolkitCore; VisualGeometryToolkitCore.install_githooks()'
```

The package's tests use ReTestItems, one item per `test/*_tests.jl` file.
`Pkg.test(; test_args = ["select_on_grid"])` runs the items whose name matches. The
research project's suite is separate:
`cd neurips2026 && julia --project=. test/runtests.jl`.
