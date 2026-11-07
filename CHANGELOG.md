# RANSACScoringDoneRight changelog

## 0.5.0 (unreleased)

The VisualGeometryToolkit 0.7 stack (taken, until it registers, from the
monorepo checkout through a temporary `[sources]` block; compat keeps
VisualGeometryToolkitCore `"0.6"` until the monorepo's version fields are bumped).

### Breaking

- **The loss-width families carry a loss width**, as VisualGeometryToolkitCore's
  `RansacScoring{DG, T}`/`MsacScoring{DG, T}` do: `MlesacScoring{DG, T}`,
  `MagsacScoring`, `WelschCostScoring`, `SmoothTLSScoring` and `GemanScoring`
  are width-free (`F()`, `F(dg)`, `F(problem)`, `T = Nothing`) or at a loss width
  (`F(dg; threshold = w)`, `with_width(F(dg), LossWidth(w))`, `T = Float64`), and
  at a loss width they implement `score!` (`scoring_workspace` is `nothing`; no
  consensus set). VisualGeometryToolkitCore's `best_pool_model` and
  VisualGeometryToolkitEval's sample-budget engine (`LossWidthSpec`) select with
  them.
- **Removed: `pool_model_score` and the 4-argument `best_pool_model(scoring, pair,
  R, loss_width)`.** Migration: `best_pool_model(F(), pair, R, w)` →
  `best_pool_model(with_width(F(dg), LossWidth(w)), pair, R)`; a model's score is
  `score!` of the scorer at its width on its residual column, its minimal sample
  given. The pool argmax is VisualGeometryToolkitCore's `Pools.best_model_index`
  (a non-finite score, `+Inf` included, never wins) and the shape check its
  `Pools.check_residual_matrix`.
- **`loss_width_scoring` takes a tag of the new `SCORING_TAGS` (public)**, the
  scoring families' `TagRegistry`: VisualGeometryToolkitCore's `"ransac"`,
  `"msac"` merged with `"mlesac"`, `"magsac"`, `"welsch_cost"`, `"smooth_tls"`,
  `"geman"`, `"student_t_uniform"` and `"sim_fit_pp"`; the retired `"gau"`,
  `"welsch"` and `"simfitpp"` name their successors. Migration:
  `loss_width_scoring(:msac, dg)` → `loss_width_scoring("msac", dg)`; it also
  takes a problem type.
- **Removed: `residual_unit_scale(pair)`** (the MEAN of the two pool-frame maps'
  scales). SIMFIT++'s grid carries its initial guess into pixels by
  VisualGeometryToolkitCore's `Pools.pool_pixel_scale(pair)`, the LARGER of the
  two. On pairs whose views differ in scale the initial guess moves by
  `max/mean − 1` (0.5–11 % for view-scale ratios 0.99–0.8); on 900 synthetic
  (pair, guess) draws per ratio the selected model changed in 1, 3 and 11 of 900
  (ratios 0.99, 0.95, 0.8) and the success flag in none.
- **`ScaleMarginal`'s prior is its `IGPrior`** (`m.prior.alpha0`,
  `m.prior.beta0`), not loose `alpha0`/`beta0` fields; the constructors are
  unchanged.
- The VisualGeometryToolkitCore 0.7 renames: `square_distances!`,
  `whitened_residual` (the signed residual), `sq_distances` (the `sq_residuals`
  argument names of `sweep!` and `loss_scale_bounds`).

### Fixed

- The covariance-given predictive `score_sq_residuals!` wrote the SIGNED whitened
  residual into its `residuals` buffer; it writes the Mahalanobis distance
  (`≥ 0`), what `mahalanobis_distances!` writes. Scores are unchanged.

### neurips2026

- The asset verify/deploy stages and the fragment comparison are
  VisualGeometryToolkitEval's `Reproduce` (`verify_assets(outdir, paperdir, names)`,
  `deploy_assets(…; replace_submitted)`, `compare_fragments`); the palette and
  its colour names are Eval's `default_palette`/`paper_color_name`; the
  Sensitivity bucketing is Eval's `results_by_method`.
- `sample_budget_miscal.jl` runs VisualGeometryToolkitEval's
  `evaluate_sample_budget` (selection only) in place of its own budget loop; its
  statistic of record is the engine's `model_imc` (it recorded `pool_imc`).
- The data-root variables are `VGTK_DATA_ROOT`/`VGTK_RESULTS_ROOT` everywhere.

## 0.4.1

VisualGeometryToolkitCore 0.6 (compat `"0.6"`): the breaking upstream release; no
change to this package's own code or public API. None of the names Core 0.6
retired or changed is used here, and a predictive model's covariance
(`scored_model`) keeps the full storage it held before. The research project
`neurips2026/` moves with it: VisualGeometryToolkitIO 0.6,
VisualGeometryToolkitEstimation 0.6.1 and VisualGeometryToolkitEval 0.5.

Until the stack's 0.6 series is registered, the workspace takes it from a
checkout of the monorepo beside this one through a temporary `[sources]`
block, removed at the release.
