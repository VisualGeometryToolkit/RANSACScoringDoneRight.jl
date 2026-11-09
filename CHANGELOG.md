# RANSACScoringDoneRight changelog

## 0.6.0 (2026-11-09)

On the VisualGeometryToolkit 0.8 stack, from General: VisualGeometryToolkitCore 0.8
(neurips2026 also Estimation 0.8 and IO 0.8; Eval 0.6.1).

### Breaking

- **`StudentTUniformScoring{P, DG, L, O}`**: the label prior `L` and the outlier
  model `O` are type parameters (was `{P, DG}`), so the sweep dispatches on them
  statically. A field or method annotated `StudentTUniformScoring{P, DG}` now names
  an abstract type. The constructors keep their forms: `outlier` (was
  `outlier_halfwidth`) is a number or `FixedHalfwidth` (the profile-marginal score,
  as before), and `label_prior` defaults to `FlatLabelPrior()`.

### New

- Label priors: `FlatLabelPrior` (γ = ½), `ProfiledLabelPrior` (γ̂ = nᵢ/N) and
  `BetaLabelPrior` (γ integrated out); `label_prior(scoring)`.
- Outlier models: `ExtremeValueHalfwidth` (the half-width estimated from each
  partition's complement), `GeneralizedParetoOutliers{:prefix | :hypothesis}`
  (the complement scored under the fitted generalized Pareto law), `TailRules`
  and `OutlierTerm`; `outlier_model(scoring)`; public `gpd_pwm_fit`, `tail_fit`,
  `gpd_support`, `gpd_outlier_cost`. The extreme-value sweep reuses the sweep's
  radix sort and allocates nothing.

## 0.5.0 (2026-11-07)

The VisualGeometryToolkit 0.7 stack, from General: VisualGeometryToolkitCore 0.7
(neurips2026 also Estimation 0.7, Eval 0.6 and IO 0.7).

### Breaking

- **The loss-width families carry a loss width**, as VisualGeometryToolkitCore's
  `RansacScoring{DG, T}`/`MsacScoring{DG, T}` do: `MlesacScoring{DG, T}`,
  `MagsacScoring`, `WelschCostScoring`, `SmoothTLSScoring` and `GemanScoring`
  are width-free (`F()`, `F(dg)`, `F(problem)`, `T = Nothing`) or at a loss width
  (`F(dg; threshold = w)`, `with_width(F(dg), LossWidth(w))`, `T = Float64`).
  VisualGeometryToolkitCore's `best_pool_model` and VisualGeometryToolkitEval's
  sample-budget engine (`LossWidthSpec`) select with them.
- **The five families are VisualGeometryToolkitCore `AbstractThresholdScoring{T}`s,
  and this package's copy of the threshold loop is deleted.** `MlesacScoring` and
  `MagsacScoring` subtype it directly (they leave `GauScoring`, which now holds
  `StudentTUniformScoring` only; nothing dispatched on it), and `LossCostScoring{T}
  <: AbstractThresholdScoring{T}` (it gains its width's type parameter). At a loss
  width they score through Core's one `score!` on a `ThresholdScoringWorkspace`
  (no longer `nothing`), bit for bit the same score, and they gain what the loop
  gives: a consensus set (`d² < w²`, the sample's points in it; `inlier_count`,
  `consensus_order`, `inlier_threshold`, `has_inlier_threshold`), and an early exit
  against a running best (`score_bound`: 1 for MLESAC, Welsch, smooth TLS and
  Geman–McClure; the quality at `d = 0` for MAGSAC++). The exit returns `-Inf`
  only for a model that provably cannot beat the bound, so no selection changes.
  Migration: `score!(nothing, sc, …)` → `score!(scoring_workspace(sc, n, d_g), sc, …)`.
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
