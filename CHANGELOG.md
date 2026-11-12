# RANSACScoringDoneRight changelog

## Unreleased (the VisualGeometryToolkit 0.9 stack)

On the VisualGeometryToolkit 0.9 stack: VisualGeometryToolkitCore 0.9, and
VisualGeometryToolkitEstimation (a new dependency, for SIMFIT++ alone);
neurips2026 also Eval and IO of that stack.

### Breaking

- **Renamed (VisualGeometryToolkitCore's and Estimation's, the owner's rulings): the robust-estimation contract's names.**
  A width's multiple of the scale is the `efficiency_constant` (was `scale_multiple`; the `EfficiencyConstant` width axis,
  was `ScaleMultiple`; the protocol's `operating_efficiency_constant`), always the plain multiple; find_*'s strategy
  keyword is `local_optimization_strategy`, and no local optimization is `nothing` (no `NoLocalOptimization`). This
  package's docs, tests and the neurips2026 sample-budget study follow. No scorer changes: results bit for bit.
- **Renamed (VisualGeometryToolkitCore's, the owner's ruling): the scorer's decision point is `decision_boundary`.**
  `inlier_threshold` → `decision_boundary`, `has_inlier_threshold` → `has_decision_boundary`, and the Gaussian–uniform
  `decision_threshold` is a method of the same `decision_boundary`. This package's method for a Student-t scorer with an
  `OutlierTerm` (no decision boundary) and MAGSAC++'s `score_bound` follow; the docs and tests too. Names only: results
  bit for bit.
- **Renamed: `TailRules`' `threshold_quantile` is `pot_threshold_quantile`** (the owner's ruling): the quantile level
  of the complement's residuals at which the peaks-over-threshold threshold u is set, the extreme-value literature's
  name (Coles 2001, §4.3; Hosking & Wallis 1987). The keyword and field of `TailRules` (and the keyword
  `ExtremeValueHalfwidth`/`GeneralizedParetoOutliers` forward to it) and configs/default.toml's `[tail_rules]` key.
  The old name is refused with an `ArgumentError` naming the new one, as a keyword and as a `[tail_rules]` key (this
  package has no retired-names list). Same value (0.9): results bit for bit. `min_excess` and `finite_shape_limit`
  are unchanged.
- **New: configs/default.toml, the stack's defaults mechanism** (the owner's ruling, via the coordinator; as
  LocalOptimizationDoneRight's): `TailRules`' keyword defaults (`[tail_rules]`: `threshold_quantile` 0.9,
  `min_excess` 5, `finite_shape_limit` -1e-3), the scale marginals' log-scale span `_MARG_L` (`[scale_marginal]`
  `log_scale_span` 7.0) and the canonical well depths `_default_depth` (`[inlier_potentials.depth]`, by loss tag:
  welsch 0.5, geman_mcclure 0.5, smooth_tls 1.0) are read there through `_config_default` (VisualGeometryToolkitCore's
  `toml_default`); the code holds no second copy. Same values: results bit for bit. MAGSAC++'s DOF = 4 stays in code
  (the owner's ruling: part of the scorer's definition, not configurable).
- **Moved: the Student-t + uniform marginal scorer is VisualGeometryToolkitCore's** (the owner's ruling: hoisted into
  the stack with no duplication; temporary [sources] to the unregistered 0.9 stack). `StudentTUniformScoring`
  (model-certain and `Predictive`), its sweep (`sweep!`, the parametric sweep, `partition_score`, the one prefix
  kernel `ig_sweep_kernel!`, the radix sort), `ScoringWorkspace` and its accessors (`sweep_order`, `penalties`,
  `sort_key`, `lg_table`), `alloc_scoring_workspace`, `default_scoring_workspace`, `score_residuals!`,
  `score_sq_residuals!`, the label priors (`AbstractLabelPrior`, `FlatLabelPrior`, `ProfiledLabelPrior`,
  `BetaLabelPrior`), `AbstractOutlierModel` and `FixedHalfwidth` moved there, and src/sweep.jl with them; this package
  no longer exports them (it re-exports nothing VGC owns): import them from VisualGeometryToolkitCore. Its tests of
  the moved code (radix_sortperm, parametric_sweep, partition_score) moved with it. Renamed with the move: the
  constructors' `prior` keyword is `scale_prior`; `_build_lg_table` → `log_gamma_table`, `_build_prior_table` →
  `label_prior_table`, `_radix_sortperm!`/`_radix_key` → `radix_sortperm!`/`radix_key`, `_suffix_sums!` →
  `suffix_sums!` (public there). The empty `GauScoring` supertype is gone (nothing dispatched on it).
- **Breaking: the scorer's priors are required** (the owner's ruling (c)): `StudentTUniformScoring(…; scale_prior,
  label_prior)` has no defaults, nor `default_scoring_workspace`'s `scale_prior` or `BetaLabelPrior`'s parameters
  (`BetaLabelPrior(0.5, 0.5)` is the former default). The stack's default (Jeffreys, flat) is VisualGeometryToolkitEstimation's
  `configs/default.toml` `[scoring.student_t_uniform]`, read by its `scoring_from_toml`, which now reads the Student-t
  table (`outlier_halfwidth`, `scale_prior`, `label_prior` and their parameters) and `config_dict` writes it; this
  package's methods for it are deleted, and `"student_t_uniform"` is Core's `SCORING_TAGS`' (still in
  `ALL_SCORING_TAGS`, the merge). Downstream callers (BlobBoards, CaliBlobs, RadiometricCamera) switch to Core's names.
- **Kept here, extending Core's types by dispatch**: the outlier models that estimate the support (`TailRules`,
  `ExtremeValueHalfwidth`, `GeneralizedParetoOutliers`, `OutlierTerm`, the generalized Pareto fit) implement Core's
  estimated-support hooks (`outlier_cost`, `scored_outlier_halfwidth`, `outlier_context!`); the marginal grid
  (`grid_evaluate(::StudentTUniformScoring, …)`, `stage_ig_shared_from_pairs`, `select_on_grid!`) launches Core's
  `ig_sweep_kernel!`; the loss-width families and the scale marginals are unchanged. Every result is bit for bit the
  same (this package's tests; 1026 scoring results and a PT-E pool selection sample of neurips2026, before and after).
- **Fixed: a scored hypothesis allocates nothing again** (the supplied-width sweep's one-work-item launch is a
  statically sized kernel instance on the CPU backend, its buffers on Core's workspace), so the allocation tests that
  failed since the one-kernel sweep (extreme_value, the scoring contract) succeed.
- **Changed: `WelschMarginal` and `SmoothTLSMarginal` share one keyword constructor** (`_centred_marginal(K; alpha0,
  prior_loss_width, outlier_halfwidth)`), a restatement Core's `check_duplication` found once the sweep left.

- **New: this package's scorers read from TOML on the local optimization's schema**: methods of
  VisualGeometryToolkitEstimation's `scoring_from_toml` (through `ALL_SCORING_TAGS`) and `config_dict` for the
  loss-width families at a `threshold` and `StudentTUniformScoring` at an `outlier_halfwidth`, each key required.
  (The Student-t scorer's scale and label priors are its constructor's defaults, Jeffreys and flat: not table keys.)
- **Changed: the prefix sweep is one KernelAbstractions kernel** (the owner's ruling on the twin audit's V2):
  `sweep!` launches `ig_sweep_kernel!` as one CPU work item over the hypothesis's sorted column instead of its own
  host scan; the kernel carries the per-point penalties, the prior table and the best k. Results bit for bit the
  scan's; 1.0–1.14× the time at n = 200–2500; `sweep!` now allocates the sorted column (8 B a point) and the
  launch's state. The exact selection kernels `exact_pm_score_kernel!`/`argmax_models_kernel!` are VisualGeometryToolkitCore's
  (V1).
- **New: `partition_score(prior, lg_half_dof, half_dof, RSS, outlier_cost)`** (public, not exported): the marginal
  score of one inlier/outlier partition, the one formula every sweep scores with (it was the private `_sweep_score`).
  Plain arithmetic in `RSS`'s precision, callable inside a KernelAbstractions kernel, so a downstream device sweep
  calls it instead of a copy (BlobBoards' per-blob sweep restated it as its `_sweep_score`). The size-indexed form
  the sweeps call is `_table_partition_score` (was `_partition_score`).

- **SIMFIT++ is VisualGeometryToolkitEstimation's.** This package's copy
  (`src/simfitpp.jl`) is deleted, and its names are no longer exported or public
  here: `SimFitPP`, `simfitpp_loss_width` and `simfitpp_select` are exported by
  VisualGeometryToolkitEstimation, `unbiased_median_variance` and
  `with_residual_unit` are public there. The same algorithm, the same defaults
  and the same selections (Estimation scores the train split by Core's threshold
  `score!`, the same per-point quality summed in the same order). The grid family
  stays here: `select_on_grid!` over a `SimFitPP`, its `width_axis`
  (`LossWidth`) and its tag `"sim_fit_pp"` take Estimation's `SimFitPP`.
  Migration: `using RANSACScoringDoneRight: SimFitPP, simfitpp_select` →
  `using VisualGeometryToolkitEstimation: SimFitPP, simfitpp_select` (likewise
  `simfitpp_loss_width`; `VgtkEstimation.unbiased_median_variance`,
  `VgtkEstimation.with_residual_unit`).
- **`SCORING_TAGS` is renamed `ALL_SCORING_TAGS`.** The registry is unchanged
  (VisualGeometryToolkitCore's `SCORING_TAGS` merged with this package's
  families); Core owns the name `SCORING_TAGS`. Migration:
  `RANSACScoringDoneRight.SCORING_TAGS` → `RANSACScoringDoneRight.ALL_SCORING_TAGS`.

### Removed duplicates

- Device uploads go through VisualGeometryToolkitCore's `ka_upload` (the private
  `_ka_upload` is deleted).
- The docstring gates are VisualGeometryToolkitCore's `test_docs` (the copied
  `test/docs_gates.jl` is deleted); the suite runs Core's `test_duplication`.
- neurips2026 writes its prose-number macros through VisualGeometryToolkitEval's
  `PaperNumbers`/`add_number!`/`write_numbers` (`prose_macros` is replaced by
  `prose_numbers`).
- Distributions and Random are no longer dependencies (only SIMFIT++ used them).
- neurips2026's figure fragments print every coordinate through VisualGeometryToolkitEval Render's `pgf_coords`
  (at `"(%.17g,%.17g)"`); panels.jl's own coordinate writers are deleted. **The committed fragments' BYTES change**
  (the nine Profile and Sensitivity fragments of paper/figures, re-rendered from the stored inputs of record; no
  experiment rerun): each number is spelled with 17 significant digits (`-1` for `-1.0`,
  `-0.98994974874371877` for `-0.9899497487437188`) and the profile lines lose their trailing space. Every number is
  unchanged: Eval's `compare_fragments` reports the same numbers, max |Δ| 0.0, in every fragment. A point with a
  non-finite y is now DROPPED (`pgf_coords`'s rule); points lost per figure: PT-E-SIFT-Profile 0 of 4223,
  ETH3D-F-SPLG-Profile 0 of 4223, HEB-H-SIFT-Profile 0 of 4223, the PT-E and ETH3D-F Sensitivity fragments 0 of 210
  each, the HEB-H Sensitivity fragments 0 of 231 each. The style sheets are byte-identical (their palette through
  Eval's `color_defs`).
- neurips2026's sample-budget study reads the CV gate, the median and the mAA from VisualGeometryToolkitEval
  (`enforce_probe_interior`, `finite_median`, `eligible_maa`; the median now drops an `Inf` error too), and the radix
  probe's Float32 key and UInt32 sort are VisualGeometryToolkitCore's (`radix_key`, `radix_sortperm!`).
- neurips2026 no longer depends on Statistics or SpecialFunctions (nothing loads them).
- neurips2026's `scripts/fetch_data.jl` is the one call `fetch_data!(project_catalog())` (VisualGeometryToolkitEval
  DataArtifacts'; the script's own command line, fetch loop and report are deleted), and the radix probe's Int32-
  permutation variant runs on VisualGeometryToolkitCore's `RadixPermBuffers{UInt32, Int32}` (its own sort deleted).
- [compat]: neurips2026 VisualGeometryToolkitEstimation 0.9, VisualGeometryToolkitEval 0.7 (temporary [sources] to the
  pin vgtk-pin-9c74e264).

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
