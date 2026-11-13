# RANSACScoringDoneRight changelog

## 0.8.0 (2026-11-13, the VisualGeometryToolkit wave-10 stack)

On the wave-10 stack: VisualGeometryToolkitCore 0.10 and VisualGeometryToolkitEstimation 0.11 (the package and its
tests); neurips2026 also VisualGeometryToolkitEval 0.8. No scoring change: the same results, bit for bit.

### Breaking

- **[compat]:** VisualGeometryToolkitCore 0.10, VisualGeometryToolkitEstimation 0.11; neurips2026 VisualGeometryToolkitEval 0.8.
  Floors raised to the gated resolve (`floor_compat --raise`): Statistics 1.11 (package, test), Documenter 1.20 and
  DocumenterInterLinks 1.1 (docs).
  Core 0.10's terminology (detection, correspondence, observation) reaches this package only through the pool record:
  `PoolPair`'s correspondences are `PoolPair(; correspondences, …)` and the way back to pixels is Core's
  `pixel_correspondences` (both renamed in Core; `pixel_residuals`, this package's one exit to pixels, is unchanged).
- **neurips2026: the ETH3D SuperPoint+LightGlue artifact is bound as `eth3d_splg_correspondences`** (`Artifacts.toml`,
  `configs/data_artifacts.toml`). The published Zenodo record is unchanged (record 23240469, the same tarball, sha256 and
  tree): its file name of record is the entry's `published_name` (VisualGeometryToolkitEval DataArtifacts), and the
  published tree's directory name stays as published (`configs/eth_f_pools.toml` `input_jld2_dir`).
- **neurips2026: the pool configs of record spell `min_correspondences`** (`eth_f_pools`, `heb_h_pools`, `pt_e_pools`),
  VisualGeometryToolkitEval's keyword for the dropped-pair floor.
- **neurips2026 no longer depends on VisualGeometryToolkitIO** directly (nothing in it loads it; VisualGeometryToolkitEval
  does).

### Other

- The whole stack is in General: the install lines (README, docs, REPRODUCE.md) and the CI, docs and Downgrade
  workflows add no other registry.
- Prose follows the terminology ruling: a correspondence where a pair's datum is meant, "benchmark" for a timing.
- CI: the Downgrade workflow is deleted (the stack's one policy: Resolver.jl does not run on Julia 1.13, and every
  release raises its compat floors to the versions its gate resolved, `floor_compat --raise`).
- Comments that pointed at files and types the 0.7.0 port deleted (`sweep.jl`, `GauScoring`, `decision_width` in
  scoring.jl) name VisualGeometryToolkitCore's owners.
- No restated code (VisualGeometryToolkitCore's check_duplication against the stack): the five loss-width scorer
  structs drop their inner constructor (it was Julia's default constructor, written out); the tests build the pool
  maps with Core's `to_homogeneous` and their constraint tuples inline.

## 0.7.2 (2026-11-13)

A patch: requires Julia 1.13 (`julia = "1.13"` in the `[compat]` of the package and of every
workspace member: test, docs and neurips2026). CI runs Julia 1.13 only. No API change.

- CI: the Downgrade workflow (the suite at the compat floors) runs on demand only: Resolver.jl, which it uses,
  does not run on Julia 1.13 yet.

## 0.7.1 (2026-11-13)

A patch: no API change, the same results. On VisualGeometryToolkitEstimation 0.10 (`[compat]`
VisualGeometryToolkitEstimation "0.10" in the package, test and neurips2026 environments). Estimation 0.10.0 retired the image-space refine of
a free SplineLens camera's controls (`refine_camera_and_poses` on a SplineLens camera now takes `fixed = (:controls,)`
only); RANSACScoringDoneRight never called that route, so nothing here changes.

## 0.7.0 (2026-11-12, the VisualGeometryToolkit 0.9 stack)

On the VisualGeometryToolkit 0.9 stack: VisualGeometryToolkitCore 0.9, and
VisualGeometryToolkitEstimation (a new dependency, for SIMFIT++ alone);
neurips2026 also Eval and IO of that stack.

### Breaking

- **On the stack's one copy of five more things** (the duplication audit's hoists; monorepo pin 9c74e264):
  the five loss-width families' constructors and width setter are VisualGeometryToolkitCore's
  `@threshold_scoring_constructors` (they gain the problem-TYPE form `F(P; threshold)`, as RANSAC/MSAC have it);
  their TOML reader and writer are VisualGeometryToolkitEstimation's `AbstractThresholdScoring` methods, so
  `config_dict` of one of them takes the registry, `config_dict(sc; tags = ALL_SCORING_TAGS)`, and a table that
  spells `outlier_halfwidth` is refused naming the scorer's own `threshold`; the eager release of a grid's device
  buffers is Core's `free_device_buffers!`, and GPUArrays is no longer a dependency. Results bit for bit.
- **[compat]:** VisualGeometryToolkitEstimation 0.9 (test too); neurips2026 Estimation 0.9 and Eval 0.7.
- **Removed: the `Exact` aggregation and the `Aggregation` supertype** (the duplication audit, the coordinator's
  ruling 2026-11-12). They were this package's second launcher of VisualGeometryToolkitCore's
  `exact_pm_score_kernel!`/`argmax_models_kernel!`: the exact selection of a loss-width family over a width grid is
  Core's `select_on_grid!(idxs, P, scoring, grid, stage_selection(:pool, pairs, residuals))`. `grid_evaluate` of a
  loss-width family takes `aggregation::LogBinned` only (`LogBinned()` the default, unchanged). The names are gone,
  so `Exact()` and `Aggregation` are `UndefVarError`s naming them. The tests' exact reference is Core's selection;
  the LogBinned grid is unchanged, bit for bit. `select_on_grid!`'s methods here take this package's staging only
  (`(; staged, pairs)`), so a grid family handed Core's `:pool` staging reaches Core's generic method.
- **Device launch policy: VisualGeometryToolkitCore's one launch policy** (the owner's ruling 2026-11-12): every grid
  kernel is instantiated through `OnBackend(backend; workgroupsize = 1)(kernel)`, one work item per workgroup on the
  CPU and one warp (32 items) per workgroup on a device; this package's `_ka_kernel` (one item on the CPU, the
  backend's choice on a device) is deleted. Results bit for bit (every selection, CPU and CUDA, before and after).
  GPU timing on one B200 (128 pairs × 800 points × 48 models, minimum of 7; the last release's `_ka_kernel` /
  one item per workgroup / now): the Student-t grid 1.54 / 29.9 / 1.31 ms, MSAC LogBinned 10.7 / 19.6 / 6.4 ms,
  MAGSAC++ LogBinned 18.2 / 26.9 / 13.8 ms, the full-data Welsch marginal 85.2 / 67.9 / 28.4 ms, the Track-B mixture
  marginal 90.8 / 62.2 / 17.5 ms. CPU unchanged (0.93–0.98 of before, within noise).
- **Fixed: `grid_evaluate(::StudentTUniformScoring)` on a GPU backend** threw (the device kernel was handed a host `Array`
  as its empty best-k output); it is uploaded to the staging's backend now. With the GPU-first staging default,
  the Student-t grid did not run on a CUDA machine at all.
- **GPU-first defaults:** `stage_ig_shared_from_pairs` and the pool-selection staging (`stage_selection`) default to
  VisualGeometryToolkitCore's `default_device()` (the accelerator when one is loaded and functional), no longer a pinned
  `KernelAbstractions.CPU()`. A `backend` supplied explicitly is unchanged.
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
- [compat]: VisualGeometryToolkitCore 0.9 and VisualGeometryToolkitEstimation 0.9 (the package and its tests);
  neurips2026 also VisualGeometryToolkitEval 0.7 and VisualGeometryToolkitIO 0.9; every one resolved from General.

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
