# Pools, grids and tuning

```@meta
CurrentModule = RansacScoringDoneRight
```

Scoring over a benchmark pool: VisualGeometryCore's light `PoolPair` record
together with its `num_points × num_models` Float32 residual matrix, both in
the pool frame `NormalizedImageFrame` (see [Conventions](../conventions.md)).
No function here opens a file; pool IO belongs to VisualGeometryEval.

## Per-pair model selection

```@docs
best_pool_model
pool_model_score
precompute_pair
```

## Grid selection: the pool-selection contract

VisualGeometryCore's `select_on_grid!(idxs, P, scoring, grid, staged)`
fills `idxs[p, t]` with the model selected for pair `p` at width `grid[t]`. Every
grid family of this package (`StudentTUniformScoring`, the width-free
loss-width families, [`LossSpec`](@ref), `ScaleMarginal`, [`SimFitPP`](@ref))
declares the staging `:rsdr`: `stage_selection(selection_staging(scoring), pairs, residuals; backend)`
is one [`stage_ig_shared_from_pairs`](@ref) shared by all of them, and its
`select_on_grid!` is one [`grid_evaluate`](@ref) (SIMFIT++: its host loop over
[`simfitpp_select`](@ref)). [`pair_chunk_budget`](@ref) sizes the device chunks.

```@docs
pair_chunk_budget
```

## Grid evaluation (CPU and GPU)

A pool is staged once ([`stage_ig_shared_from_pairs`](@ref)) and every scorer is
evaluated over a grid of its native width by one verb, [`grid_evaluate`](@ref),
which dispatches on the scoring. The kernels are written with KernelAbstractions
and run on `CPU()` or a CUDA backend.

```@docs
IGSharedData
stage_ig_shared_from_pairs
grid_evaluate
Aggregation
Exact
LogBinned
```

## Native widths

Each scorer's hyperparameter lives on its own width axis: the outlier half-width
for the marginal scores, the loss width for the threshold-based family.
VisualGeometryCore's `width_axis(scoring)` names it (this package states it for
its grid families); the scale multiple → native maps `to_native` and
`native_map`, and the conversions `outlier_halfwidth_to_loss_width` /
`loss_width_to_outlier_halfwidth`, are VisualGeometryCore's.

## Scale marginals of redescending losses

```@docs
WelschMarginal
LossSpec
loss_spec
loss_scale_bounds
```

## Scale priors

The empirical-Bayes variants of the paper fit VisualGeometryCore's `IGPrior`
from per-pair noise-variance estimates on a validation set: IG-MLE by type-II
maximum likelihood, IG-Robust with `α₀ = 1` and a median-anchored `β₀`. The
fits (`PriorSpec`, `fit_prior`, `ALL_PRIOR_SPECS`) are VisualGeometryEval's
`Tuning`; a fitted prior enters a scorer as `StudentTUniformScoring(P, a; prior)`.

## SIMFIT++ loss-width estimation

A step-for-step port of Edstedt's SIMFIT++ ([Parskatt/simfitpp](https://github.com/Parskatt/simfitpp)),
with a pool adapter.

```@docs
SimFitPP
simfitpp_loss_width
simfitpp_select
unbiased_median_variance
pixel_residuals
residual_unit_scale
with_residual_unit
```
