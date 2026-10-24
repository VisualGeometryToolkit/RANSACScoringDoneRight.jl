# Overview

```@meta
CurrentModule = RANSACScoringDoneRight
```

`RANSACScoringDoneRight` is the scoring package of the paper *RANSAC Scoring
Done Right*: the scale-free marginal score and its sort-and-sweep, Student-t
posterior weighting, the threshold-based baselines it is compared against, and
CPU/GPU grid evaluators that score a stored hypothesis pool at every value of a
hyperparameter grid. It is problem-agnostic: a consumer supplies residuals, or a
problem implementing VisualGeometryToolkitCore's `AbstractConstraintProblem`.

It sits in a stack of Julia packages:

| package | role |
|:---|:---|
| VisualGeometryToolkitCore | geometric types, `Uncertain`/`whiten`, the robust-loss shapes, the contracts (problem protocol, scoring, local optimization, pool selection), the Student-t posterior toolkit, the MSAC/RANSAC baseline scorers, and the benchmark pool record `PoolPair` |
| **RANSACScoringDoneRight** | scoring: the marginal score and the paper baselines on the scoring contract, their grid evaluators on the pool-selection contract; depends only on VisualGeometryToolkitCore |
| VisualGeometryToolkitEstimation | the RANSAC loop, the concrete problems (homography, fundamental and essential matrix, …) and the core local optimizations |
| LocalOptimizationDoneRight | the state-of-the-art local-optimization strategies |
| VisualGeometryToolkitEval | pool IO, statistics, tuning (including the scale-prior fits), the evaluation engine, rendering |

A caller composes them: VisualGeometryToolkitEstimation's `find_*` take a scorer from
this package and a strategy from LocalOptimizationDoneRight as
`scoring = …, local_optimization = …`.

## Scoring versus selection

A RANSAC *score* assigns a value to one model hypothesis from its residuals;
*selection* keeps the hypothesis with the best score. This package does both, at
two granularities:

- **one hypothesis**: [`score_residuals!`](@ref) (from residuals) or
  [`score_sq_residuals!`](@ref) (from a problem and a model) returns the score
  and the inlier count of the best partition;
- **a pool of hypotheses**: [`best_pool_model`](@ref) selects the best stored
  model of one benchmark pair, and [`grid_evaluate`](@ref) does so for every
  pair of a staged pool at every width of a grid, which is what tuning and the
  paper's evaluation consume.

The scale-free marginal score also *selects an inlier partition*: the sweep
returns the inlier count `k` it maximized over, and the scale of that partition
(VisualGeometryToolkitCore's `consensus_s2`) follows in closed form. That partition and scale
seed local optimization through the scoring contract (`consensus_order`,
`inlier_count`, `inlier_threshold`) and VisualGeometryToolkitCore's posterior-weighting
layer (`PosteriorWeightState`).

## The contracts this package implements

Both contracts are VisualGeometryToolkitCore's; this package implements them for its
own types.

- **Scoring.** [`StudentTUniformScoring`](@ref) implements `scoring_workspace`
  (a [`ScoringWorkspace`](@ref)), `score!`, `inlier_count`,
  `consensus_order`, `inlier_threshold`, `scale_prior`, `outlier_halfwidth`,
  `check_workspace` and `with_width`/`width_axis`; its `{Predictive}` form adds
  `is_predictive` and `scored_model`. Any RANSAC written against the contract,
  VisualGeometryToolkitEstimation's among them, runs it unchanged.
- **Pool selection.** Every grid family here declares one staging
  (`selection_staging`, `stage_selection`: [`stage_ig_shared_from_pairs`](@ref)
  shared by all of them) and implements `select_on_grid!` with its device
  kernels. VisualGeometryToolkitCore's generic `select_on_grid!` and
  [`best_pool_model`](@ref) serve any other scorer at its width.

## The scorers

**The scale-free marginal score.** For a fixed inlier partition the Gaussian
inlier scale is integrated out under a conjugate inverse-gamma prior, leaving a
closed form in the residual sum of squares, the inlier count and the outlier
half-width `a` of the uniform outlier component. Sorting the residuals once lets
one sweep score every partition, so a hypothesis costs O(N log N).
[`StudentTUniformScoring`](@ref) is the spec (its scale prior a keyword,
Jeffreys by default), [`ScoringWorkspace`](@ref) the sweep buffers.
`StudentTUniformScoring{Predictive}` adds the model's
estimation covariance to every point's residual covariance; it is an option, not
the default (see [Conventions](conventions.md)).

**The threshold-based family.** RANSAC, MSAC, MLESAC (the paper's GaU) and
MAGSAC++ are singleton types (VisualGeometryToolkitCore's `RansacScoring` and `MsacScoring`,
[`MlesacScoring`](@ref), [`MagsacScoring`](@ref)), together with three robust
costs (Welsch, smooth TLS, Geman–McClure). They form the
[`LossWidthScoring`](@ref) union: each takes a loss width, and each scores
through one per-residual scalar, [`residual_quality`](@ref). Every scoring
carries its constraint codimension `DG`, because a loss width means a different
multiple of the scale at each codimension.

**The mixture's decision boundary.** [`GauInlier`](@ref) is the
Gaussian-uniform mixture; [`decision_threshold`](@ref) and its inverse
`decision_width` are the one map between an outlier half-width and the residual
at which the MAP inlier/outlier decision flips, so a grid of outlier half-widths
is never confused with a grid of thresholds.

## Scale priors

VisualGeometryToolkitCore's `IGPrior(α₀, β₀)` is the inverse-gamma prior on σ²; `IGPrior()` (also
spelled `JeffreysPrior()`) is its `α₀ = β₀ = 0` limit, the non-informative
Jeffreys prior, which needs no validation data. In the conjugate update the
prior acts as `2α₀` pseudo-residuals carrying `2β₀` of pseudo-RSS, so where a
pair has many inliers the data dominate and the informative priors approach the
Jeffreys limit.

The paper's empirical-Bayes variants fit the prior from per-pair noise-variance
estimates on a validation set (VisualGeometryToolkitEval's `Tuning.PriorSpec` and
`Tuning.fit_prior`):

| variant | prior | fit |
|:---|:---|:---|
| Jeffreys | `IGPrior(0, 0)` | none (`JeffreysPriorSpec`) |
| IG-MLE | `IG(α₀, β₀)` | type-II maximum likelihood: a Gamma fit to `1/σ̂²` |
| IG-Robust | `IG(1, median(σ̂²))` | fixed `α₀ = 1`, median-anchored `β₀` (`IGRobustSpec`) |

IG-Robust trades the dispersion information of IG-MLE for robustness to the
heavy σ̂² tails of pairs with few inliers, the regime where the prior matters.

## Widths and native units

Every scorer has one hyperparameter, a *width*, and each family's width lives on
its own axis: the outlier half-width for the marginal scores, the loss width for
the threshold-based family. VisualGeometryToolkitCore's `width_axis(scoring)` names a
scorer's axis (`OutlierHalfwidth` or `LossWidth`); its `to_native` maps a
scale multiple onto it and `native_map` is that map curried, so that
grids for different families can be laid out over comparable neighbourhoods of
the same noise scale. The vocabulary is fixed: `loss_width` is a loss's scale
parameter, `outlier_halfwidth` the half-width of the uniform outlier support,
`decision_threshold` the residual at which the MAP rule flips, `s2` a variance
and `ŝ`/`s_hat_sq` the scale estimate and its square.

## The pool frame

The benchmark pool — VisualGeometryToolkitCore's `PoolPair` record together with its
`num_points × num_models` Float32 residual matrix — stores correspondences,
models and residuals in **`NormalizedImageFrame`**: the image centre at the
origin, scaled by half the image diagonal so the corners lie at unit radius.
It is the one declared exception to the stack's rule that a public function
returns results in the frame its inputs came in. Every width applied to pool
data (outlier half-width, loss width, IG prior, every `grid_evaluate` grid) is
in that frame's unit, and [`pixel_residuals`](@ref) is the one exit to pixels.
[Conventions](conventions.md) states the rule and what it demands of this
package.

## Where the paper's experiments live

This repository holds the package and, in `neurips2026/`, the paper's
research project: the manuscript, the drivers that draw its Profile and
Sensitivity figures, their configs of record and the asset index
`neurips2026/REPRODUCE.md`, plus the scoring studies
([Experiments](guide/experiments.md)). The figures regenerate byte for byte
from the six stored evaluation runs of record; the evaluation that produced
those runs (pools, coarse probes, the per-pair grid evaluation and the
bootstrap) predates the current stack and is documented in REPRODUCE.md but
not yet ported. The bootstrap evaluation harness that once lived in this
package moved to VisualGeometryToolkitEval.
