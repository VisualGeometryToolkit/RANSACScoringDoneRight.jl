# Scoring

```@meta
CurrentModule = RANSACScoringDoneRight
```

The scale-free marginal score and its sort-and-sweep, on VisualGeometryToolkitCore's
scoring contract, and the loss-width ("threshold-based") scorings it is
compared against.

## Problems and the scoring contract

A problem is scored through VisualGeometryToolkitCore's constraint-problem protocol:
it subtypes `AbstractConstraintProblem` and implements `data_size`,
`codimension`, `minimal_sample_size` and `constraint_with_jacobians`;
predictive scoring additionally needs `solver_covariance`, declared by
`provides_solver_covariance`. A scorer implements VisualGeometryToolkitCore's scoring
contract (`scoring_workspace`, `score!`, `inlier_count`, `consensus_order`, …),
and the scale prior `IGPrior` (Jeffreys: `JeffreysPrior()`), the Student-t +
uniform mixture toolkit and the Gaussian-uniform boundary (`GauInlier`,
`decision_threshold`) are VisualGeometryToolkitCore's too. The concrete problems
(homography, fundamental and essential matrix, …) live in VisualGeometryToolkitEstimation.

## The scale-free marginal score

The Student-t + uniform marginal score itself is VisualGeometryToolkitCore's
(moved there on 2026-11-11):
[`StudentTUniformScoring`](@extref VisualGeometryToolkitCore.StudentTUniformScoring)
carries the configuration (outlier model, model degrees of freedom, the scale
and label priors, both required: the stack's defaults are
VisualGeometryToolkitEstimation's `[scoring.student_t_uniform]`, read by
`scoring_from_toml`), [`ScoringWorkspace`](@extref VisualGeometryToolkitCore.ScoringWorkspace)
the scratch buffers of the sort-and-sweep, and
`StudentTUniformScoring{Predictive}` scores a model with its solver covariance.
This package extends it by dispatch: its grids (`grid_evaluate`,
`select_on_grid!`) and the outlier models below.

## Outlier models that estimate the support

Core's estimated-support sweep scores these through their methods of its
hooks (`outlier_cost`, `scored_outlier_halfwidth`, `outlier_context!`);
model-certain, codimension 1.

```@docs
TailRules
ExtremeValueHalfwidth
GeneralizedParetoOutliers
OutlierTerm
tail_fit
gpd_pwm_fit
gpd_support
gpd_outlier_cost
```

## Loss-width scorings

The baselines of the paper (RANSAC, MSAC, GaU/MLESAC, MAGSAC++) and three
robust costs, each a type carrying its codimension and scored through one
per-residual scalar, VisualGeometryToolkitCore's `residual_quality`. Each has
VisualGeometryToolkitCore's two forms of `RansacScoring`/`MsacScoring`: width-free
(`F(dg)`, the grid family) and at a loss width (`F(dg; threshold = w)`, or
`with_width(F(dg), LossWidth(w))`), which selects by `score!`. RANSAC and MSAC
are VisualGeometryToolkitCore's baseline scorers; this package adds their grid
methods.

```@docs
LossWidthScoring
MlesacScoring
MagsacScoring
WelschCostScoring
SmoothTLSScoring
GemanScoring
loss_width_scoring
ALL_SCORING_TAGS
```
