# Scoring

```@meta
CurrentModule = RansacScoringDoneRight
```

The scale-free marginal score and its sort-and-sweep, on VisualGeometryCore's
scoring contract, and the loss-width ("threshold-based") scorings it is
compared against.

## Problems and the scoring contract

A problem is scored through VisualGeometryCore's constraint-problem protocol:
it subtypes `AbstractConstraintProblem` and implements `data_size`,
`codimension`, `minimal_sample_size` and `constraint_with_jacobians`;
predictive scoring additionally needs `solver_covariance`, declared by
`provides_solver_covariance`. A scorer implements VisualGeometryCore's scoring
contract (`scoring_workspace`, `score!`, `inlier_count`, `consensus_order`, …),
and the scale prior `IGPrior` (Jeffreys: `JeffreysPrior()`), the Student-t +
uniform mixture toolkit and the Gaussian-uniform boundary (`GauInlier`,
`decision_threshold`) are VisualGeometryCore's too. The concrete problems
(homography, fundamental and essential matrix, …) live in RobustVisualGeometry.

## The scale-free marginal score

[`StudentTUniformScoring`](@ref) carries the configuration (outlier half-width,
model degrees of freedom, scale prior, the profile-prior flag);
[`ScoringWorkspace`](@ref) carries the scratch buffers of the sort-and-sweep —
the scoring contract's `scoring_workspace`. `StudentTUniformScoring{Predictive}`
scores a model with its solver covariance.

```@docs
StudentTUniformScoring
Predictive
ScoringWorkspace
score_residuals!
score_sq_residuals!
```

## Loss-width scorings

The baselines of the paper (RANSAC, MSAC, GaU/MLESAC, MAGSAC++) and three
robust costs, each a singleton type carrying its codimension and scored through
one per-residual scalar, VisualGeometryCore's `residual_quality`. RANSAC and
MSAC are VisualGeometryCore's baseline scorers (`RansacScoring`, `MsacScoring`);
this package adds their grid methods.

```@docs
LossWidthScoring
MlesacScoring
MagsacScoring
WelschCostScoring
SmoothTLSScoring
GemanScoring
loss_width_scoring
```
