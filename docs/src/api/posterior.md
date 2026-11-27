# Posterior weighting

```@meta
CurrentModule = RANSACScoringDoneRight
```

After marginalizing the scale, the inlier density is a Student-t; the selected
partition's scale estimate and the per-point posterior inlier probabilities
follow in closed form. That toolkit — the scale estimators (`IGPrior`,
`ScaleMAP` on `EstimationProblem(SquaredScale, …)`, `consensus_s2`), the Student-t + uniform mixture
(`StudentTMixture`, the scoring contract's `inlier_probabilities!`), the
posterior-weight state and its expectation–conditional-maximization (ECM)
updates (`PosteriorWeightState`, `square_distances!`, `update_weights!`, `update_scale!`, `update_nu!`,
`update_inlier_prior!`) and their update policies
(`Estimated`, `FixedS2`, `FixedPrior`, `FixedNu`, `ConsensusDof`) — is
VisualGeometryToolkitCore's, documented there.

A scored partition seeds it through the scoring contract: after `score!` on a
[`StudentTUniformScoring`](@extref VisualGeometryToolkitCore.StudentTUniformScoring), the first `inlier_count` entries of
`consensus_order` are the swept partition, and
`PosteriorWeightState(sq_distances, n, model_dof, k; order, prior = scale_prior(scoring), …)`
is its Student-t with the E-step run once.
