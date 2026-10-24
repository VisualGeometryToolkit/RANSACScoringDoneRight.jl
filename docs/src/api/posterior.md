# Posterior weighting

```@meta
CurrentModule = RANSACScoringDoneRight
```

After marginalizing the scale, the inlier density is a Student-t; the selected
partition's scale estimate and the per-point posterior inlier probabilities
follow in closed form. That toolkit — the scale estimators (`IGPrior`,
`fit_scale`, `partition_student_t`), the Student-t + uniform mixture
(`StudentTMixture`, `posterior_weights!`, `posterior_inlier_probs!`), the
posterior-weight state and its expectation–conditional-maximization (ECM)
updates (`PosteriorWeightState`, `update_scale!`, `update_nu!`,
`update_inlier_prior!`, `mixture_ecm!`) and their update policies
(`Estimated`, `FixedS2`, `FixedPrior`, `FixedNu`, `ConsensusDof`) — is
VisualGeometryToolkitCore's, documented there.

A scored partition seeds it through the scoring contract: after `score!` on a
[`StudentTUniformScoring`](@ref), the first `inlier_count` entries of
`consensus_order` are the swept partition, and
`PosteriorWeightState(sq_residuals, n, model_dof, k; order, prior = scale_prior(scoring), …)`
is its Student-t with the E-step run once.
