# Conventions

The stack's shared conventions — naming, signatures, data arguments, frames and
normalization, pixel origins, derivatives — are one canonical page in
VisualGeometryToolkitCore's documentation,
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/).
This page does not restate them. It states the three things that are true of
RansacScoringDoneRight and of nowhere else in the stack.

## The pool frame is the one declared exception to "frame in = frame out"

Everywhere else the rule is pixels in, pixels out, with normalization private to
one estimator. The benchmark pool is the exception, and it is declared rather
than silent. The pool record — VisualGeometryToolkitCore's `PoolPair` (its `Pools`
module documents the record, its maps and its accessors) — stores its
correspondences and models in **`NormalizedImageFrame`**: image centre, scaled
by HALF the image diagonal, so the image corners lie at unit radius. The
record's residual matrix is not part of it: a pool's residuals are a separate
`num_points × num_models` Float32 matrix in the same unit, and every pool
scorer here takes the record AND its matrix — `best_pool_model(scoring, pair,
R, …)`, `pool_model_score`, `precompute_pair(pair, R)`,
`stage_ig_shared_from_pairs(pairs, residuals)`. No scorer opens a file.

The exception exists because the pool is shared STORAGE, not an estimator's
input: a pool is written once and scored many times, across scenes whose images
differ in size. Storing one frame keeps a single global outlier half-width and
one IG scale prior correctly specified across the whole benchmark; converting
per pair at scoring time would not.

What the rule then demands of this package:

* **Every width on pool data is in the pool frame's unit and says so**: the
  outlier half-width, the loss width, the IG scale prior, every
  [`grid_evaluate`](@ref) grid and `POOL_FRAME_R_MAX`.
* **There is one exit to pixels.** [`pixel_residuals`](@ref) is the image-plane
  Sampson residual of the pool's models in pixels, computed from the light
  record alone: the correspondences come back through the stored maps
  (VisualGeometryToolkitCore's `pixel_csponds`), each model by composition with the
  normalization's sides (`inv(D) ∘ M ∘ S`). It never reads the stored residual
  matrix, so it does not depend on that matrix's residual kind — tangent
  Sampson for an essential pool, image-plane Sampson for F and H.
  [`residual_unit_scale`](@ref) is a CONVENTION, not a conversion — the two
  images are scaled independently, so no single scalar relates the frames; use
  it for a soft initial guess and never for anything a bound gates.

[`simfitpp_select`](@ref) on a pool record is the deliberate exception to the
exception: it is pixels in, pixels out, running on `pixel_residuals` so that
SIMFIT++'s published pixel bounds (0.25–8 px) mean what they say.

## `POOL_FRAME_R_MAX = 1` is a declared constant, not a tuning knob

`POOL_FRAME_R_MAX` is the upper limit of the domain `[0, r_max]` over which the
density normalizer `Z̃` of every redescending marginal (Welsch, smooth-TLS,
Geman-McClure) is integrated. It is in the pool frame's half-diagonal unit.

It reads like a geometric bound, and it was first justified as one — under the
FULL-diagonal frame every point lies within radius ½ of the centre, so no
residual exceeds 1. Pools are written in the HALF-diagonal frame, where a
point-to-point residual reaches 2, so that justification no longer holds and the
constant truncates the outlier support at half the geometric maximum.

It stays 1 deliberately. Every constant tuned for these marginals — the
cross-validated prior centres and outlier half-widths, and the published tables
built on them — was cross-validated with `r_max = 1`, so the value is part of
their definition, and one value keeps the truncation identical across scenes,
probe and evaluation. Changing it moves every marginal score and invalidates
every constant tuned against it.

## `StudentTUniformScoring{Predictive}` is an option, not a default

`StudentTUniformScoring{Predictive}` incorporates the model's estimation
covariance `Σ̃_θ` into the per-point scores. It is kept, and it is not the
default, because it was measured rather than assumed.

On the reference rerun — 320 tasks, thresholds tuned per scoring AND per stage,
paired seeds — against model-certain scoring, pooled ΔAUC@10°:

| Problem | At selection | After local optimization |
|---|---|---|
| Homography | −0.003 (tie) | — |
| Absolute pose | −0.001 (tie) | — |
| Fundamental | +0.011 | −0.029 |
| Essential | −0.010 | +0.007 |

Its one clear win is the essential matrix after local optimization at N = 60
(+0.016). It costs 13–82× per hypothesis at N = 60 and 68–565× at N = 250, even
with the kinetic enumeration. The loss is largest where the predicted variance
is most heterogeneous across inliers.

Two things that verdict does NOT say. The search is EXACT — the enumeration
visits every top-k set attainable at any `μ > 0` — so this is a property of the
FIRST-ORDER predictive model, not of the implementation. And whether
linearization is the cause is open: an oracle-covariance experiment settles it.
The option stays for that reason.

## See also

* [Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/)
  — the shared text this page links to rather than restates
* [API Reference](api/pools.md) — the functions named above
