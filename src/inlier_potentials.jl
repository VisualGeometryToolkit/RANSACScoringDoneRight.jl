# =============================================================================
# inlier_potentials.jl — the inlier-loss layer: canonical depths and the Z̃
# normalizer integrand, on VisualGeometryToolkitCore's loss shapes.
#
# THE SHAPE MATH IS VisualGeometryToolkitCore's, and only VGC's: every path in this
# package — the scale-quadrature data term (`_accumulate_data!`, shared by the
# Track-B mixture and the full-data marginals), the Z̃ quadrature below, the
# loss-width sweep bin weights — calls `VisualGeometryToolkitCore.ρ` (and `weight`, its
# s-derivative, where needed) directly. The depth-parameterised potential this
# subsystem works in is VGC's UNIT-WIDTH shape, scaled by the well depth:
#
#     ρ̃(sq_r; sq_loss_width, Δ) = Δ · ρ(shape, sq_r / sq_loss_width, 1)
#
# (VGC's ρ(shape, s, sq_loss_width) is sq_loss_width·g(s/sq_loss_width), so
# this is Δ·g(sq_r/sq_loss_width).) The restated copies `_rho` (host) and
# `_mix_rho_ka` (device) are gone: kept for bit-exactness with themselves, they
# were a second statement of math VGC owns. Against them the VGC spelling agrees
# bit for bit for Welsch and smooth-TLS and to the last ulp or so for
# Geman-McClure, whose copy rounded Δ·s/(w² + s) where VGC rounds w²·s/(w² + s)
# (w the loss width) — accepted (2026-09-21).
#
# What this file DOES own is the canonical depths and the Z̃ integrand: it is
# included FIRST (before the scale marginals and Track B), so both subsystems —
# the full-data scale marginals (welsch_marginal.jl / smoothtls_marginal.jl)
# and the mixture reformulation (inlier_losses.jl) — dispatch on one set of
# shape tags and one depth per shape.
# =============================================================================

# -----------------------------------------------------------------------------
# The loss shapes
# -----------------------------------------------------------------------------

# The shape TAGS are VisualGeometryToolkitCore's loss family, used here by their own
# names — `AbstractLossShape`, `GaussianShape`, `WelschShape`,
# `GemanMcclureShape`, `SmoothTLSShape`. The formula layer below stays
# RSDR-owned (methods on the VGC tags — RSDR owns the functions, no piracy).
# Every method dispatches on the bare, fieldless tag and takes the squared loss
# width and the depth explicitly; VGC's deployed loss
# (`deploy(shape, s2; scale_multiple)`) composes a width on instead, and this
# facet does not use it.

# Holy trait — the canonical (fixed) well depth of each shape: the Δ of the
# classical scaled potential (½ for Welsch and Geman-McClure, 1 for smooth-TLS).
# The fixed-depth users — the full-data scale marginals
# (`loss_spec(::ScaleMarginal)`) — read their Δ here.
@inline _default_depth(::WelschShape)       = 0.5
@inline _default_depth(::GemanMcclureShape) = 0.5
@inline _default_depth(::SmoothTLSShape)    = 1.0

# THE NAMES MATCH VisualGeometryToolkitCore: `sq_r` is its `s` (a squared RESIDUAL),
# `sq_loss_width` its squared loss width. There is no scale multiple in this
# facet: these losses MARGINALISE the loss width rather than deploy a shape at
# `scale_multiple·σ`, so `sq_loss_width` is the whole width, not
# `scale_multiple^2 * s2` decomposed — which is why `deploy` and `retune` do not
# appear here.

# -----------------------------------------------------------------------------
# The WELL DEPTH Δ = ρmax (the density floor is e^{−Δ}) is a dimensionless
# log-contrast, SWEPT on VAL; Δ→∞ removes the floor ⇒ Gaussian ⇒ Student-t (the
# sweep anchor). The fixed-depth forms use the shape's trait depth above.
# -----------------------------------------------------------------------------

# (`_influence` — the IRLS weight 2·ρ̃′ in both a depth-scaled and a fixed-depth
#  form — is deleted. Its only caller was the deleted posterior-weights helper of
#  the retired fixed-depth layer, and the two forms disagreed: the depth-scaled
#  one was ρ̃′, not 2ρ̃′, for Welsch and Geman. The IRLS weight of a VGC shape is
#  VGC's `weight`.)

# log Z̃(sq_loss_width; Δ) — the loss's normaliser on [−r_max, r_max] — is
# VisualGeometryToolkitCore's `truncated_log_normaliser`.
