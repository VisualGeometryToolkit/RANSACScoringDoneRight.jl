# =============================================================================
# smoothtls_marginal.jl — marginal smooth-TLS GLOBAL scoring (the scale-marginal
# analogue of the smooth-TLS score): `ScaleMarginal{SmoothTLSShape}`, scored
# exactly as the marginal Welsch — welsch_marginal.jl's header states the
# integrand, the prior and the swept CV parameter (`prior_loss_width`, the
# IG-prior centre on the smooth-TLS loss width), inlier_losses.jl's the
# smooth-TLS potential and why its flat floor (e^{-1} beyond the cutoff) needs
# the Z̃ truncation to the outlier support, the declared `POOL_FRAME_R_MAX`.
# =============================================================================

# `SmoothTLSMarginal` is the `ScaleMarginal{SmoothTLSShape}` alias
# (welsch_marginal.jl); only the keyword constructor is family-specific.
"""
    SmoothTLSMarginal(; alpha0=1.0, prior_loss_width, outlier_halfwidth)

Marginal smooth-TLS scoring spec.  `prior_loss_width` is the IG-prior centre on
the smooth-TLS loss width (`beta0 = alpha0·prior_loss_width²` ⇒ `2·alpha0`
pseudo-observations at pseudo-squared-width `prior_loss_width²`);
`outlier_halfwidth` bounds the density normalizer Z̃ (set to
`POOL_FRAME_R_MAX`, the declared outlier support in the pool frame's unit —
required, no default).
`alpha0`/`beta0` match `IGPrior`, as for `WelschMarginal`.  The evidence is the
trapezoid scale quadrature of [`loss_spec`](@ref)`(::ScaleMarginal)`.
"""
SmoothTLSMarginal(; alpha0::Real = 1.0, prior_loss_width::Real, outlier_halfwidth::Real) =
    _centred_marginal(SmoothTLSShape, alpha0, prior_loss_width, outlier_halfwidth)

# The evidence of this spec is evaluated by the ONE device kernel
# (`grid_evaluate(::ScaleMarginal, …)`, marg_sweep_ka.jl), on CPU() or a GPU backend.
