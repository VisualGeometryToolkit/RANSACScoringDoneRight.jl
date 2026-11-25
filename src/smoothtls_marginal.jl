# =============================================================================
# smoothtls_marginal.jl — marginal smooth-TLS GLOBAL scoring (the scale-marginal
# analogue of the smooth-TLS score): `ScaleMarginal{SmoothTLSShape}`, scored
# exactly as the marginal Welsch — welsch_marginal.jl's header states the
# integrand, the prior and the swept CV parameter (`prior_loss_width`, the
# IG-prior centre on the smooth-TLS loss width), inlier_losses.jl's the
# smooth-TLS potential and why its flat floor (e^{-1} beyond the cutoff) needs
# the Z̃ truncation to the outlier support, the declared `POOL_FRAME_R_MAX`.
# =============================================================================

# The spec is `ScaleMarginal{SmoothTLSShape}`; its constructors are every
# scale-marginal family's (welsch_marginal.jl): nothing here is family-specific.

# The evidence of this spec is evaluated by the ONE device kernel
# (`grid_evaluate(::ScaleMarginal, …)`, marg_sweep_ka.jl), on CPU() or a GPU backend.
