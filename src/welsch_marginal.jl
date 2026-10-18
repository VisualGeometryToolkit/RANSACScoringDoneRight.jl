# =============================================================================
# welsch_marginal.jl — marginal-Welsch GLOBAL scoring: the scale-marginal SPECS.
# The evidence itself is the device kernel's (`grid_evaluate(::ScaleMarginal, …)`,
# marg_sweep_ka.jl), by the package's one scale quadrature (inlier_losses.jl).
#
# The Student-t marginal (`StudentTUniformScoring`) marginalises a single inlier scale
# σ² out of a Gaussian inlier block under a Jeffreys prior (closed form, via the
# sort-and-sweep).  The marginal-Welsch score is the SOFT analogue: a single
# loss width is marginalised out of a redescending Welsch inlier density under a
# PROPER inverse-gamma prior (Jeffreys diverges for Welsch — its width
# saturates as a scale), every point counted as an inlier.
#
# The potentials, the density f(r; w) = e^{−ρ̃}/Z̃ and why Z̃ is truncated to
# [−h, h] (h the outlier half-width) are stated ONCE, in inlier_losses.jl's
# header. Over all n residuals, with ρ̃ at the shape's canonical depth and
# t = log sq_loss_width:
#   prior   = IG(α, β) on sq_loss_width   (β = α·prior_loss_width²)
#   ψ(t) = Σᵢ ρ̃(sq_rᵢ; sq_loss_width) + n·logZ̃(sq_loss_width) + α·t + β/sq_loss_width
#                                                              (neg-log integrand)
#   evidence  log Z = log ∫ exp(−ψ(t)) dt over t0 ± _MARG_L
#
# `sq_loss_width` — never a bare `s` — because VisualGeometryToolkitCore spells the
# squared RESIDUAL `s` in `ρ(shape, s, sq_loss_width)`; see the name note in
# inlier_potentials.jl.
#
# The swept CV parameter (what `grid_evaluate(::ScaleMarginal, …)` sweeps for this family, in
# its NATIVE units) is the IG-prior centre on the loss width,
# `prior_loss_width` — NOT a decision threshold; `α` and the outlier
# half-width (the Z̃ domain) are held fixed.  This file also hosts what every
# scale-marginal spec shares (the one parametric `ScaleMarginal`).
# =============================================================================

# ONE parametric spec for every scale-marginal family: the loss shape is the
# type parameter, so `WelschMarginal`/`SmoothTLSMarginal` are aliases, and the
# only per-family code is the keyword constructors. The shape math is
# VisualGeometryToolkitCore's.
# `alpha0`/`beta0`, exactly as `IGPrior` spells the SAME two IG hyperparameters.
# They were `α`/`β` here and `alpha0`/`beta0` there — one pair of quantities
# under two spellings, one struct apart, with the constructors already taking
# them as `α₀`/`β₀`.
struct ScaleMarginal{K<:AbstractLossShape}
    alpha0::Float64
    beta0::Float64
    outlier_halfwidth::Float64   # the Z̃ domain half-width
    t0::Float64                  # centre of the scale domain t0 ± _MARG_L, log(prior-centre²)
end
const WelschMarginal    = ScaleMarginal{WelschShape}
const SmoothTLSMarginal = ScaleMarginal{SmoothTLSShape}
"""
    ScaleMarginal{K}(alpha0, beta0; outlier_halfwidth, t0 = nothing)

The scale-marginal spec of loss shape `K` at the IG prior `(alpha0, beta0)` on
the squared loss width — e.g. one fit by type-II MLE (empirical Bayes), where
`beta0` is not tied to a prior centre. `outlier_halfwidth` bounds the density
normalizer Z̃ (REQUIRED: it is the declared outlier support, `POOL_FRAME_R_MAX`
in the pool frame's unit); `t0` the
centre of the log-scale domain, by default the prior mode
`log(beta0/(alpha0 + 1))` (or `log((outlier_halfwidth/4)²)` for an improper
prior). THE one constructor of every family's spec: the keyword forms below
only name the prior centre.
"""
function ScaleMarginal{K}(α₀::Real, β₀::Real; outlier_halfwidth::Real,
                          t0::Union{Nothing,Real} = nothing) where {K<:AbstractLossShape}
    # `t0_centre` is a LOG-scale (t = log sq_loss_width), so it is NOT named
    # after a squared width: both branches below are already inside a `log`.
    t0_centre = t0 !== nothing ? Float64(t0) :
              (α₀ > 0 && β₀ > 0 ? log(β₀ / (α₀ + 1)) : log((outlier_halfwidth / 4)^2))
    return ScaleMarginal{K}(Float64(α₀), Float64(β₀), Float64(outlier_halfwidth), t0_centre)
end

# The prior-centre form of every family: `2·alpha0` pseudo-observations at
# pseudo-squared-width `beta0/alpha0 = prior_loss_width²`, its scale domain
# centred at log(prior_loss_width²).
_centred_marginal(::Type{K}, alpha0::Real, prior_loss_width::Real, outlier_halfwidth::Real) where {K} =
    ScaleMarginal{K}(alpha0, alpha0 * prior_loss_width^2; outlier_halfwidth,
                     t0 = log(prior_loss_width^2))

"""
    WelschMarginal(; alpha0=1.0, prior_loss_width, outlier_halfwidth)

Marginal-Welsch scoring spec.  `prior_loss_width` is the IG-prior centre on the
Welsch loss width (`beta0 = alpha0·prior_loss_width²` ⇒ `2·alpha0`
pseudo-observations at pseudo-squared-width `prior_loss_width²`);
`outlier_halfwidth` bounds the density normalizer Z̃ — REQUIRED, as for every
scale-marginal spec.
`alpha0`/`beta0` are the same two IG hyperparameters `IGPrior` names. For an `(alpha0, beta0)` prior not tied to a
centre, [`ScaleMarginal{WelschShape}`](@ref ScaleMarginal)`(alpha0, beta0; outlier_halfwidth)`.

The evidence is the trapezoid scale quadrature of
[`loss_spec`](@ref)`(::ScaleMarginal)`.
"""
WelschMarginal(; alpha0::Real = 1.0, prior_loss_width::Real, outlier_halfwidth::Real) =
    _centred_marginal(WelschShape, alpha0, prior_loss_width, outlier_halfwidth)

# (`welsch_sq_scale_mle` — the per-pair Welsch squared-scale MLE, the argmax of
#  the Hermite-interpolated integrand — is deleted with its `_hermite_peak`: no
#  caller in RSDR, RVG, VisualGeometryToolkitEval or LODR.)

# The scale domain of every full-data scale marginal: t = log sq_loss_width ∈
# t0 ± _MARG_L around the prior centre (e^{±7} ≈ 1000× the centre scale),
# snapped to the absolute lattice t = i·h, and its default node count (the
# measured accuracy and speed against the Hermite spline it replaced:
# marg_sweep_ka.jl's header).
const _MARG_L  = 7.0
const _MARG_NG = 129

# The outlier support of ALL redescending marginals (Welsch / smooth-TLS /
# Geman-McClure): the domain `[0, r_max]` their density normalizer Z̃ is
# integrated over, in the pool frame's unit (`NormalizedImageFrame`, half the
# image diagonal; see VisualGeometryToolkitCore's `PoolPair`). It is a DECLARED constant — not a
# geometric bound, not a per-scene data max: every constant tuned for these
# marginals (the cross-validated prior centres and outlier half-widths, and the
# paper tables built on them) was cross-validated with r_max = 1, so the value is
# part of their definition, and one value keeps the truncation identical across
# scenes, probe and evaluation.
#
# It was first justified geometrically, for the FULL-diagonal frame: every point
# within radius ½ of the centre, so no residual exceeds the diagonal, 1. Pools are
# written in the HALF-diagonal frame, where a point-to-point residual reaches 2, so
# r_max = 1 truncates the outlier support at half the geometric maximum. It stays 1
# by decision (2026-09-22): changing it moves every marginal score and every
# constant tuned against it.
const POOL_FRAME_R_MAX = 1.0

# =============================================================================
# Fair common-cutoff map for the outlier half-width.
#
# Puts an outlier half-width on the SAME scale-multiple axis (σ-units) as MSAC's
# loss width (`scale_multiple·ŝ`).  The scale multiple is the inlier/outlier
# density-crossover point in σ-units; the marginals share the uniform outlier
# model, so their native map IS the Gaussian-Uniform boundary — `GauInlier`
# plus `decision_width` (scoring.jl).  No closed form is restated here.
# Tuning the scale multiple over the common window (e.g. [0.8, 6]) compares
# METHODS, not parameterizations.
# =============================================================================

# (`outlier_halfwidth_to_loss_width` / `loss_width_to_outlier_halfwidth`, the
#  conversion of the two native axes at an operating scale multiple, moved to
#  VisualGeometryToolkitCore's width_axes.jl with `to_native`.)

# (A Welsch outlier-half-width grid — the reference scale times `_unit_halfwidth` over a grid of
#  scale multiples — is deleted: no caller; the tuning layer (VisualGeometryToolkitEval)
#  builds its own grid.)
