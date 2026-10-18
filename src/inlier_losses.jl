# =============================================================================
# inlier_losses.jl — Track B: redescending losses as inlier/outlier MIXTURES
# =============================================================================
#
# Every robust loss is reformulated as an explicit two-component mixture that
# reuses the Student-t (`StudentTUniformScoring`) sort-and-sweep machinery:
#
#   p(r) = π · [ ∫ inlier(r; w) · p(w) dw ]  +  (1−π) · Uniform(−h, h)
#
# (w the loss width, h the outlier half-width). The loss width is
# MARGINALISED (like σ² in the Student-t); the outlier half-width is the single
# CV parameter (exactly like the Student-t's).
#
# The Gaussian inlier is the closed-form (conjugate) special case — it reduces
# to `_sweep_score(::IGPrior, …)` in sweep.jl bit-for-bit.  Welsch / Geman-
# McClure / smooth-TLS are NON-conjugate, so ∫dw is a 1-D quadrature over the
# log-squared-width axis t = log sq_loss_width (VGC's `s` is a squared
# RESIDUAL — see the name note in inlier_potentials.jl).
#
# Quadrature choice — FIXED-node trapezoid logsumexp, the package's ONE scale
# quadrature (this mixture and the full-data scale marginals of
# welsch_marginal.jl both run it: `_trapezoid_logevidence`, marg_sweep_ka.jl).
#   The log-integrand −ψ_k(t) is smooth and decays towards the domain
#   endpoints; ~100 nodes reproduce a 40001-node reference to ~1e-15 relative
#   when the endpoints carry no mass.  The 13-node cubic-Hermite spline the
#   full-data marginal used until 2026-09-30 was off by a median 3e-3 … 0.18 and
#   up to Inf in log-evidence (marg_sweep_ka.jl's header has the measurement),
#   which fails a <1e-6 accuracy requirement.  Fixed
#   nodes also make the per-node data sufficient statistic D_k(w²_j)
#   incremental in k (the evidence kernel `marg_mix_evidence_kernel!`).
#
# Per pair, codim-1 residual r (squared sq_r = r²), squared loss width w²:
#   Welsch      ρ̃ = ½(1 − e^{−sq_r/w²})                    ∈ [0, ½)
#   Geman-McC.  ρ̃ = ½ sq_r/(w² + sq_r)                     ∈ [0, ½)  (e^{−ρ_GM} density)
#   smooth-TLS  ρ̃ = 2·sq_r/w² − (sq_r/w²)²  (sq_r≤w²); 1   ∈ [0, 1]  (compact support)
# Inlier density  f(r; w) = e^{−ρ̃(r²; w²)} / Z̃(w²; h),  Z̃(w²; h) = ∫_{−h}^{h} e^{−ρ̃} dr.
# Minimal-sample points (sq_r=0) have ρ̃(0; w²)=0 and, like the Gaussian, are
# CONDITIONED ON — they contribute nothing to the data term and are excluded
# from the (k−m) inlier-normalizer count (mirrors ν = dg·(k−m) in the Gaussian).
#
# The Gaussian inlier density is a proper density over ℝ (Z ≡ 1), so the
# outlier half-width appears ONLY in its outlier term — never an inlier
# normalizer.  For the redescending losses the e^{−ρ̃} density has a flat floor
# (non-integrable on ℝ), so it must be truncated to [−h, h] via Z̃ — the outlier
# half-width appears in BOTH halves.
# =============================================================================

# -----------------------------------------------------------------------------
# The inlier-loss layer (canonical depths, ρ̃, Z̃ integrand) lives in
# inlier_potentials.jl — included FIRST so the scale marginals share it.
# This file owns the MIXTURE machinery built on top: quadrature grids and the
# evidence sweep.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LossQuad — fixed-node quadrature config, built once per (pair, shape)
# -----------------------------------------------------------------------------
#
# Holds the log-scale grid t_j ∈ [t0−L, t0+L], squared loss widths
# sq_loss_width_nodes[j] = exp(t_j), the data-independent per-node normalizer
# logZ̃, the prior neg-log base_j = α₀·t_j + β₀/w²_j, and the log trapezoid
# weight ltw_j = log(trapezoid weight·Δt) (½ at the two endpoints, 1 elsewhere).
# All the per-k evidence needs beyond these is the incremental data term D_k.
#
# `z_domain` is the UPPER LIMIT of the Z̃ integration domain [0, z_domain].
# `loss_spec` stores `r_max` there precisely BECAUSE its Z̃ is decoupled from
# the outlier half-width.
struct LossQuad
    z_domain::Float64
    prior::IGPrior
    t0::Float64
    L::Float64
    tnodes::Vector{Float64}
    sq_loss_width_nodes::Vector{Float64}
    logZnodes::Vector{Float64}
    base::Vector{Float64}      # α₀·t_j + β₀/sq_loss_width_nodes[j]
    ltw::Vector{Float64}       # log(trap weight · Δt)
end

# The four per-node tables every trapezoid scale-quadrature needs: squared loss
# widths, depth-Δ normalizers, IG-prior base, and log trapezoid weights, for
# `loss_spec`'s swept depth Δ.
function _quad_tables(shape::AbstractLossShape, Δ::Float64, tnodes::Vector{Float64},
                      r_max::Float64, prior::IGPrior)
    NG = length(tnodes)
    Δt = tnodes[2] - tnodes[1]
    sq_loss_width_nodes = exp.(tnodes)
    logZnodes = [truncated_log_normaliser(shape, sq_loss_width, Δ, r_max) for sq_loss_width in sq_loss_width_nodes]
    base = [prior.alpha0 * tnodes[j] + prior.beta0 / sq_loss_width_nodes[j] for j in 1:NG]
    ltw = Vector{Float64}(undef, NG)
    logΔt = log(Δt)
    @inbounds for j in 1:NG
        ltw[j] = (j == 1 || j == NG) ? (log(0.5) + logΔt) : logΔt
    end
    return sq_loss_width_nodes, logZnodes, base, ltw
end

"""
    loss_scale_bounds(sq_residuals, n; pad=4.0) -> (t_lo, t_hi)

Data-driven grid bounds on `t = log sq_loss_width` for a redescending-loss sweep
over squared residuals `sq_residuals[1:n]`.  The scale-posterior peak for
partition k lands near `sq_loss_width ≈ RSS_k/(k−m)`, which ranges from ≈ the
tightest residual scale (small k) to ≈ the all-point scale (k=n); the returned
`[t_lo, t_hi]` brackets that range with `pad` decades (natural-log) of margin on
each side so the trapezoid grid covers every k's peak.  Supply to
`loss_spec(...; t_lo, t_hi)`.
"""
function loss_scale_bounds(sq_residuals::AbstractVector{Float64}, n::Int; pad::Float64=4.0)
    min_sq_loss_width = Inf; max_sq_loss_width = 0.0
    @inbounds for i in 1:n
        x = sq_residuals[i]
        (isfinite(x) && x > 0) || continue
        x < min_sq_loss_width && (min_sq_loss_width = x)
        x > max_sq_loss_width && (max_sq_loss_width = x)
    end
    max_sq_loss_width <= 0 && (max_sq_loss_width = 1.0)
    min_sq_loss_width == Inf && (min_sq_loss_width = max_sq_loss_width)
    # Cap the dynamic range at 24 natural-log units (~10 decades) so a single
    # accidentally-tiny residual cannot drag t_lo down and starve resolution
    # near the scale-posterior peak.
    tlo = max(log(min_sq_loss_width), log(max_sq_loss_width) - 24.0) - pad
    return (tlo, log(max_sq_loss_width) + pad)
end

# (Deleted, with no caller in RSDR, RVG, VisualGeometryToolkitEval or LODR: the fixed-depth
#  Stage 1–4 layer — the batch `_inlier_logev`, the trait-parameterised
#  quadrature `sweep!` pair, and the marginalised-scale responsibilities with
#  their `_scale_post_logweights!` and the Gaussian-shape `posterior_*!`
#  forwarders. The two `sweep!` methods were also an Aqua-reported ambiguity.
#  The Δ-parameterised `LossSpec` layer below is the live Track-B CPU path; the
#  device kernels mirror it.)

# =============================================================================
# Stage 5 (SELECT) — LossSpec: a Δ-parameterised quadrature-marginal selector
# =============================================================================
#
# `LossSpec` bundles (shape, well-depth Δ, quadrature grid) so Δ can never
# drift out of sync with the Z̃ nodes baked at that Δ.  The grid's Z̃ is
# normalised over the FIXED [0, r_max] domain, so the inlier evidence E_in(k)
# does NOT depend on the outlier half-width — compute it ONCE per Δ, then
# profile the outlier half-width by an O(n) re-argmax over k (the device
# `marg_mix_profile_kernel!`).
#
# The Gaussian (Student-t) member is Δ→∞: it uses the closed-form Gamma sweep
# (`StudentTUniformScoring`), NOT this quadrature — it is the anchor the finite-Δ
# curves converge onto.
# =============================================================================
"""
    LossSpec{K<:AbstractLossShape}

The depth-`Δ` quadrature-marginal spec of the redescending loss shape `K`: the
shape, its well depth `Δ`, the quadrature grid with `Z̃` baked at that depth
over `[0, r_max]`, and `r_max`.

Build it with [`loss_spec`](@ref), which keeps `Δ` and the grid in step, and
evaluate a staged pool with it through [`grid_evaluate`](@ref).
"""
struct LossSpec{K<:AbstractLossShape}
    shape::K
    Δ::Float64
    quad::LossQuad     # quad.logZnodes baked at depth Δ over [0, r_max]
    r_max::Float64
end

"""
    loss_spec(shape, prior; Δ, r_max=POOL_FRAME_R_MAX, t_lo, t_hi, NG=1537) -> LossSpec

Build a depth-`Δ` quadrature-marginal spec for the loss `shape` under IG scale
`prior`. Z̃ is normalised over the FIXED domain `[0, r_max]` (the declared
outlier support `POOL_FRAME_R_MAX`, in the pool frame's half-diagonal unit), so
the outlier half-width is decoupled from the quadrature.  `[t_lo, t_hi]` are the
log-scale grid bounds (use `loss_scale_bounds(sq_residuals, n)`).
"""
function loss_spec(shape::AbstractLossShape, prior::IGPrior; Δ::Real,
                   r_max::Real=POOL_FRAME_R_MAX, t_lo::Real, t_hi::Real,
                   NG::Int=1537)
    return _loss_spec(shape, prior, Float64(Δ), Float64(r_max),
                      collect(range(Float64(t_lo), Float64(t_hi); length=NG)))
end

"""
    loss_spec(m::ScaleMarginal{K}; NG=_MARG_NG) -> LossSpec

The quadrature of the full-data scale marginal `m`: shape `K` at its canonical
depth, `m`'s IG prior, Z̃ over `[0, m.outlier_halfwidth]` and `NG` log-scale nodes spanning `m.t0 ± _MARG_L`. The
nodes lie on the ABSOLUTE lattice `t = i·h`, `h = 2·_MARG_L/(NG − 1)` (the
domain snapped by at most `h/2`), so the specs of every grid point share their
nodes and `grid_evaluate(::ScaleMarginal, …)` sums each model's data term once for the whole
grid.
"""
function loss_spec(m::ScaleMarginal{K}; NG::Int=_MARG_NG) where {K}
    shape = K()
    i0 = _marg_lattice_start(m.t0, NG)
    return _loss_spec(shape, IGPrior(m.alpha0, m.beta0), _default_depth(shape), m.outlier_halfwidth,
                      _marg_lattice(i0:(i0 + NG - 1), NG))
end

# The absolute log-scale lattice of the full-data marginals: node i at t = i·h.
_marg_lattice_spacing(NG::Int) = 2 * _MARG_L / (NG - 1)
_marg_lattice(is, NG::Int) = [i * _marg_lattice_spacing(NG) for i in is]
_marg_lattice_start(t0::Float64, NG::Int) = round(Int, (t0 - _MARG_L) / _marg_lattice_spacing(NG))

function _loss_spec(shape::AbstractLossShape, prior::IGPrior, Δ::Float64, r_max::Float64,
                    tnodes::Vector{Float64})
    tlo = first(tnodes); thi = last(tnodes)
    sq_loss_width_nodes, logZnodes, base, ltw = _quad_tables(shape, Δ, tnodes, r_max, prior)
    quad = LossQuad(r_max, prior, (tlo + thi) / 2, (thi - tlo) / 2,
                    tnodes, sq_loss_width_nodes, logZnodes, base, ltw)
    return LossSpec(shape, Δ, quad, r_max)
end

# (`inlier_evidence!` and the host `sweep!` over a `LossSpec` — the host
#  evaluation of this mixture's evidence and its outlier-half-width profile,
#  with their `_logev_from_acc` logsumexp — are deleted: the evidence has ONE
#  evaluation, the KernelAbstractions kernels under `grid_evaluate(::LossSpec, …)`
#  (marg_sweep_ka.jl), which run on CPU() as on a GPU. No caller outside this
#  package used the host copy; it was the device kernel's bit-exact reference,
#  and the gate now checks the kernel against a plain host loop instead.)
