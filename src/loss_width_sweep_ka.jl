# =============================================================================
# Loss-width-family scoring — KernelAbstractions, CPU() or a GPU backend.
#
# Every loss-width family (RANSAC, MSAC, MLESAC, MAGSAC++, and the Welsch /
# smooth-TLS / Geman-McClure losses) has ONE per-residual quality
# `residual_quality(scoring, d², loss_width)`. Its grid here is the LOG-BINNED
# aggregation, `LogBinned(per_decade)`: residuals histogrammed on a log-d² grid,
# the SAME quality evaluated at bin centres — the fast cross-validation path,
# `histogram_pm_score_kernel!`. The EXACT selection (every residual through the
# quality) is VisualGeometryToolkitCore's: `select_on_grid!` on the `:pool`
# staging, `exact_pm_score_kernel!` + `argmax_models_kernel!` (this package's
# second launcher of those kernels, the `Exact` aggregation, is deleted,
# 2026-11-12). The model argmax and error gather are the shared
# `argmax_models_kernel!` / `_chunked_argmax_eval` (marg_sweep_ka.jl). Reads
# `IGSharedData.sorted_sq_flat` (per-model squared residuals, sorted ascending,
# in the pool's precision).
#
# Usage:
#   staged = stage_ig_shared_from_pairs(pairs, residuals; backend)
#   errs = grid_evaluate(loss_width_scoring("magsac", codimension(PT)),
#                        staged, loss_widths)                       # Float32, LogBinned(200)

# (`_ka_softplus` lived here, byte-identical to `_softplus` in
# scoring_eval_common.jl. The `_ka_` prefix implied a device constraint there is
# none: `_build_score_weight_matrix` fills a host `Array{Float32}`, so the host
# definition serves. Deleted; this file uses LogExpFunctions' `log1pexp` and
# `_SOFTPLUS_HALF`.)

# =============================================================================
# LogBinned — the loss-width grid's aggregation of a family's per-residual quality
# =============================================================================

"""
    LogBinned(bins_per_decade = 200)

The fast cross-validation aggregation: residuals histogrammed on a log-spaced d² grid.
The grid has `bins_per_decade` bins per decade, spanning 0.01·min_loss_width² to
100·max_loss_width², and the family's quality is evaluated at the bin centres. Out-of-range and non-finite
residuals score as outliers.
"""
struct LogBinned
    bins_per_decade::Int
end
LogBinned() = LogBinned(200)

# (The exact selection is VisualGeometryToolkitCore's `exact_pm_score_kernel!` under its
# `select_on_grid!` on the `:pool` staging: per (pair, model) every residual through the
# family's `residual_quality`, at every loss width, in Float64.)

# =============================================================================
# Histogram-based score kernel (Shekhovtsov ScoreWeightsGU pattern)
# =============================================================================
# Per-residual scoring `f(d², loss_width)` costs O(softplus) or O(gamma_inc) per call.
# Per (pair, model, loss width) the naive cost is `K · T · f_cost`.
#
# Histogram trick: precompute a bin-weight matrix
#   W[bin, t] = residual_quality(scoring, d²_bin, loss_width_t)
# on a *log-spaced* grid in d² (one shot on the host).  Per (pair, model, t)
# the kernel then:
#   1. for each k in 1..K: bin = floor((log(d²) - log d²_min) · inv_log_spacing) + 1
#   2. accumulate `s_t += W[bin, t]` for all T loss widths (one residual read,
#      T cheap loads from cached `W`).
# Argmax over models is preserved within binning resolution; empirically
# 99.5% per-pair argmax match at LogBinned(200).  Log spacing
# (vs Shek's linear bins) gives uniform per-decade resolution, which is
# what our 5-decade log probe grid needs — linear bins squashed small-loss-width
# signal into bin 0 and broke argmax.
#
# This is the `LogBinned` aggregation of every loss-width family; the exact
# selection is VisualGeometryToolkitCore's `exact_pm_score_kernel!`.

# --- Kernel 1: per (pair, model) thread, LOG-spaced bins ---
# Bin a residual d² ∈ (d²_min, d²_max] into N_bins log-spaced bins:
#   bin = floor(log(d²/d²_min) · N_bins/log_range) + 1
# Out-of-range residuals (d² < d²_min or d² > d²_max) snap to the nearest end
# bin where W is already ≈0 → no spurious contribution.
#
# Log spacing matches our 5-decade probe grid resolution per decade and
# gives every loss width in the grid well-resolved bins.  Linear spacing would
# collapse small-loss-width signal into bin 0 and break argmax over models.
@kernel function histogram_pm_score_kernel!(
        scores_out,                       # (chunk_size, M_max, T) Float64 — drift-safe accumulator
        @Const(sorted_sq_flat),           # Float32
        @Const(pair_offset),
        @Const(K_per_pair),
        @Const(M_per_pair),
        @Const(W_flat),                   # Float32 (N_bins · T) col-major
        N_bins::Int32,
        log_d2_min,                       # eltype drives FP32/FP64 path for log()
        inv_log_spacing,                  # same eltype as log_d2_min
        T_val::Int32,
        p_offset::Int32)                  # chunk_local p_local + p_offset → p_global
    p_local, m = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)      # global pair index for staging-array reads
    M = Int(M_per_pair[p])
    if m <= M
        K = Int(K_per_pair[p])
        nb = Int(N_bins)
        Tn = Int(T_val)
        base = pair_offset[p]
        col_base = base + (Int64(m) - Int64(1)) * Int64(K)

        @inbounds for t in 1:Tn
            scores_out[p_local, m, t] = 0.0
        end
        @inbounds for k in 1:K
            # Bin lookup: precision driven by `log_d2_min` / `inv_log_spacing` eltype.
            # Float32 here keeps the bin lookup cheap on device.
            # Bin-index flips at ±1 near boundaries
            # are bounded; downstream argmax stability is verified per-cell.
            d2 = sorted_sq_flat[col_base + k]   # Float32 load
            # A NaN or +Inf residual is an OUTLIER. Every histogram family scores
            # an outlier at quality 0 (MAGSAC++ and smooth-TLS exactly, beyond
            # loss_width; MLESAC and Welsch/Geman in the limit), so skipping it adds
            # exactly that — as Core's exact kernel and the per-pair
            # `best_pool_model` do (and the deleted host histogram,
            # `bin_residuals_stereo`, did). It used to fall through: NaN
            # failed `d2 > 0` and took bin 1, the PERFECT-inlier weight, and +Inf
            # reached `floor(Int, Inf)` (an InexactError on the CPU backend).
            isfinite(d2) || continue
            bin = d2 > 0 ? floor(Int, (log(d2) - log_d2_min) * inv_log_spacing) + 1 : 1
            bin = min(nb, max(1, bin))
            for t in 1:Tn
                # Score accumulator stays Float64 (K residuals × T loss widths
                # is enough to drift in Float32).
                scores_out[p_local, m, t] += Float64(W_flat[bin + (t - 1) * nb])
            end
        end
    end
end

# --- Kernel 2: per (pair, loss width) thread, argmax over m and the error gather: VisualGeometryToolkitCore's
# `argmax_models_kernel!` (every grid evaluator's, `_chunked_argmax_eval`).

# -----------------------------------------------------------------------------
# Threshold-scoring families (D): singleton types replacing the `method::Symbol`
# + if/elseif kernel selection. `grid_evaluate` dispatches on these via
# `residual_quality` and `support_sq_loss_width` below; a config's tag converts at the boundary with
# `loss_width_scoring("ransac", dg)` (the tags are `ALL_SCORING_TAGS`).
# -----------------------------------------------------------------------------
# The loss-width-swept scorings, split by whether a mixture sits underneath (the two
# branches and the traits are declared in scoring.jl).
#
# `ThresholdFamily` used to be the supertype of all seven. It could not be:
# "takes a loss width" is true of both branches, so using it as the hierarchy forced
# the mixture branch's members and StudentTUniformScoring into disconnected roots even
# though they posit the SAME model. The members that take a loss width are now the
# `LossWidthScoring` union, which is what the sweep entry points accept.
#
# The mixture branch's fixed-scale member is MLESAC, the marginal likelihood over
# labels at fixed scale. Its tag is `:mlesac` (the tag rule: the type's name without
# its family suffix).
# EVERY ONE CARRIES ITS CODIMENSION, as `StudentTUniformScoring{P, DG}` does. loss width is
# not a bare length: for a codim-DG residual with Gaussian inliers
# ||r||^2/sigma^2 ~ chi^2_DG, so one quantile factor q maps to a DIFFERENT
# multiple of sigma per codimension, and the outlier density carries DG as the
# exponent `(2·outlier_halfwidth)^-DG`. A scoring that cannot state its codimension cannot state
# what its loss width means.
#
# It was previously unstateable: `dg` arrived as a keyword on
# `grid_evaluate_loss_width` (then `_ka`) "for API parity", defaulted to 0, and was dropped —
# so the loss-width branch silently ran at one codimension while the marginal
# branch tracked the problem's.
#
# DG DEFAULTS TO 1 for a bare `T()`, as VisualGeometryToolkitCore's
# `deploy(shape, s2; codim)` defaults; a problem-aware constructor states it.
# (The outlier-half-width <-> loss-width crossing,
# `outlier_halfwidth_to_loss_width`/`loss_width_to_outlier_halfwidth`, takes its
# `d_g` explicitly.)
# RANSAC and MSAC are VisualGeometryToolkitCore's baseline scorers (`RansacScoring`,
# `MsacScoring`, with their `residual_quality`); this package adds their grid
# methods and the paper baselines below.

# Each family has VisualGeometryToolkitCore's two forms of `RansacScoring{DG, T}` /
# `MsacScoring{DG, T}`: WIDTH-FREE (`T = Nothing`, `F()`, `F(dg)`, `F(problem)`),
# the grid family a loss width is swept over, and AT A LOSS WIDTH
# (`F(dg; threshold = w)`, `T = Float64`, or `with_width(F(dg), LossWidth(w))`),
# a scorer of the scoring contract's `score!` — the form VisualGeometryToolkitCore's
# `best_pool_model` and VisualGeometryToolkitEval's sample-budget engine select with.

"""
    MlesacScoring{DG, T}

MLESAC (GaU) at codimension `DG`: the Gaussian-uniform log-density at a fixed
scale, normalised to 1 at `d = 0` (see [`residual_quality`](@extref VisualGeometryToolkitCore.residual_quality)). Tag
`"mlesac"` ([`ALL_SCORING_TAGS`](@ref)). `MlesacScoring()`, `MlesacScoring(dg)`,
`MlesacScoring(problem)`: the width-free grid family; `MlesacScoring(dg; threshold = w)`:
the scorer at loss width `w`.
"""
struct MlesacScoring{DG, T} <: AbstractThresholdScoring{T}
    threshold::T
end

"""
    MagsacScoring{DG, T}

MAGSAC++ at codimension `DG`: the σ-marginalised quality, zero from the cutoff
on; its loss width is that cutoff, `_MAGSAC_SCALE_MULTIPLE·σ_max` (see
[`residual_quality`](@extref VisualGeometryToolkitCore.residual_quality)). Tag `"magsac"`;
the forms of [`MlesacScoring`](@ref).
"""
struct MagsacScoring{DG, T} <: AbstractThresholdScoring{T}
    threshold::T
end

"""
    WelschCostScoring{DG, T}

The Welsch cost as a loss-width score at codimension `DG`: `1 − ρ/w²` with
VisualGeometryToolkitCore's Welsch `ρ`, no mixture underneath. Tag `"welsch_cost"`;
the forms of [`MlesacScoring`](@ref).
"""
struct WelschCostScoring{DG, T} <: LossCostScoring{T}
    threshold::T
end

"""
    SmoothTLSScoring{DG, T}

The smooth truncated-least-squares cost as a loss-width score at codimension
`DG`: `1 − ρ/w²` with VisualGeometryToolkitCore's smooth-TLS `ρ`. Tag
`"smooth_tls"`; the forms of [`MlesacScoring`](@ref).
"""
struct SmoothTLSScoring{DG, T} <: LossCostScoring{T}
    threshold::T
end

"""
    GemanScoring{DG, T}

The Geman–McClure cost as a loss-width score at codimension `DG`: `1 − ρ/w²`
with VisualGeometryToolkitCore's Geman–McClure `ρ`. Tag `"geman"`; the forms of
[`MlesacScoring`](@ref).
"""
struct GemanScoring{DG, T} <: LossCostScoring{T}
    threshold::T
end

"""
    LossWidthScoring{DG}

The loss-width-parameterised scorings of both branches at codimension `DG` — the
paper's "threshold-based family". A Union, not a supertype: "takes a loss width" cuts
across the mixture branch and `LossCostScoring`, so it cannot be a supertype of both
without implying a mixture underneath the losses.

This is the type the loss-width sweep entry points accept. Widening them to
`AbstractScoring` would readmit `StudentTUniformScoring`, which has no loss width and would
fail several frames down inside GPU staging instead of at the call site.
"""
const LossWidthScoring{DG} = Union{RansacScoring{DG}, MsacScoring{DG}, MlesacScoring{DG},
                             MagsacScoring{DG}, WelschCostScoring{DG},
                             SmoothTLSScoring{DG}, GemanScoring{DG}}
# The paper baselines this package owns (RANSAC/MSAC's are VisualGeometryToolkitCore's).
const _OwnLossWidthScoring{DG} = Union{MlesacScoring{DG}, MagsacScoring{DG}, WelschCostScoring{DG},
                                       SmoothTLSScoring{DG}, GemanScoring{DG}}

# At a loss width the five are VisualGeometryToolkitCore `AbstractThresholdScoring{Float64}`s:
# Core's one threshold loop scores them (`score!` on a `ThresholdScoringWorkspace`,
# with its consensus set d² < w²), from their `residual_quality` below.
#
# Their `score_bound`s, the per-point maximum b with 0 ≤ quality ≤ b for every
# finite d² ≥ 0 (to a few ulps of b, which the early exit's margin absorbs):
#   MLESAC     log1pexp((w² − d²)/(2w²))/log1pexp(½): the argument is ≤ ½, the
#              value at d = 0 is 1.0 exactly, and log1pexp ≥ 0. b = 1.
#   Welsch     1 − ρ/w² = exp(−d²/w²): −expm1(−s/w²) ∈ [0, 1]. b = 1.
#   SmoothTLS  (1 − d²/w²)² below w², 0 from w² on. b = 1.
#   Geman      w²/(w² + d²) ∈ (0, 1]. b = 1. (A finite d² so large that w²·d²
#              overflows scores −Inf, which only lowers the sum; a sample point
#              there makes the full score NaN, which beats no bound.)
#   MAGSAC++   raw_cut − ρ(r): ρ is a loss, 0 at r = 0 and increasing to raw_cut
#              at the cutoff, and 0 past it, so b = raw_cut, the quality at
#              d = 0, which depends on the width.
# Measured 2026-11-07 on 2.2·10⁵ residuals per family and width (w from 0.01 to
# 10³, d/w from 0 to 10³⁰): no quality above its b, none below −1 ulp of b.
score_bound(::Union{MlesacScoring{<:Any, Float64}, WelschCostScoring{<:Any, Float64},
                    SmoothTLSScoring{<:Any, Float64}, GemanScoring{<:Any, Float64}}) = 1.0
score_bound(sc::MagsacScoring{<:Any, Float64}) = residual_quality(sc, 0.0, decision_boundary(sc, nothing))

codimension(::_OwnLossWidthScoring{DG}) where {DG} = DG

# The constructors and the width setter are VisualGeometryToolkitCore's, exactly as
# `RansacScoring`/`MsacScoring` have them (`@threshold_scoring_constructors`):
# bare `F()` is codim 1 (see above); `F(dg; threshold)`, `F(problem; threshold)`
# and `F(P; threshold)` at a loss width (positive and finite), width-free
# without one; `with_width(::F{DG}, LossWidth(w))`. (This package's own loop and
# `_at_loss_width` are deleted, 2026-11-12.)
@threshold_scoring_constructors MlesacScoring
@threshold_scoring_constructors MagsacScoring
@threshold_scoring_constructors WelschCostScoring
@threshold_scoring_constructors SmoothTLSScoring
@threshold_scoring_constructors GemanScoring

# Where these sit in the elimination table (scoring.jl): RANSAC and MSAC profile
# the labels, MLESAC and MAGSAC++ marginalise them, and every one of the seven
# runs at a supplied scale — that is what having a loss width MEANS. RANSAC is MSAC with
# the inlier RSS term dropped, keeping only the count, which is why it is the one
# baseline measurably worse.

# The cost branch scores by a VisualGeometryToolkitCore shape. All three carry a
# 95%-efficiency constant in VGC (`efficiency_constant(shape, codim)`), so each
# has its own loss width `efficiency_constant·ŝ`. `SmoothTLSShape` is not an `AbstractMEstimator` (a
# truncation has no influence function), but VGC tabulates its constant anyway,
# so membership of that supertype is not what decides whether a loss width exists.
loss_shape(::WelschCostScoring) = VgtkCore.WelschShape()
loss_shape(::GemanScoring)      = VgtkCore.GemanMcclureShape()
loss_shape(::SmoothTLSScoring)  = VgtkCore.SmoothTLSShape()

# ...AND MSAC, which is not a coincidence: profiling the labels is a max over
# z in {0,1}, i.e. a min of the two NLLs, and a min of a quadratic against a
# constant IS a truncated rho of one width. Marginalising the labels is instead
# a log-sum-exp depending on sigma and a JOINTLY, which `rho(shape, s, sq_loss_width)`
# cannot express — so MlesacScoring/MagsacScoring correctly have no method here.
#
# MSAC's cost is literally `min(d^2, loss_width^2)`, which is `rho(TLSShape(), s, sq_loss_width)`
# now that VGC's rho is total over the family (0.29: it used to be scoped to
# the Geman-Reynolds liftable shapes).
#
# RANSAC is the SAME shape under the other reduction: its count is the indicator
# `d^2 < loss_width^2`, which is exactly `weight(TLSShape(), s, sq_loss_width) = d/ds min(s, sq_loss_width)`.
# A `CountShape` was briefly defined here to say that with a rho of its own;
# deleted, because it restated a VGC shape RSDR already names one line up, and
# it joined VGC's loss family while implementing only `rho` — inheriting the
# rest of the interface by accident, including a `bounded` default that is wrong
# for it. No `loss_shape(::RansacScoring)`: RANSAC's quality is TLS's
# `weight` directly (`residual_quality` below), so nothing would consume one.
# (`loss_shape(::MsacScoring) === TLSShape()` is VisualGeometryToolkitCore's, beside the type.)

# (A loss scoring's efficiency loss width, `efficiency_constant(shape, codim)·ŝ`,
#  has no accessor here: nothing consumes one, and accessors are added only where
#  something does. The arithmetic is VGC's: `loss_width(deploy(shape, ŝ²; codim))`.)

"""
    loss_width_scoring(tag) -> LossWidthScoring
    loss_width_scoring(tag, dg::Integer) -> LossWidthScoring
    loss_width_scoring(tag, problem) -> LossWidthScoring
    loss_width_scoring(tag, ::Type{<:AbstractConstraintProblem}) -> LossWidthScoring

The width-free [`LossWidthScoring`](@ref) family a tag of [`ALL_SCORING_TAGS`](@ref)
names, at a codimension: `"ransac"`, `"msac"`, `"mlesac"`, `"magsac"`,
`"welsch_cost"`, `"smooth_tls"` or `"geman"`. A retired tag (`"gau"`) is an
`ArgumentError` naming its successor, and a tag of a family that takes no loss
width (`"student_t_uniform"`, `"sim_fit_pp"`) is refused. The codimension is
`dg`, or `codimension(problem)`; bare, it is 1.

Supply the codimension whenever the problem has one other than 1: the tag
vocabulary must not be a back door that silently re-defaults `DG`.
"""
function loss_width_scoring(tag::AbstractString, dg::Integer = 1)
    S = ALL_SCORING_TAGS[tag]
    (S isa Type && S <: LossWidthScoring) || throw(ArgumentError(
        "loss_width_scoring: tag \"$tag\" names $S, which takes no loss width; the loss-width tags are " *
            join(("\"$t\"" for (t, F) in ALL_SCORING_TAGS if F isa Type && F <: LossWidthScoring), ", ")))
    return S(dg)
end
loss_width_scoring(tag::AbstractString, problem::AbstractConstraintProblem) =
    loss_width_scoring(tag, codimension(problem))
loss_width_scoring(tag::AbstractString, ::Type{P}) where {P <: AbstractConstraintProblem} =
    loss_width_scoring(tag, codimension(P))

# =============================================================================
# The per-residual QUALITY — one scalar per family, of the SQUARED residual d²
# and the loss width. Core's exact kernel calls it at every residual,
# LogBinned at the bin centres. Plain Float64 functions, callable inside a
# kernel (MAGSAC++'s incomplete gammas included).
# =============================================================================

# The per-residual QUALITY is VisualGeometryToolkitCore's `residual_quality` (its
# RANSAC and MSAC methods beside those types); the paper baselines add theirs.

# The losses: 1 − ρ/sq_loss_width, VisualGeometryToolkitCore's ρ normalised by its
# depth sq_loss_width = loss_width². IT IS THE QUALITY, NOT THE IRLS WEIGHT:
# scoring sums a per-point contribution, while VGC's `weight` is dρ/ds. They
# coincide for Welsch only — Geman-McClure's weight is the SQUARE of its
# quality. Depth is written as sq_loss_width rather than probed as
# ρ(sh, Inf, sq_loss_width), which is NaN for Geman-McClure.
@inline function residual_quality(sc::LossCostScoring, d2::Float64, loss_width::Float64)
    sq_loss_width = loss_width * loss_width
    return 1.0 - VgtkCore.ρ(loss_shape(sc), d2, sq_loss_width) / sq_loss_width
end

# MLESAC: the Gaussian-uniform log-density, normalised by its value at d = 0.
@inline function residual_quality(::MlesacScoring, d2::Float64, loss_width::Float64)
    sq_loss_width = loss_width * loss_width
    return log1pexp((sq_loss_width - d2) / (2.0 * sq_loss_width)) / _SOFTPLUS_HALF
end

# MAGSAC++: `raw_cut − raw_i`, the paper's binned partition formula (Sec 5.2.4),
# zero from the cutoff on. It is the PAPER's ρ (Barath, Noskova, Ivashechkin &
# Matas, MAGSAC++ §2.2), for 0 ≤ r ≤ κ·σ_max with R = r²/(2σ_max²) and
# κ = `_MAGSAC_SCALE_MULTIPLE`:
#
#   ρ(r) = C(n)·2^{(n+1)/2}/σ_max
#          · [ σ_max²/2 · γ((n+1)/2, R) + r²/4 · ( Γ((n−1)/2, R) − Γ((n−1)/2, κ²/2) ) ]
#
# saturating at ρ(κ·σ_max) beyond; the quality is ρ(κ·σ_max) − ρ(r). superansac's
# C++ deviates in three places: it takes σ_max²/4 for the per-residual r²/4, it
# truncates at σ_max (it computes κ²σ_max² and never reads it), and it writes
# C(n) = 2^{-n/2}·Γ(n/2) for (2^{n/2}·Γ(n/2))⁻¹. The truncation matters most: the
# paper's weight falls to ZERO at κ·σ_max, which makes ρ a valid M-estimator
# loss; cutting at σ_max severs it at ~80% of its peak. (C(n) is a constant
# factor and cannot change the arg-best.) THE Γ FACTORS ARE THE POINT: the reference
# (Shekhovtsov ScoreWeightsMAGSAC._WS) is written in the UNNORMALIZED incomplete
# gammas γ(o,x) = Γ(o)·P(o,x) and Γ(o,x) = Γ(o)·Q(o,x); Julia's `gamma_inc`
# returns the regularized P, Q. The constants that depend only on the DOF and
# the cutoff κ are evaluated once, below.
const _MAGSAC_C      = 1.0 / (2.0^(_MAGSAC_DOF/2) * gamma(_MAGSAC_DOF/2) * 0.99)
const _MAGSAC_ORD_UP = (_MAGSAC_DOF - 1)/2      # incomplete-gamma ORDERS, not half-widths
const _MAGSAC_ORD_LO = (_MAGSAC_DOF + 1)/2
const _MAGSAC_Γ_UP   = gamma(_MAGSAC_ORD_UP)
const _MAGSAC_Γ_LO   = gamma(_MAGSAC_ORD_LO)
const _MAGSAC_P1_CUT = gamma_inc(_MAGSAC_ORD_LO, _MAGSAC_SCALE_MULTIPLE^2/2)[1]
const _MAGSAC_Q_CUT  = gamma_inc(_MAGSAC_ORD_UP, _MAGSAC_SCALE_MULTIPLE^2/2)[2]
@inline function residual_quality(::MagsacScoring, d2::Float64, loss_width::Float64)
    d2 < loss_width * loss_width || return 0.0
    σ_max = loss_width / _MAGSAC_SCALE_MULTIPLE   # MAGSAC++ parametrises σ̄ = loss_width/κ
    coeff = _MAGSAC_C * 2.0^_MAGSAC_ORD_LO / σ_max
    raw_cut = coeff * σ_max^2 / 2 * _MAGSAC_Γ_LO * _MAGSAC_P1_CUT
    R = d2 / (2 * σ_max^2)
    _, Q_R = gamma_inc(_MAGSAC_ORD_UP, R); P1R, _ = gamma_inc(_MAGSAC_ORD_LO, R)
    raw_i = coeff * (σ_max^2 / 2 * _MAGSAC_Γ_LO * P1R +
                     d2 / 4 * _MAGSAC_Γ_UP * (Q_R - _MAGSAC_Q_CUT))
    return raw_cut - raw_i
end

# Compact support: the quality is exactly 0 from loss_width on, so on sorted residuals
# nothing past max_loss_width² scores (Core's exact kernel stops there). MLESAC and the
# Welsch / Geman-McClure costs reach 0 only asymptotically.
# (VisualGeometryToolkitCore's `support_sq_loss_width`, whose default is Inf and which states RANSAC's and MSAC's.)
VgtkCore.support_sq_loss_width(::Union{MagsacScoring, SmoothTLSScoring}, max_loss_width::Float64) = max_loss_width * max_loss_width
# Every member of this package's loss-width families scores by the sum of its per-residual quality (Core's
# `has_residual_quality`; RANSAC's and MSAC's are Core's).
VgtkCore.has_residual_quality(::Union{MlesacScoring, MagsacScoring, WelschCostScoring, SmoothTLSScoring, GemanScoring}) = true

"""
    _build_score_weight_matrix(scoring, loss_widths, N_bins, d2_min, d2_max)
        -> Matrix{Float32}  size (N_bins, T)

Precompute `W[bin, t] = residual_quality(scoring, d²_bin, loss_width_t)` on the host at the bin
centres of a log-spaced grid. The centres are `d²_bin = d²_min · (d²_max/d²_min)^((bin-0.5)/N_bins)`,
and `W` holds the SAME scalar quality Core's exact kernel evaluates per residual. Returned
column-major so the kernel can index `W[bin + (t-1)·N_bins]`.
"""
function _build_score_weight_matrix(scoring::LossWidthScoring,
                                     loss_widths::AbstractVector{<:Real},
                                     N_bins::Int,
                                     d2_min::Float64, d2_max::Float64)
    T = length(loss_widths)
    W = Array{Float32}(undef, N_bins, T)
    log_spacing = log(d2_max / d2_min) / Float64(N_bins)
    log_d2_min = log(d2_min)
    # Bin centres depend only on `bin`, so they are built once rather than
    # re-exponentiated for every loss-width column (T is typically ~200).
    d2c = [exp(log_d2_min + (Float64(bin) - 0.5) * log_spacing) for bin in 1:N_bins]
    @inbounds for t in 1:T
        loss_width = Float64(loss_widths[t])
        for bin in 1:N_bins
            W[bin, t] = Float32(residual_quality(scoring, d2c[bin], loss_width))
        end
    end
    return W
end

# =============================================================================
# Dispatcher
# =============================================================================

"""
    grid_evaluate([S = Float32,] scoring::LossWidthScoring, staged::IGSharedData, loss_widths;
                  aggregation = LogBinned(), return_idxs = false, pair_chunk_size = 256)
        -> Matrix{Float64}

Per-pair selection errors of a loss-width family over a loss-width grid. They form a `(P, T)` matrix
of the error (imc) of the model `scoring` selects at each loss width in `loss_widths`
(`NormalizedImageFrame` units; with `return_idxs`, also the selected pool
indices). Runs on the staged data's backend — KernelAbstractions `CPU()` or a GPU.

`aggregation` is the [`LogBinned`](@ref) aggregation (default 200 bins per
decade; the fast cross-validation path) of the family's one per-residual quality.
The exact selection (every residual) is VisualGeometryToolkitCore's
`select_on_grid!` on the `:pool` staging. `S` is the bin-index precision
(Float32, the default, is the production setting); scores accumulate in Float64.

`scoring` is a `LossWidthScoring` member — build it with [`loss_width_scoring`](@ref),
supplying the problem's codimension.
"""
function grid_evaluate(::Type{S}, scoring::LossWidthScoring, staged::IGSharedData,
                       loss_widths::AbstractVector{<:Real};
                       aggregation::LogBinned = LogBinned(),
                       return_idxs::Bool = false,
                       pair_chunk_size::Int = 256) where {S<:AbstractFloat}
    P = length(staged.K_per_pair)
    cs = max(1, min(pair_chunk_size, P))
    return _grid_evaluate_loss_width(aggregation, S, scoring, staged,
                              Vector{Float64}(loss_widths), cs, return_idxs)
end
grid_evaluate(scoring::LossWidthScoring, staged::IGSharedData, loss_widths::AbstractVector{<:Real};
              kwargs...) =
    grid_evaluate(Float32, scoring, staged, loss_widths; kwargs...)

# The LOG-binned weight-matrix trick, pair-chunked to keep `scores_pmt` bounded
# (at M_max≈5500, T=200, a full 20k-pair pool is 179 GiB; a 256-pair chunk ~2.3).
function _grid_evaluate_loss_width(agg::LogBinned, ::Type{S}, scoring, shared, loss_widths::Vector{Float64},
                            cs::Int, return_idxs::Bool) where {S}
    backend = shared.backend
    T = length(loss_widths); M_max = Int(shared.M_max)
    min_loss_width, max_loss_width = extrema(loss_widths)
    d2_min = 0.01 * min_loss_width * min_loss_width
    d2_max = 100.0 * max_loss_width * max_loss_width
    nbins = max(64, ceil(Int, agg.bins_per_decade * log10(d2_max / d2_min)))
    W = _build_score_weight_matrix(scoring, loss_widths, nbins, d2_min, d2_max)
    W_d = ka_upload(Float32, backend, vec(W))
    log_d2_min   = S(log(d2_min))
    inv_log_spacing = S(nbins) / S(log(d2_max / d2_min))
    k1 = OnBackend(backend; workgroupsize = 1)(histogram_pm_score_kernel!)
    out = _chunked_argmax_eval(shared, T; cs, return_idxs) do scores_pmt, s, n
        k1(scores_pmt, shared.sorted_sq_flat, shared.pair_offset,
           shared.K_per_pair, shared.M_per_pair,
           W_d, Int32(nbins), log_d2_min, inv_log_spacing, Int32(T),
           Int32(s); ndrange = (n, M_max))
        KernelAbstractions.synchronize(backend)
    end
    free_device_buffers!(W_d)
    return out
end
