# =============================================================================
# Loss-width-family scoring — KernelAbstractions, CPU() or a GPU backend.
#
# Every loss-width family (RANSAC, MSAC, MLESAC, MAGSAC++, and the Welsch /
# smooth-TLS / Geman-McClure losses) has ONE per-residual quality
# `residual_quality(scoring, d², loss_width)`, and two ALGORITHMS aggregate it per
# (pair, model, loss width), chosen explicitly:
#
#   Exact()               every residual through the quality — the reference
#                         (final evaluation, per-pair selection);
#                         `exact_pm_score_kernel!`.
#   LogBinned(per_decade) residuals histogrammed on a log-d² grid, the SAME
#                         quality evaluated at bin centres — the fast
#                         cross-validation path; `histogram_pm_score_kernel!`.
#
# Each is written ONCE and serves both backends; the model argmax and error
# gather are the shared `argmax_models_kernel!` / `_chunked_argmax_eval`
# (marg_sweep_ka.jl). Reads `IGSharedData.sorted_sq_flat` (per-model squared
# residuals, sorted ascending, in the pool's precision).
#
# Usage:
#   staged = stage_ig_shared_from_pairs(pairs, residuals; backend)
#   errs = grid_evaluate(loss_width_scoring(:magsac, codimension(PT)),
#                        staged, loss_widths)                       # Float32, LogBinned(200)
#   errs = grid_evaluate(Float64, loss_width_scoring(:msac, dg), staged,
#                        loss_widths; aggregation = Exact())

# (`_ka_softplus` lived here, byte-identical to `_softplus` in
# scoring_eval_common.jl. The `_ka_` prefix implied a device constraint there is
# none: `_build_score_weight_matrix` fills a host `Array{Float32}`, so the host
# definition serves. Deleted; this file uses LogExpFunctions' `log1pexp` and
# `_SOFTPLUS_HALF`.)

# =============================================================================
# Aggregation — how a family's per-residual quality is summed per model
# =============================================================================

"""
    Aggregation

How a loss-width family's per-residual quality is summed over a model's residuals:
[`Exact`](@ref) or [`LogBinned`](@ref). Both evaluate the SAME scalar quality;
only the aggregation differs.
"""
abstract type Aggregation end

"""
    Exact()

Every residual through the family's scalar quality — the reference. The
quality is evaluated in Float64 whatever the kernel precision `S` of
[`grid_evaluate`](@ref).
"""
struct Exact <: Aggregation end

"""
    LogBinned(bins_per_decade = 200)

The fast cross-validation aggregation: residuals histogrammed on a log-spaced d² grid.
The grid has `bins_per_decade` bins per decade, spanning 0.01·min_loss_width² to
100·max_loss_width², and the family's quality is evaluated at the bin centres. Out-of-range and non-finite
residuals score as outliers.
"""
struct LogBinned <: Aggregation
    bins_per_decade::Int
end
LogBinned() = LogBinned(200)

# --- Exact: per (pair, model) thread; every residual, every loss width ---
# The quality is evaluated in Float64 and accumulated in Float64. A non-finite
# residual is an outlier (quality 0, skipped). Residuals are sorted ascending,
# so for a family of COMPACT support (quality 0 from loss_width on: `_support_sq`) nothing
# past max_loss_width² can score and the loop stops there.
@kernel function exact_pm_score_kernel!(
        scores_out,                       # (chunk_size, M_max, T) Float64
        @Const(sorted_sq_flat),
        @Const(pair_offset),
        @Const(K_per_pair),
        @Const(M_per_pair),
        @Const(loss_widths),               # (T,) Float64
        T_val::Int32,
        p_offset::Int32,                  # chunk-local p_local + p_offset → p_global
        scoring,                          # the loss-width family (a singleton the kernel specialises on)
        support_sq::Float64)              # max_loss_width² for compact families, Inf otherwise
    p_local, m = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)
    M = Int(M_per_pair[p])
    if m <= M
        K = Int(K_per_pair[p]); Tn = Int(T_val)
        col_base = pair_offset[p] + (Int64(m) - Int64(1)) * Int64(K)
        @inbounds for t in 1:Tn
            scores_out[p_local, m, t] = 0.0
        end
        @inbounds for k in 1:K
            d2 = Float64(sorted_sq_flat[col_base + k])
            isfinite(d2) || continue
            d2 >= support_sq && break
            for t in 1:Tn
                scores_out[p_local, m, t] += residual_quality(scoring, d2, loss_widths[t])
            end
        end
    end
end

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
# This is the `LogBinned` aggregation of every loss-width family; `Exact` is
# `exact_pm_score_kernel!` above.

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
            # exactly that — as the Exact kernel and the per-pair
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

# --- Kernel 2: per (pair, loss width) thread --- argmax over m, gather error_r.
@kernel function argmax_models_kernel!(
        errors_out,                       # (chunk_size, T) Float64
        idxs_out,                         # (chunk_size, T) Int32  (0-size when unused)
        @Const(scores_pmt),               # (chunk_size, M_max, T) Float64
        @Const(M_per_pair),
        @Const(errors_imc_flat),
        @Const(errors_offset),
        p_offset::Int32)                  # chunk_local p_local + p_offset → p_global
    p_local, t = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)      # global pair index for staging-array reads
    # GPU backends pad ndrange to workgroup multiples: padded threads MUST exit
    # before touching memory (they read garbage M and corrupt errors/idxs rows).
    if p_local <= size(errors_out, 1) && t <= size(errors_out, 2) && p <= length(M_per_pair)
    M = Int(M_per_pair[p])
    best_score = -Inf
    best_m = Int32(1)
    @inbounds for m in Int32(1):Int32(M)
        s = scores_pmt[p_local, m, t]      # chunk-local read
        if s > best_score
            best_score = s
            best_m = m
        end
    end
    @inbounds errors_out[p_local, t] = errors_imc_flat[errors_offset[p] + best_m]
    if size(idxs_out, 1) > 0
        @inbounds idxs_out[p_local, t] = best_m
    end
    end   # ndrange padding guard
end

# -----------------------------------------------------------------------------
# Threshold-scoring families (D): singleton types replacing the `method::Symbol`
# + if/elseif kernel selection. `grid_evaluate` dispatches on these via
# `residual_quality` and `_support_sq` below; Symbol-based callers convert at the boundary with
# `loss_width_scoring(:ransac, dg)`.
# -----------------------------------------------------------------------------
# The loss-width-swept scorings, split by whether a mixture sits underneath (the two
# branches and the traits are declared in scoring.jl).
#
# `ThresholdFamily` used to be the supertype of all seven. It could not be:
# "takes a loss width" is true of both branches, so using it as the hierarchy forced
# GauScoring's members and StudentTUniformScoring into disconnected roots even though
# they posit the SAME model. The members that take a loss width are now the
# `LossWidthScoring` union, which is what the sweep entry points accept.
#
# `GauScoring` is likewise no longer a concrete type. The whole mixture branch
# is Gaussian-Uniform, so the name belongs to the family; the member it used to
# name is MLESAC, the marginal likelihood over labels at fixed scale. Its tag is
# `:mlesac` (the tag rule: the type's name without its family suffix).
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

"""
    MlesacScoring{DG}()

MLESAC (GaU) at codimension `DG`: the Gaussian-uniform log-density at a fixed
scale, normalised to 1 at `d = 0` (see [`residual_quality`](@ref)). Symbol name
`:mlesac`.
"""
struct MlesacScoring{DG}     <: GauScoring end

"""
    MagsacScoring{DG}()

MAGSAC++ at codimension `DG`: the σ-marginalised quality, zero from the cutoff
on; its loss width is that cutoff, `_MAGSAC_SCALE_MULTIPLE·σ_max` (see
[`residual_quality`](@ref)). Symbol name `:magsac`.
"""
struct MagsacScoring{DG}     <: GauScoring end

"""
    WelschCostScoring{DG}()

The Welsch cost as a loss-width score at codimension `DG`: `1 − ρ/w²` with
VisualGeometryToolkitCore's Welsch `ρ`, no mixture underneath. Symbol name `:welsch_cost`.
"""
struct WelschCostScoring{DG} <: LossCostScoring end

"""
    SmoothTLSScoring{DG}()

The smooth truncated-least-squares cost as a loss-width score at codimension
`DG`: `1 − ρ/w²` with VisualGeometryToolkitCore's smooth-TLS `ρ`. Symbol name
`:smooth_tls`.
"""
struct SmoothTLSScoring{DG}  <: LossCostScoring end

"""
    GemanScoring{DG}()

The Geman–McClure cost as a loss-width score at codimension `DG`: `1 − ρ/w²`
with VisualGeometryToolkitCore's Geman–McClure `ρ`. Symbol name `:geman`.
"""
struct GemanScoring{DG}      <: LossCostScoring end

"""
    LossWidthScoring{DG}

The loss-width-parameterised scorings of both branches at codimension `DG` — the
paper's "threshold-based family". A Union, not a supertype: "takes a loss width" cuts
across `GauScoring` and `LossCostScoring`, so it cannot be a supertype of both
without implying a mixture underneath the losses, which is exactly what
the retired `ThresholdFamily` got wrong.

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

codimension(::_OwnLossWidthScoring{DG}) where {DG} = DG

# The seven families and their Symbol names, stated ONCE. The constructors below
# and `_family_type` are both generated from this, so adding a family is one row
# rather than several transcriptions of the same list kept in step by hand.
const _SCORING_FAMILIES = ((:ransac, RansacScoring), (:msac, MsacScoring),
                           (:mlesac, MlesacScoring), (:magsac, MagsacScoring),
                           (:welsch_cost, WelschCostScoring), (:smooth_tls, SmoothTLSScoring),
                           (:geman, GemanScoring))
# The constructors of the families this package owns (RANSAC's and MSAC's are
# VisualGeometryToolkitCore's, with the same forms).
const _OWN_SCORING_FAMILIES = _SCORING_FAMILIES[3:end]

# Bare `T()` is codim 1 (see above); `T(problem)` takes it from the problem,
# exactly as `StudentTUniformScoring(problem, a)` does.
for (_, T) in _OWN_SCORING_FAMILIES
    # `nameof`: the table stores the TYPE, and a bare `RansacScoring` as a value
    # is the UnionAll `RansacScoring{DG} where DG`. That is right on the RHS but
    # is not a legal function NAME, so the left-hand side needs the symbol.
    Tn = nameof(T)
    @eval begin
        $Tn() = $Tn{1}()
        $Tn(dg::Integer) = $Tn{Int(dg)}()
        $Tn(problem::AbstractConstraintProblem) = $Tn{codimension(problem)}()
    end
end

# Where these sit in the elimination table (scoring.jl): RANSAC and MSAC profile
# the labels, MLESAC and MAGSAC++ marginalise them, and every one of the seven
# runs at a supplied scale — that is what having a loss width MEANS. RANSAC is MSAC with
# the inlier RSS term dropped, keeping only the count, which is why it is the one
# baseline measurably worse.

# The cost branch scores by a VisualGeometryToolkitCore shape. All three carry a
# 95%-efficiency constant in VGC (`efficiency_constant(shape, codim)`), so each
# has its own loss width `scale_multiple·ŝ`. `SmoothTLSShape` is not an `AbstractMEstimator` (a
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

# The Symbol vocabulary, generated from `_SCORING_FAMILIES`. `_family_type` maps
# to the TYPE, not to an instance, which is what lets the codimension overload
# construct directly: `typeof(loss_width_scoring(Val(s))).name.wrapper(dg)` used to
# build a throwaway instance and reach into `DataType.name.wrapper` —
# undocumented internals — to recover a name the table already spells out.
# (The inverse direction, `family_symbol`, is gone: nothing read it.)
for (sym, T) in _SCORING_FAMILIES
    @eval _family_type(::Val{$(QuoteNode(sym))}) = $T
end
_family_type(::Val{X}) where {X} =
    error("loss_width_scoring: unknown method :$X " *
          "(expected :ransac/:msac/:mlesac/:magsac/:welsch_cost/:smooth_tls/:geman)")

"""
    loss_width_scoring(method::Symbol[, dg | problem]) -> LossWidthScoring

The `LossWidthScoring` member a Symbol names — `:ransac`, `:msac`, `:mlesac`,
`:magsac`, `:welsch_cost`, `:smooth_tls`, `:geman` (the tag rule: the type's
name without `Scoring`, snake_cased) — at codimension `dg`, or at
`codimension(problem)`. Bare, it is codimension 1.

Supply the codimension whenever the problem has one other than 1: the Symbol
vocabulary must not be a back door that silently re-defaults `DG`.

It returns a member of the `LossWidthScoring` union, and is named for it.
"""
loss_width_scoring(v::Val) = _family_type(v)()
loss_width_scoring(s::Symbol) = loss_width_scoring(Val(s))
loss_width_scoring(s::Symbol, dg::Integer) = _family_type(Val(s))(dg)
loss_width_scoring(s::Symbol, problem::AbstractConstraintProblem) =
    loss_width_scoring(s, codimension(problem))

# =============================================================================
# The per-residual QUALITY — one scalar per family, of the SQUARED residual d²
# and the loss width. Both aggregations call it: Exact at every residual,
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
# nothing past max_loss_width² scores (the Exact kernel stops there). MLESAC and the
# Welsch / Geman-McClure costs reach 0 only asymptotically.
_support_sq(::Union{RansacScoring, MsacScoring, MagsacScoring, SmoothTLSScoring},
            max_loss_width::Float64) = max_loss_width * max_loss_width
_support_sq(::LossWidthScoring, ::Float64) = Inf

"""
    _build_score_weight_matrix(scoring, loss_widths, N_bins, d2_min, d2_max)
        -> Matrix{Float32}  size (N_bins, T)

Precompute `W[bin, t] = residual_quality(scoring, d²_bin, loss_width_t)` on the host at the bin
centres of a log-spaced grid. The centres are `d²_bin = d²_min · (d²_max/d²_min)^((bin-0.5)/N_bins)`,
and `W` holds the SAME scalar quality the Exact kernel evaluates per residual. Returned
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

`aggregation` chooses the algorithm: [`LogBinned`](@ref) (default, 200 bins
per decade; the fast cross-validation path) or [`Exact`](@ref) (every residual;
the reference). Both evaluate the family's one per-residual quality. `S` is the
LogBinned kernel's bin-index precision (Float32, the default, is the production
setting); Exact evaluates in Float64. Scores accumulate in Float64 either way.

`scoring` is a `LossWidthScoring` member — build it with [`loss_width_scoring`](@ref),
supplying the problem's codimension.
"""
function grid_evaluate(::Type{S}, scoring::LossWidthScoring, staged::IGSharedData,
                       loss_widths::AbstractVector{<:Real};
                       aggregation::Aggregation = LogBinned(),
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

function _grid_evaluate_loss_width(::Exact, ::Type{S}, scoring, shared, loss_widths::Vector{Float64},
                            cs::Int, return_idxs::Bool) where {S}
    backend = shared.backend
    T = length(loss_widths); M_max = Int(shared.M_max)
    loss_widths_d = _ka_upload(backend, Float64, loss_widths)
    support = _support_sq(scoring, maximum(loss_widths))
    k1 = _ka_kernel(exact_pm_score_kernel!, backend, 2)
    out = _chunked_argmax_eval(shared, T; cs, return_idxs) do scores_pmt, s, n
        k1(scores_pmt, shared.sorted_sq_flat, shared.pair_offset,
           shared.K_per_pair, shared.M_per_pair, loss_widths_d, Int32(T), Int32(s),
           scoring, support; ndrange = (n, M_max))
        KernelAbstractions.synchronize(backend)
    end
    _free_local!(loss_widths_d)
    return out
end

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
    W_d = _ka_upload(backend, Float32, vec(W))
    log_d2_min   = S(log(d2_min))
    inv_log_spacing = S(nbins) / S(log(d2_max / d2_min))
    k1 = _ka_kernel(histogram_pm_score_kernel!, backend, 2)
    out = _chunked_argmax_eval(shared, T; cs, return_idxs) do scores_pmt, s, n
        k1(scores_pmt, shared.sorted_sq_flat, shared.pair_offset,
           shared.K_per_pair, shared.M_per_pair,
           W_d, Int32(nbins), log_d2_min, inv_log_spacing, Int32(T),
           Int32(s); ndrange = (n, M_max))
        KernelAbstractions.synchronize(backend)
    end
    _free_local!(W_d)
    return out
end
