#!/usr/bin/env julia
# =============================================================================
# scoring_eval_common.jl — the summed quality over a row subset, and the
# per-pair staging precompute
#
# Contains:
#   - `_loss_width_sum!`, Σ quality over some rows (SIMFIT++'s train split). A
#     family's `score!` AT a loss width (`MlesacScoring(dg; threshold = w)`, …,
#     or `with_width(F, LossWidth(w))`) is VisualGeometryToolkitCore's one
#     threshold loop: the five families are `AbstractThresholdScoring`s, as
#     RANSAC and MSAC are;
#   - the per-pair staging precompute (PrecomputedPair)
# Every function here takes the light pool record (VisualGeometryToolkitCore's
# `PoolPair`) together with its residual matrix `R` (num_points × num_models,
# Float32, NormalizedImageFrame units): the record no longer carries the
# residuals, and no scorer opens a file. The record's residual-matrix contract
# and the pool argmax are VisualGeometryToolkitCore's (`Pools.check_residual_matrix`,
# `Pools.best_model_index`).
# The JLD2/sidecar HARNESS (SceneEval + scene IO, coarse-range sidecar
# lookups) moved to the evaluation engine (src/evaluation/) with the rest
# of the eval subsystem — 2026-07-18 (now VisualGeometryToolkitEval)
# =============================================================================

# (loss_width_sweep_ka.jl — `LossWidthScoring`, `residual_quality` — is included
# by the module before this file.)

# =============================================================================
# A loss-width family's score at a loss width
#
# VisualGeometryToolkitCore's threshold `score!`: ONE CPU loop over the SAME
# per-residual quantity the grids evaluate, Σᵢ residual_quality(scoring, dᵢ²,
# loss_width) — the scalar of the KernelAbstractions Exact/LogBinned kernels
# (loss_width_sweep_ka.jl), fed exactly as staging (`precompute_pair`) feeds
# them: d² squared in the residuals' precision, a non-finite square an outlier
# (quality 0), the model's minimal-sample points exact inliers (d² = 0). It is
# the exact scorer on one pair, kept on the CPU: staging one pair (a sort of
# every column) plus a kernel launch cost 7–39× this loop for RANSAC/MSAC at
# n = 300 (benchmark 2026-09-22, KA.CPU() on 32 threads against this
# single-threaded loop), and callers already thread over pairs. This package's
# copy of that loop for its own five families is deleted (2026-11-07).
#
# CONTRACTS, one for the whole stack:
#   * the loss width is supplied linear, never squared, for every family;
#   * the no-model sentinel of a pool argmax is 0, never a valid index
#     (VisualGeometryToolkitCore's `Pools.best_model_index`: the first strict best,
#     a non-finite score never wins);
#   * RANSAC's inlier test is strict, d² < loss_width² on the pool-precision
#     squares — the grids' test.
# =============================================================================

# Σ residual_quality(scoring, dᵢ², loss_width) over the indices `rows` of the
# residual vector `d`, d² squared in `d`'s own precision (the staging contract),
# a non-finite d² an outlier (quality 0, VisualGeometryToolkitCore's rule: a finite d
# whose square overflows too); each point's quality into `scores` (or not, for
# `nothing`) — the arithmetic of VisualGeometryToolkitCore's threshold `score!`
# on a row subset, for SIMFIT++'s MSAC-best over a train split (simfitpp.jl).
@inline function _quality(scoring, d, loss_width::Float64)
    d2 = Float64(d * d)
    return isfinite(d2) ? residual_quality(scoring, d2, loss_width) : 0.0
end
@inline function _loss_width_sum!(scores, scoring::LossWidthScoring, d::AbstractVector, rows,
                                  loss_width::Float64)
    s = 0.0
    @inbounds for i in rows
        q = _quality(scoring, d[i], loss_width)
        scores === nothing || (scores[i] = q)
        s += q
    end
    return s
end

# log(1 + eˣ), stable over the whole range, is LogExpFunctions' `log1pexp` (a
# dependency of SpecialFunctions already; the local `_softplus` restated it).
# GaU's argument never exceeds ½; `mixture_nll` (VisualGeometryToolkitCore's) needs
# the whole range.
const _SOFTPLUS_HALF = log1pexp(0.5)      # MLESAC's quality normaliser (loss_width_sweep_ka.jl)

# (`pool_model_score` and the 4-argument `best_pool_model(scoring, pair, R,
#  loss_width)` — a width-free family evaluated at a loss width — are deleted:
#  a family AT its loss width has `score!` (above), so VisualGeometryToolkitCore's
#  `best_pool_model(with_width(scoring, LossWidth(w)), pair, R)` is the one pool
#  selection, through the same arithmetic. The local pool argmax `_best_model`
#  and shape check `_check_residuals` are VisualGeometryToolkitCore's
#  `Pools.best_model_index` and `Pools.check_residual_matrix`.)

# (`evaluate_at` — the single-loss-width, threaded counterpart of
#  `grid_evaluate` — is deleted: no caller in RSDR, RVG, VisualGeometryToolkitEval or LODR. A
#  one-column `grid_evaluate` answers the same question.)

# (`grid_evaluate` — the CPU grid over a Symbol-named method — is deleted with
#  everything only it used: the 500-LINEAR-bin histogram (`bin_residuals_stereo`,
#  `_weight_matrix_*`, `_grid_best_errors!`, `_fill_grid_hist!`), the exact host
#  loop for the loss costs (`_fill_grid_exact!`), and the container accessors
#  `_resid_mat`/`_npts`. Every grid is a KernelAbstractions evaluator now, CPU()
#  or GPU, all methods of the one verb `grid_evaluate`, dispatched on the
#  scoring: the loss-width families (Exact or LogBinned), the marginal, the
#  scale marginals and the mixture marginal.)

# (`_maa` — mean average accuracy as the area under the sorted accuracy curve —
#  is deleted: a third implementation of VisualGeometryToolkitCore's
#  `Synthetic.mean_average_accuracy(errors; max_error)`, the same area as a mean of
#  max(0, 1 − e/max_error). It agreed on finite errors to 8e-16 but not at the edges:
#  it returned 0.0 for no errors (VGC: NaN) and let a NaN error poison the sum
#  (VGC: a failure, scoring 0). Every caller now uses VGC's.)

# (`fit_ig_mle_prior` is deleted: no caller anywhere, and a second copy of
#  `fit_prior(::InverseGamma, ::MLE)` — the same `fit_mle(Gamma, 1 ./ σ²)` —
#  with different edge handling (a `(3, 2μ)` fallback where `fit_prior` returns
#  `nothing`). Of the four scripts its docstring named, only one still exists,
#  LocalOptimizationDoneRight's `_research/exploratory/test_sigma2_synthetic.jl`,
#  which mentions it in a comment and does not call it.)

# -----------------------------------------------------------------------------
# Per-pair staging precompute (`stage_ig_shared_from_pairs`, ig_sweep_ka.jl).
# -----------------------------------------------------------------------------

"""
    PrecomputedPair

Sorted squared residuals per model of one pool pair — the staging unit of
[`stage_ig_shared_from_pairs`](@ref): sorting once here makes every device
sweep an O(n) linear scan over prefix sums.
"""
struct PrecomputedPair
    sorted_sq::Matrix{Float32}   # (n, M) — sorted r² per model column, in the residuals' Float32
    errors_imc::Vector{Float64}  # (M,) — selection pose error `pool_imc` [deg] per model (baked at precompute)
    n::Int                       # number of points
end

"""
    precompute_pair(pair::PoolPair, R) -> PrecomputedPair

Per-pair staging precompute of `pair` and its residual matrix `R`
(`num_points × num_models`, Float32, `NormalizedImageFrame` units). Each model
column is squared in Float32, its minimal-sample points zeroed (exact inliers; an
out-of-range index THROWS, as in every other scorer) and sorted; the selection
error `pool_imc` is kept per model.
"""
function precompute_pair(pair::PoolPair, R::AbstractMatrix{Float32})
    check_residual_matrix(pair, R)
    n = num_points(pair); M = num_models(pair)
    sorted_sq = Matrix{Float32}(undef, n, M)
    for j in 1:M
        col = view(sorted_sq, 1:n, j)
        square_distances!(col, view(R, 1:n, j), n)   # in the residuals' own precision
        zero_sample!(col, view(pair.sample_indices, :, j))
        sort!(col)
    end
    # Bake the selection metric `pool_imc` per model (rotation-only for a
    # homography pool, max(Δrot, Δtrans) otherwise). The GPU path stages only
    # this one error array (no errors_t on device), so imc must be combined
    # here — this single repurpose makes every GPU path (IG + loss width, both staged
    # from PrecomputedPair) report on imc.
    PrecomputedPair(sorted_sq, [pool_imc(pair, j) for j in 1:M], n)
end
