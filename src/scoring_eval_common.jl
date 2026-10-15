#!/usr/bin/env julia
# =============================================================================
# scoring_eval_common.jl — per-pair model selection + shared eval plumbing
#
# Contains:
#   - `best_pool_model(scoring, pair, R, loss_width)`: the index of a pair's
#     best stored model under a width-free loss-width family at a loss width
#     (the 3-argument form — a scorer AT its width — is VisualGeometryCore's,
#     generic over `score!`)
#   - the per-pair staging precompute (PrecomputedPair)
# Every function here takes the light pool record (VisualGeometryCore's
# `PoolPair`) together with its residual matrix `R` (num_points × num_models,
# Float32, NormalizedImageFrame units): the record no longer carries the
# residuals, and no scorer opens a file.
# The JLD2/sidecar HARNESS (SceneEval + scene IO, coarse-range sidecar
# lookups) moved to the evaluation engine (src/evaluation/) with the rest
# of the eval subsystem — 2026-07-18 (now VisualGeometryEval)
# =============================================================================

# StudentTUniformScoring, ScoringWorkspace, score_residuals!, codimension are defined
# in this package. The concrete problem types
# (HomographyProblem/FundMatProblem/EssentialProblem) live in the consumer and
# are given as the leading `problem_type` argument where one is needed.
# (loss_width_sweep_ka.jl — `LossWidthScoring`, `residual_quality` — is included
# by the module before this file.)

# =============================================================================
# Per-pair model selection — the index of a pair's best stored model
#
# ONE CPU loop per scoring kind, over the SAME per-residual quantity the grids
# evaluate:
#   * a loss-width family (`LossWidthScoring`): Σᵢ residual_quality(scoring, dᵢ², loss_width) — the
#     scalar of the KernelAbstractions Exact/LogBinned kernels
#     (loss_width_sweep_ka.jl), fed exactly as staging (`precompute_pair`) feeds them:
#     d² squared in the pool's precision, a non-finite residual an outlier
#     (skipped), the model's minimal-sample points exact inliers (d² = 0);
#   * the marginal (`StudentTUniformScoring`): the sort-and-sweep `score_residuals!`.
# It IS the exact scorer on one pair, kept on the CPU: staging one pair (a sort
# of every column) plus a kernel launch cost 7–39× this loop for RANSAC/MSAC and
# 2.2× for the marginal at n = 300 (benchmark 2026-09-22, KA.CPU() on 32
# threads against this single-threaded loop), and callers already thread over
# pairs. It replaced score_ransac/score_msac/score_gau/score_magsac, which each
# restated its family's formula, and score_marginal.
#
# CONTRACTS, one for the whole stack:
#   * the loss width is supplied linear, never squared, for every family —
#     `score_msac` took its square, the only one that did;
#   * the NO-MODEL SENTINEL IS 0, never a valid index, so indexing with it
#     throws instead of silently selecting model 1 (the old per-pair functions
#     returned 1). A loss-width family returns 0 only for a pair with no models; the
#     marginal also when no model scores finitely;
#   * ties go to the lowest index;
#   * RANSAC's inlier test is strict, d² < loss_width² on the pool-precision squares —
#     the grids' test.
# =============================================================================

"""
    pool_model_score(scoring::LossWidthScoring, pair::PoolPair, R, j, loss_width[, q0]) -> Float64

Score of the stored pool model `j` of `pair` under a loss-width family (`MsacScoring`,
`RansacScoring`, `MagsacScoring`, the loss scorings) at `loss_width`.
It is `Σᵢ residual_quality(scoring, dᵢ², loss_width)` over the model's column of the
pair's residual matrix `R` (`num_points(pair) × num_models(pair)`, Float32, in
`NormalizedImageFrame` units — the pool frame's, as is `loss_width`). The ONE
per-model loss-width score: [`best_pool_model`](@ref) is its argmax, the Exact grid
([`grid_evaluate`](@ref) with `Exact()`) computes the same value, and a running-best
selector over a pool calls it per `(pair, j)` rather than re-summing the quality.

The stack's minimal-sample rule, the same in every scorer: the model's
minimal-sample points (`pair.sample_indices[:, j]`) are fit exactly by
construction and carry no information, so they enter at ZERO residual whatever
was stored for them — here each contributes the constant `q0 =
residual_quality(scoring, 0, loss_width)`, identical for every model of the pool, so the
ranking is that of the non-sample points alone. (The marginal scorers zero the
same entries and subtract the model's `m` dof in the partition evidence,
`ν = dg·(k − m)`.) A non-finite stored d is an outlier (quality 0); finiteness
is tested before squaring, so a finite d whose square overflows also scores 0.
An invalid sample index throws. `q0` is an optional precomputed cache.
"""
@inline function pool_model_score(scoring::LossWidthScoring, pair::PoolPair, R::AbstractMatrix{Float32},
                                   j::Int, loss_width::Float64,
                                   q0::Float64 = residual_quality(scoring, 0.0, loss_width))
    s = _loss_width_column_score(scoring, R, j, 1:num_points(pair), loss_width)
    for i in view(pair.sample_indices, :, j)
        d = R[i, j]
        s += q0 - (isfinite(d) ? residual_quality(scoring, Float64(d * d), loss_width) : 0.0)
    end
    return s
end
pool_model_score(scoring::LossWidthScoring, pair::PoolPair, R::AbstractMatrix{Float32}, j::Integer,
                 loss_width::Real) =
    pool_model_score(scoring, pair, R, Int(j), Float64(loss_width))

# One model column's loss-width-family score over the rows `rows` of a residual matrix:
# Σ residual_quality(scoring, dᵢ², loss_width), d² squared in the matrix's own precision
# (the staging contract), a non-finite d skipped as an outlier. The one such
# sum: every pool model's score above, and SIMFIT++'s MSAC-best over a train
# split (simfitpp.jl).
@inline function _loss_width_column_score(scoring::LossWidthScoring, D::AbstractMatrix, j::Int, rows,
                                   loss_width::Float64)
    s = 0.0
    @inbounds for i in rows
        d = D[i, j]
        isfinite(d) && (s += residual_quality(scoring, Float64(d * d), loss_width))
    end
    return s
end

# The index of the best of models 1:M under `score(j)`: the FIRST strictly best,
# `0` when none scores above −Inf (the no-model sentinel). A NaN score never
# wins. The one pool argmax — every `best_pool_model` and SIMFIT++'s.
@inline function _best_model(score::F, M::Integer) where {F}
    best_j = 0; best = -Inf
    for j in 1:M
        s = score(j)
        s > best && (best = s; best_j = j)
    end
    return best_j
end

# A pair's residual matrix is `num_points × num_models`: its rows index the
# pair's correspondences and its columns the stored models. Checked at every
# scorer entry, so a matrix of another pair is refused rather than read.
function _check_residuals(pair::PoolPair, R::AbstractMatrix)
    size(R) == (num_points(pair), num_models(pair)) || throw(DimensionMismatch(
        "residual matrix of size $(size(R)) for a pool pair of $(num_points(pair)) points " *
        "and $(num_models(pair)) models (expected num_points × num_models)"))
    return nothing
end

"""
    best_pool_model(scoring::LossWidthScoring, pair::PoolPair, R, loss_width::Real) -> Int

Index of the best-scoring stored pool model of `pair` under the width-free
loss-width family `scoring` at `loss_width`, scored on its residual matrix `R`
(`num_points(pair) × num_models(pair)`, Float32, `NormalizedImageFrame` units —
`loss_width` is in that unit too): per model `Σᵢ residual_quality(scoring, dᵢ²,
loss_width)` ([`pool_model_score`](@ref)), exactly the Exact grid's
([`grid_evaluate`](@ref)) per-model score on this one pair. `0` means no model
(see the sentinel contract above); a matrix of another shape is a
`DimensionMismatch`. A scorer AT its width (`MsacScoring(dg; threshold = w)`,
a `StudentTUniformScoring`) goes through VisualGeometryCore's 3-argument
`best_pool_model(scoring, pair, R)` — the same arithmetic, through `score!`.
"""
function best_pool_model(scoring::LossWidthScoring, pair::PoolPair, R::AbstractMatrix{Float32},
                         loss_width::Real)
    _check_residuals(pair, R)
    width = Float64(loss_width); q0 = residual_quality(scoring, 0.0, width)
    return _best_model(j -> pool_model_score(scoring, pair, R, j, width, q0), num_models(pair))
end

# log(1 + eˣ), stable over the whole range, is LogExpFunctions' `log1pexp` (a
# dependency of SpecialFunctions already; the local `_softplus` restated it).
# GaU's argument never exceeds ½; `mixture_nll` (VisualGeometryCore's) needs
# the whole range.
const _SOFTPLUS_HALF = log1pexp(0.5)      # MLESAC's quality normaliser (loss_width_sweep_ka.jl)

# (`evaluate_at` — the single-loss-width, threaded counterpart of
#  `grid_evaluate` — is deleted: no caller in RSDR, RVG, VisualGeometryEval or LODR. A
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
#  is deleted: a third implementation of VisualGeometryCore's
#  `Synthetic.mean_average_accuracy(errors; T)`, the same area as a mean of
#  max(0, 1 − e/T). It agreed on finite errors to 8e-16 but not at the edges:
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
    _check_residuals(pair, R)
    n = num_points(pair); M = num_models(pair)
    sorted_sq = Matrix{Float32}(undef, n, M)
    for j in 1:M
        col = view(sorted_sq, 1:n, j)
        square_residuals!(col, view(R, 1:n, j), n)   # in the residuals' own precision
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
