# =============================================================================
# SIMFIT++ — less biased noise-scale (loss-width) estimation for threshold-
# robust RANSAC (Edstedt, "Less Biased Noise Scale Estimation for Threshold-
# Robust RANSAC", CVPRW 2025). A step-for-step port of the reference
# implementation, github.com/Parskatt/simfitpp (src/simfitpp/simfitpp.py,
# scale_estimators.py, noise_models.py), kept in the reference's structure:
#
#   noise model        χ²_{d_g} on the squared residual (reference: χ²₁ for
#                      the scalar Sampson residual of F / E)
#   UnbiasedMed        `unbiased_median_variance`: the median of the squared
#                      residuals BELOW the current loss width, de-biased for
#                      that truncation by a five-step fixed-point iteration on
#                      the quantile the truncated median corresponds to, then
#                      divided by the χ² quantile
#   SIMFITPP.__call__  `simfitpp_loss_width`: `max_iter` rounds of a random
#                      train/validation split; the geometry is estimated on the
#                      train split at the current loss width (fitting bias
#                      removed by scoring held-out residuals); the validation
#                      residuals give the scale, hence the new loss width
#                      loss_width = sqrt(ppf(alpha)·σ²); out-of-bounds proposals are
#                      skipped; accepted proposals enter a running mean; the
#                      loop stops on `ftol` or on a repeated (3-decimal)
#                      value; the final estimate is bounds-checked and, in the
#                      multi-pair setting, filtered in log space with the
#                      `num_previous_est` earlier estimates.
#
# The geometry estimator is a callback pair (`fit`, `residuals_sq`) so the
# same loop runs on a full RANSAC (the reference: PoseLib at the loss width
# with refinement) or on a hypothesis pool (`simfitpp_select`: the MSAC-best
# pool hypothesis scored on the train split — the paper's fixed-pool protocol,
# no refinement).
# =============================================================================

"""
    SimFitPP(; alpha = 0.99, max_iter = 4, ftol = 0.01, train_fraction = 0.5,
               min_loss_width = 0.25, max_loss_width = 8.0, dg = 1)

SIMFIT++ configuration (the reference's constructor defaults; the reference
calls the loss width its threshold). `alpha` is the inlier coverage the loss
width targets (loss_width² = ppf_{χ²_dg}(alpha)·σ²); `min_loss_width` and
`max_loss_width` bound an acceptable loss width in the residual unit of the
residuals handed to the estimator (pixels in the reference); `dg` is the
residual codimension (the χ² degrees of freedom).
"""
Base.@kwdef struct SimFitPP
    alpha::Float64          = 0.99
    max_iter::Int           = 4
    ftol::Float64           = 0.01
    train_fraction::Float64 = 0.5
    min_loss_width::Float64 = 0.25
    max_loss_width::Float64 = 8.0
    dg::Int                 = 1
end

"""
    with_residual_unit(cfg::SimFitPP, unit_scale) -> SimFitPP

The same configuration with the loss-width bounds expressed in another residual
unit. `min_loss_width`/`max_loss_width` are multiplied by `unit_scale` (e.g.
`residual_unit_scale(pair)` to carry the reference's pixel bounds into a
pool stored in the pool frame). The argument was a bare `s`, the letter this package
otherwise spends on a noise scale.
"""
with_residual_unit(cfg::SimFitPP, unit_scale::Real) =
    SimFitPP(cfg.alpha, cfg.max_iter, cfg.ftol, cfg.train_fraction,
             cfg.min_loss_width * unit_scale, cfg.max_loss_width * unit_scale, cfg.dg)

"""
    unbiased_median_variance(residuals_sq, sq_loss_width; dg = 1) -> σ²

The reference's `UnbiasedMed`: the noise VARIANCE from the squared residuals
below `sq_loss_width`, corrected for the truncation. With `med` the median of
the kept residuals and `ratio = sq_loss_width / med`, the quantile `q` that the truncated median
corresponds to under χ²_dg solves `q = F(ratio · F⁻¹(q)) / 2`, iterated five
times from `q = 1/2`; the estimate is `med / F⁻¹(q)`. For `dg == 1` a ratio
below 5 has no useful fixed point and the naive `med / F⁻¹(1/2)` is returned
(the reference's `min_ratio`). Returns `NaN` when no residual lies below the
loss width.

**It returns σ², not σ** — which is why the name no longer says "scale". It was
`unbiased_median_scale`, sitting in the same package as a partition-scale
verb that returned a LINEAR σ̂. Two exported `*_scale` names, one squared and one
not, is the ambiguity VisualGeometryToolkitCore's `s2` convention exists to remove.
"""
function unbiased_median_variance(residuals_sq::AbstractVector{<:Real}, sq_loss_width::Real; dg::Integer = 1)
    dist = Chisq(dg)
    min_ratio = dg == 1 ? 5.0 : 0.0
    kept = Float64[r for r in residuals_sq if r < sq_loss_width]
    isempty(kept) && return NaN
    med = median(kept)
    ratio = sq_loss_width / med
    q = 0.5
    σ²0 = med / quantile(dist, q)
    ratio < min_ratio && return σ²0
    for _ in 1:5
        q = cdf(dist, ratio * quantile(dist, q)) / 2
    end
    return med / quantile(dist, q)
end

"""
    simfitpp_loss_width(fit, residuals_sq, n, initial_loss_width, cfg = SimFitPP();
                        num_previous_est = 0, rng = Random.default_rng())
        -> (loss_width, success::Bool)

The reference's `SIMFITPP.__call__` over `n` correspondences.
`fit(loss_width, train)` estimates the geometry from the correspondences indexed
by `train` at `loss_width` and returns a model handle (or `nothing` on
failure); `residuals_sq(model, val)` returns the squared residuals of that model
on the held-out indices `val` (or `nothing`). `initial_loss_width` seeds the
iteration and is returned unchanged when the estimate fails
(`success == false`); with `num_previous_est = p > 0` earlier estimates behind
`initial_loss_width`, the successful estimate is filtered as
`exp(log(loss_width)/(p+1) + p·log(initial_loss_width)/(p+1))`.
"""
function simfitpp_loss_width(fit, residuals_sq, n::Integer, initial_loss_width::Real,
                            cfg::SimFitPP = SimFitPP();
                            num_previous_est::Integer = 0,
                            rng::AbstractRNG = default_rng())
    inner_num_success = 0
    previous_loss_width = Float64(initial_loss_width)
    seen = Set{Float64}()
    coverage_chisq_quantile = quantile(Chisq(cfg.dg), cfg.alpha)
    loss_width = NaN
    for _ in 1:cfg.max_iter
        perm = randperm(rng, n)
        train_end = floor(Int, n * cfg.train_fraction)      # the reference's int()
        train = @view perm[1:train_end]
        val   = @view perm[train_end + 1:end]
        model = fit(previous_loss_width, train)
        r2 = model === nothing ? nothing : residuals_sq(model, val)
        σ_sq = r2 === nothing ? NaN : unbiased_median_variance(r2, previous_loss_width * previous_loss_width; dg = cfg.dg)
        proposed_loss_width = sqrt(coverage_chisq_quantile * σ_sq)
        (proposed_loss_width >= cfg.min_loss_width && proposed_loss_width <= cfg.max_loss_width) || continue    # NaN fails both
        loss_width = proposed_loss_width / (inner_num_success + 1) +
             previous_loss_width * inner_num_success / (inner_num_success + 1)
        abs(loss_width - previous_loss_width) < cfg.ftol && break
        key = round(loss_width; digits = 3)                  # the reference's 3-decimal key
        key in seen && break
        push!(seen, key)
        inner_num_success += 1
        previous_loss_width = loss_width
    end
    final = loss_width
    out_of_bounds = isnan(final) || final <= cfg.min_loss_width || final >= cfg.max_loss_width
    success = !out_of_bounds
    if !success
        final = Float64(initial_loss_width)
    else
        p = num_previous_est
        final = exp(log(final) / (p + 1) + p / (p + 1) * log(Float64(initial_loss_width)))
    end
    return final, success
end

# -----------------------------------------------------------------------------
# Hypothesis-pool adapter
# -----------------------------------------------------------------------------

# The MSAC-best pool model over the rows `rows` — the pool argmax
# (`_best_model`) of MSAC's quality summed over those rows (`_loss_width_column_score`),
# both scoring_eval_common.jl's: 0 for an empty pool. Squared in the matrix's own
# precision, as every pool scorer squares (it squared in Float64 until
# 2026-09-30).
_msac_best_over(distances::AbstractMatrix, rows, loss_width::Real) =
    _best_model(j -> _loss_width_column_score(MsacScoring(), distances, j, rows, Float64(loss_width)),
                size(distances, 2))

"""
    residual_unit_scale(pair::PoolPair) -> Float64

Pool-frame residual units per pixel for the pair: the mean isotropic scale of
the two pool-frame maps (pixel → `NormalizedImageFrame`). A CONVENTION, not an
exact conversion — the pool frame scales the two images independently and the
Sampson denominator mixes both images' gradients, so no single scalar relates
the two frames per point. Use it only for soft quantities (an initial
loss-width guess from a pool-frame-unit grid); anything the bounds gate must
run on [`pixel_residuals`](@ref).
"""
function residual_unit_scale(pair::PoolPair)
    return (pair.source_map[1, 1] + pair.destination_map[1, 1]) / 2
end

"""
    pixel_residuals(pair::PoolPair) -> Matrix{Float32}

The image-plane Sampson residuals of the pair's stored models in the PIXEL
frame, `num_points × num_models`. Pixels are the frame the SIMFIT++ reference works in, so
its pixel bounds apply to them without any scalar approximation (the pool frame
scales the two images independently, cf. [`residual_unit_scale`](@ref)). One
n×M scan of the light record: the correspondences come back to pixels through
the stored maps (`pixel_csponds`), each model by composition with the
pool normalization's two sides, `inv(D) ∘ M ∘ S` (a pool `F` → `T₂ᵀ F T₁`, a
pool `H` → `T₂⁻¹ H T₁`). A model type with neither law (a bare matrix) has no
`sampson_distance` and is a `MethodError`.

It never reads the pool's stored residual matrix, so it does not depend on that
matrix's residual kind. The stored residuals of an essential pool are TANGENT
Sampson distances (the pool generator's `residual_kind = :tangent_sampson`),
those of an F or H pool image-plane Sampson (`:sampson`); this exit is
image-plane Sampson for every pool, the quantity SIMFIT++'s pixel bounds are
stated in. (It used to return the stored matrix itself when both maps were the
identity — the image-plane quantity for an F/H pool, but the tangent one for an
E pool: two routes to two quantities under one name.)
"""
function pixel_residuals(pair::PoolPair)
    n, M = num_points(pair), num_models(pair)
    (n == 0 || M == 0) && return Matrix{Float32}(undef, n, M)
    N = pool_normalization(pair)
    S, D = source_map(N), destination_map(N)
    iD = inv(D)
    px = pixel_csponds(pair)
    u = [SVector{2,Float64}(x) for x in sources(px)]
    v = [SVector{2,Float64}(x) for x in targets(px)]
    out = Matrix{Float32}(undef, n, M)
    @inbounds for (j, m) in enumerate(pair.models)
        mp = iD ∘ m ∘ S
        for i in 1:n
            out[i, j] = Float32(sampson_distance(mp, u[i], v[i]))
        end
    end
    return out
end

"""
    simfitpp_select(residuals::AbstractMatrix, initial_loss_width, cfg = SimFitPP();
                    num_previous_est = 0, rng = default_rng())
        -> (best_idx, loss_width, success)
    simfitpp_select(pair::PoolPair, initial_loss_width_px, cfg = SimFitPP(); ...)

SIMFIT++ on a fixed hypothesis pool: `residuals` is the pool's n×M residual
matrix (points × models) in the unit `initial_loss_width` and the bounds of
`cfg` are expressed in. The geometry "estimator" is the MSAC-best pool
hypothesis on the train split at the current loss width, scored by its
validation-split residuals; the returned index is the MSAC argmax over ALL
points at the final loss width.

The pair method is PIXELS in, pixels out: it runs on
[`pixel_residuals`](@ref)`(pair)`, the pool models' residuals in pixels, so
`initial_loss_width_px`, the bounds of `cfg` (the reference's pixel defaults,
0.25–8 px) and the returned loss width are all pixels. To work in the pool
frame's unit instead, call the matrix method on the pair's stored residual
matrix with bounds in that unit ([`with_residual_unit`](@ref)).
"""
function simfitpp_select(D::AbstractMatrix, initial_loss_width::Real, cfg::SimFitPP = SimFitPP();
                         num_previous_est::Integer = 0, rng::AbstractRNG = default_rng())
    n = size(D, 1)
    (n == 0 || size(D, 2) == 0) && return (0, Float64(initial_loss_width), false)
    fit(loss_width, train) = (j = _msac_best_over(D, train, loss_width); j == 0 ? nothing : j)
    residuals_sq(j, val) = square_residuals!(Vector{Float64}(undef, length(val)),
                                             view(D, val, j), length(val))
    loss_width, success = simfitpp_loss_width(fit, residuals_sq, n, initial_loss_width, cfg;
                                     num_previous_est, rng)
    return (_msac_best_over(D, 1:n, loss_width), loss_width, success)
end
simfitpp_select(pair::PoolPair, initial_loss_width_px::Real, cfg::SimFitPP = SimFitPP();
                num_previous_est::Integer = 0, rng::AbstractRNG = default_rng()) =
    simfitpp_select(pixel_residuals(pair), initial_loss_width_px, cfg; num_previous_est, rng)
