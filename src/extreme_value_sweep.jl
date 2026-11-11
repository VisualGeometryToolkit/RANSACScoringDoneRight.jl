# =============================================================================
# extreme_value_sweep.jl — the sort-and-sweep of StudentTUniformScoring with an
# outlier model that estimates the support (extreme_value.jl). The sort is the
# sweep's own (`_sort_perm!`: the Θ(n) radix on the exact-size path), and the
# per-partition score is `_table_partition_score_cost`, the one formula every sweep
# shares (the inlier term, the scale prior); the label prior is the
# workspace's `prior_table`. Only the outlier term differs per model:
#
#   ExtremeValueHalfwidth                  the complement of prefix k is the
#       suffix z[k+1:n] of the sorted distances and its upper tail a suffix of
#       that: one O(log n) threshold and O(1) moments from two prefix sums
#   GeneralizedParetoOutliers(:prefix)     Θ(n²): the density changes with
#       every prefix's fit and Σ log p does not decompose
#   GeneralizedParetoOutliers(:hypothesis) Θ(n): one fit to the upper tail of
#       all distances, a fixed per-point cost, a suffix sum
#   OutlierTerm                            the caller's cost per prefix
#
# Allocation-free: every buffer is the workspace's.
# =============================================================================

# The sorted distances and their suffix sums in the workspace's tail buffers
# (z[1:n], T1[1:n+1], T2[1:n+1]), and the sorted squared distances in its
# sort_key buffer, gathered together after the sort. A non-finite distance
# (the sort puts it last) is clamped to the largest finite one for the tail
# fit; its squared distance stays non-finite, so no finite-RSS partition holds
# it — it is an outlier, as in the supplied-width sweep.
function _tail_buffers!(sws::ScoringWorkspace, scores, n::Int)
    t = sws.tail
    length(t) >= 3n + 2 || throw(ArgumentError(
        "the workspace's tail buffers cover $(div(length(t) - 2, 3)) points, not $n; build it from this scoring"))
    z = view(t, 1:n)
    S1 = view(t, n+1:2n+1)
    S2 = view(t, 2n+2:3n+2)
    q = view(sws.sort_key, 1:n)
    zfinite = 0.0
    @inbounds for i in 1:n
        qi = Float64(scores[sws.perm[i]])
        q[i] = qi
        zi = sqrt(qi)                      # the distance (d_g = 1: the absolute residual)
        isfinite(zi) ? (zfinite = zi) : (zi = zfinite)
        z[i] = zi
    end
    _suffix_sums!(S1, S2, z)
    return z, S1, S2, q
end

# The sweep over the prefixes of the sorted squared distances `q`, given the
# complement's cost of each (an empty complement costs nothing).
@inline function _tail_sweep!(cost, sws::ScoringWorkspace, s::StudentTUniformScoring, q, n::Int)
    m = s.model_dof
    prior_table = _maybe_prior(sws)
    RSS = 0.0
    best_S, best_k = -Inf, 0
    @inbounds for k in 1:n
        RSS += q[k]
        k <= m && continue
        S = _table_partition_score_cost(s.scale_prior, sws.lg_table, RSS, k, m, 1.0, k == n ? 0.0 : cost(k))
        prior_table === nothing || (S += prior_table[k])
        if S > best_S
            best_S, best_k = S, k
        end
    end
    return best_S, best_k
end

function _outlier_sweep!(model::_EstimatedOutliers, sws::ScoringWorkspace,
                         s::StudentTUniformScoring{Nothing, 1}, scores::Vector{T}, n::Int) where {T}
    n == 0 && return (-Inf, 0)
    _sort_perm!(sws.perm, scores, n; scratch = sws.scratch, radix = sws.radix)
    z, S1, S2, q = _tail_buffers!(sws, scores, n)
    ctx = _outlier_context!(model, sws, z, S1, S2, n)
    S, k = _tail_sweep!(k -> _outlier_cost(model, ctx, z, S1, S2, k, n), sws, s, q, n)
    sws.outlier_halfwidth = k == n ? z[n] : _scored_outlier_halfwidth(model, ctx, z, S1, S2, k, n)
    return S, k
end

# Per model: what is precomputed per hypothesis, the complement's cost of
# prefix k, and the uniform-equivalent half-width recorded at k*.
_outlier_context!(::AbstractOutlierModel, sws, z, S1, S2, n) = nothing

_outlier_cost(m::ExtremeValueHalfwidth, _, z, S1, S2, k, n) =
    (n - k) * log(2 * _tail_fit(m.rules, z, k + 1, S1, S2).outlier_halfwidth)
_scored_outlier_halfwidth(m::ExtremeValueHalfwidth, _, z, S1, S2, k, n) =
    _tail_fit(m.rules, z, k + 1, S1, S2).outlier_halfwidth

function _outlier_context!(m::GeneralizedParetoOutliers{:hypothesis}, sws, z, S1, S2, n)
    fit = _tail_fit(m.rules, z, 1, S1, S2)
    endpoint = gpd_support(fit, z, 1, n)
    suffix = S1                          # suffix[k] = Σ_{i≥k} cost_i, over T1 (no longer needed)
    suffix[n+1] = 0.0
    @inbounds for i in n:-1:1
        suffix[i] = suffix[i+1] + gpd_outlier_cost(z[i], endpoint, fit.shape)
    end
    return (; suffix, outlier_halfwidth = _uniform_equivalent(endpoint, fit.shape))
end
_outlier_cost(::GeneralizedParetoOutliers{:hypothesis}, ctx, z, S1, S2, k, n) = ctx.suffix[k+1]
_scored_outlier_halfwidth(::GeneralizedParetoOutliers{:hypothesis}, ctx, z, S1, S2, k, n) = ctx.outlier_halfwidth
function _outlier_cost(m::GeneralizedParetoOutliers{:prefix}, _, z, S1, S2, k, n)
    fit = _tail_fit(m.rules, z, k + 1, S1, S2)
    return _gpd_complement_cost(z, k + 1, n, gpd_support(fit, z, k + 1, n), fit.shape)
end
function _scored_outlier_halfwidth(m::GeneralizedParetoOutliers{:prefix}, _, z, S1, S2, k, n)
    fit = _tail_fit(m.rules, z, k + 1, S1, S2)
    return _uniform_equivalent(gpd_support(fit, z, k + 1, n), fit.shape)
end

_outlier_cost(m::OutlierTerm, _, z, S1, S2, k, n) = Float64(m.cost(k, z))
_scored_outlier_halfwidth(::OutlierTerm, _, z, S1, S2, k, n) = NaN

# A caller's OutlierTerm states no uniform-equivalent width, so a scorer with
# one has no inlier threshold (the scoring contract's `has_inlier_threshold`).
has_inlier_threshold(::Type{<:StudentTUniformScoring{P, DG, L, <:OutlierTerm}}) where {P, DG, L<:AbstractLabelPrior} = false

