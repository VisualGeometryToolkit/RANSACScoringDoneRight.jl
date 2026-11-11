# =============================================================================
# extreme_value_sweep.jl — this package's outlier models on VisualGeometryToolkitCore's
# estimated-support sweep (the sweep of a `StudentTUniformScoring` whose outlier
# model estimates the support: Core's sort, tail buffers, per-partition score and
# label prior, moved there 2026-11-11). Only the outlier term differs per model,
# and it is stated here through Core's hooks (`outlier_context!`, `outlier_cost`,
# `scored_outlier_halfwidth`):
#
#   ExtremeValueHalfwidth                  the complement of prefix k is the
#       suffix z[k+1:n] of the sorted distances and its upper tail a suffix of
#       that: one O(log n) threshold and O(1) moments from two suffix sums
#   GeneralizedParetoOutliers(:prefix)     Θ(n²): the density changes with
#       every prefix's fit and Σ log p does not decompose
#   GeneralizedParetoOutliers(:hypothesis) Θ(n): one fit to the upper tail of
#       all distances, a fixed per-point cost, a suffix sum
#   OutlierTerm                            the caller's cost per prefix
#
# `z` are the sorted distances, `S1`/`S2` their suffix sums (Core's `suffix_sums!`).
# Allocation-free: every buffer is the workspace's.
# =============================================================================

outlier_cost(m::ExtremeValueHalfwidth, _, z, S1, S2, k, n) =
    (n - k) * log(2 * _tail_fit(m.rules, z, k + 1, S1, S2).outlier_halfwidth)
scored_outlier_halfwidth(m::ExtremeValueHalfwidth, _, z, S1, S2, k, n) =
    _tail_fit(m.rules, z, k + 1, S1, S2).outlier_halfwidth

function outlier_context!(m::GeneralizedParetoOutliers{:hypothesis}, z, S1, S2, n)
    fit = _tail_fit(m.rules, z, 1, S1, S2)
    endpoint = gpd_support(fit, z, 1, n)
    suffix = S1                          # suffix[k] = Σ_{i≥k} cost_i, over T1 (no longer needed)
    suffix[n+1] = 0.0
    @inbounds for i in n:-1:1
        suffix[i] = suffix[i+1] + gpd_outlier_cost(z[i], endpoint, fit.shape)
    end
    return (; suffix, outlier_halfwidth = _uniform_equivalent(endpoint, fit.shape))
end
outlier_cost(::GeneralizedParetoOutliers{:hypothesis}, ctx, z, S1, S2, k, n) = ctx.suffix[k+1]
scored_outlier_halfwidth(::GeneralizedParetoOutliers{:hypothesis}, ctx, z, S1, S2, k, n) = ctx.outlier_halfwidth
function outlier_cost(m::GeneralizedParetoOutliers{:prefix}, _, z, S1, S2, k, n)
    fit = _tail_fit(m.rules, z, k + 1, S1, S2)
    return _gpd_complement_cost(z, k + 1, n, gpd_support(fit, z, k + 1, n), fit.shape)
end
function scored_outlier_halfwidth(m::GeneralizedParetoOutliers{:prefix}, _, z, S1, S2, k, n)
    fit = _tail_fit(m.rules, z, k + 1, S1, S2)
    return _uniform_equivalent(gpd_support(fit, z, k + 1, n), fit.shape)
end

outlier_cost(m::OutlierTerm, _, z, S1, S2, k, n) = Float64(m.cost(k, z))
scored_outlier_halfwidth(::OutlierTerm, _, z, S1, S2, k, n) = NaN

# A caller's OutlierTerm states no uniform-equivalent width, so a scorer with
# one has no decision boundary (the scoring contract's `has_decision_boundary`).
has_decision_boundary(::Type{<:StudentTUniformScoring{P, DG, L, <:OutlierTerm}}) where {P, DG, L<:AbstractLabelPrior} = false
