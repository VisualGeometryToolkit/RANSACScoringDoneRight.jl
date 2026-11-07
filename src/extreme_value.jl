# =============================================================================
# extreme_value.jl — the label priors and the outlier models of
# StudentTUniformScoring, and the generalized Pareto fit they use.
#
#   label priors    the prior of a partition's labels, a table over the inlier
#                   count k (the workspace's `prior_table`): flat γ = ½, γ
#                   profiled at k/N, or γ integrated out under Beta(α, β)
#   outlier models  the outlier term of a partition: a supplied half-width
#                   (the profile-marginal score), the half-width estimated as
#                   the endpoint of a peaks-over-threshold fit to the
#                   complement's upper tail, or the fitted generalized Pareto
#                   law itself as the outlier density
#
# The sweep that uses them is in extreme_value_sweep.jl. Every rule here is
# equivariant under a change of the residuals' units, so the selected partition
# is invariant to it.
# =============================================================================

# -----------------------------------------------------------------------------
# Label priors
# -----------------------------------------------------------------------------

"""
    AbstractLabelPrior

The prior of a partition's labels in [`StudentTUniformScoring`](@ref): it adds
a term that depends on the partition only through its inlier count `k`, held
in the workspace's `prior_table`, so the sort-and-sweep stays exact.
"""
abstract type AbstractLabelPrior end

"""
    FlatLabelPrior()

The inlier probability fixed at γ = ½: every partition has prior weight
`2^{-N}`, so the label prior adds nothing (the profile-marginal score's choice).
"""
struct FlatLabelPrior <: AbstractLabelPrior end

"""
    ProfiledLabelPrior()

The inlier probability profiled at γ̂ = k/N: the partition gains
`k log(k/N) + (N-k) log((N-k)/N)`.
"""
struct ProfiledLabelPrior <: AbstractLabelPrior end

"""
    BetaLabelPrior(alpha = 0.5, beta = 0.5)

The inlier probability integrated out under its conjugate prior
`Beta(alpha, beta)` (Jeffreys' by default): the partition gains
`log B(k + alpha, N - k + beta) - log B(alpha, beta)`.
"""
struct BetaLabelPrior <: AbstractLabelPrior
    alpha::Float64
    beta::Float64
    function BetaLabelPrior(alpha::Real = 0.5, beta::Real = 0.5)
        (alpha > 0 && beta > 0) ||
            throw(ArgumentError("BetaLabelPrior: alpha and beta must be positive, got ($alpha, $beta)"))
        return new(Float64(alpha), Float64(beta))
    end
end

# The workspace's table of the label term for k = 1..n (empty for the flat prior).
_build_prior_table(n::Int, ::FlatLabelPrior) = Float64[]
_build_prior_table(n::Int, ::ProfiledLabelPrior) = _build_prior_table(n)
function _build_prior_table(n::Int, p::BetaLabelPrior)
    c = logbeta(p.alpha, p.beta)
    return [logbeta(k + p.alpha, n - k + p.beta) - c for k in 1:n]
end

# -----------------------------------------------------------------------------
# Outlier models
# -----------------------------------------------------------------------------

"""
    AbstractOutlierModel

The outlier component of [`StudentTUniformScoring`](@ref): what a partition's
complement pays.
"""
abstract type AbstractOutlierModel end

"""
    FixedHalfwidth(outlier_halfwidth)

The uniform outlier density at a supplied half-width per axis (the
profile-marginal score): the complement pays `log(2·outlier_halfwidth)` per
point and axis.
"""
struct FixedHalfwidth <: AbstractOutlierModel
    outlier_halfwidth::Float64
    function FixedHalfwidth(outlier_halfwidth::Real)
        outlier_halfwidth > 0 ||
            throw(ArgumentError("outlier_halfwidth must be positive, got $outlier_halfwidth"))
        return new(Float64(outlier_halfwidth))
    end
end

"""
    TailRules(; threshold_quantile = 0.9, min_excess = 5, finite_shape_limit = -1e-3)

The peaks-over-threshold rules both extreme-value outlier models share: the
generalized Pareto law is fitted by probability-weighted moments (Hosking &
Wallis 1987) to the excesses over the complement's `threshold_quantile`
quantile. Fewer than `min_excess` excesses, or a shape at or above
`finite_shape_limit` (no finite endpoint), fall back to the largest residual
of the complement; an endpoint below it is raised to it.
"""
struct TailRules
    threshold_quantile::Float64
    min_excess::Int
    finite_shape_limit::Float64
    function TailRules(; threshold_quantile::Real = 0.9, min_excess::Integer = 5,
                       finite_shape_limit::Real = -1e-3)
        0 < threshold_quantile < 1 ||
            throw(ArgumentError("threshold_quantile must lie in (0, 1), got $threshold_quantile"))
        min_excess >= 2 || throw(ArgumentError("min_excess must be at least 2, got $min_excess"))
        finite_shape_limit <= 0 ||
            throw(ArgumentError("finite_shape_limit must be non-positive, got $finite_shape_limit"))
        return new(Float64(threshold_quantile), Int(min_excess), Float64(finite_shape_limit))
    end
end

"""
    ExtremeValueHalfwidth(; tail rules...)

The uniform outlier density at a half-width estimated from each partition's
complement: the endpoint `u - ω/ξ` of the generalized Pareto law the
[`TailRules`](@ref) (its keywords) fit to the complement's upper tail.
"""
struct ExtremeValueHalfwidth <: AbstractOutlierModel
    rules::TailRules
end
ExtremeValueHalfwidth(; kw...) = ExtremeValueHalfwidth(TailRules(; kw...))

"""
    GeneralizedParetoOutliers(; fit_per = :prefix, tail rules...)

The fitted generalized Pareto law itself as the outlier density: the law of
the [`TailRules`](@ref) fit, carried down to zero by threshold stability, with
scale `|ξ|·a` at zero and endpoint `a` — the uniform density at `ξ = -1`, and
the uniform one where there is no finite-endpoint fit. `fit_per = :prefix`
refits the tail for every partition (exact, `Θ(N²)` per hypothesis);
`:hypothesis` fits it once to the upper tail of all residuals (`Θ(N)`,
without the prefix-optimality guarantee).
"""
struct GeneralizedParetoOutliers{F} <: AbstractOutlierModel
    rules::TailRules
    function GeneralizedParetoOutliers{F}(rules::TailRules) where {F}
        F in (:prefix, :hypothesis) ||
            throw(ArgumentError("fit_per must be :prefix or :hypothesis, got $(repr(F))"))
        return new{F}(rules)
    end
end
GeneralizedParetoOutliers(; fit_per::Symbol = :prefix, kw...) = GeneralizedParetoOutliers{fit_per}(TailRules(; kw...))
fit_per(::GeneralizedParetoOutliers{F}) where {F} = F

"""
    OutlierTerm(cost)

A caller's outlier term, for research baselines that are not one of the
models above (block statistics of the complement): `cost(k, z)` is the
complement's `-log` density for the partition of the `k` smallest residuals,
`z` the sorted absolute residuals.
"""
struct OutlierTerm{F} <: AbstractOutlierModel
    cost::F
end

# The models that estimate the support from the residuals (they need the
# sweep's sorted distances and prefix sums).
const _EstimatedOutliers = Union{ExtremeValueHalfwidth, GeneralizedParetoOutliers, OutlierTerm}
_needs_tail(::AbstractOutlierModel) = true
_needs_tail(::FixedHalfwidth) = false

# -----------------------------------------------------------------------------
# The generalized Pareto fit
# -----------------------------------------------------------------------------

"""
    gpd_pwm_fit(n, sum_excess, sum_rank_excess) -> (shape, scale)

The probability-weighted-moment fit of a generalized Pareto law
`G(y) = 1 - (1 + shape·y/scale)^(-1/shape)` to `n` excesses, from their sum
and their rank-weighted sum `Σ_j j·y_(j)` (ascending ranks), with the
plotting positions `(j - 0.35)/n` (Hosking & Wallis 1987; Hosking's `k` is
`-shape`). Taking the sums rather than the excesses lets the sweep fit every
prefix's tail in O(1) from prefix sums.
"""
function gpd_pwm_fit(n::Integer, sum_excess::Float64, sum_rank_excess::Float64)
    a0 = sum_excess / n
    a1 = a0 - (sum_rank_excess - 0.35 * sum_excess) / n^2
    d = a0 - 2a1
    return (-(a0 / d - 2), 2a0 * a1 / d)
end

"""
    gpd_pwm_fit(excess) -> (shape, scale)

The same fit from the excesses themselves (any order).
"""
function gpd_pwm_fit(excess::AbstractVector{<:Real})
    y = sort(Float64.(excess))
    return gpd_pwm_fit(length(y), sum(y), sum(j * y[j] for j in eachindex(y)))
end

# The no-fit result: the uniform support of the largest value.
_no_tail_fit(zmax) = (; outlier_halfwidth = zmax, fitted_endpoint = -Inf, shape = NaN)

# Statistics' default quantile (type 7: alpha = beta = 1) of the sorted finite
# values `v`, in O(1). `Statistics.quantile(v, p; sorted = true)` gives the same
# value but scans all of `v` for NaN on every call, which would make the sweep
# Θ(n²); the sweep's distances are finite (non-finite ones are clamped,
# extreme_value_sweep.jl). Pinned to it by the tests.
@inline function _sorted_quantile(v::AbstractVector{Float64}, p::Float64)
    n = length(v)
    aleph = fma(n, p, 1 - p)                 # Statistics' alpha = beta = 1
    j = clamp(trunc(Int, aleph), 1, n - 1)
    γ = clamp(aleph - j, 0.0, 1.0)
    a, b = n == 1 ? (v[1], v[1]) : (v[j], v[j + 1])
    return (isfinite(a) && isfinite(b) && a ≈ b) ? a + γ * (b - a) : (1 - γ) * a + γ * b
end

# The fit of `rules` to the upper part z[lo:end] of the sorted values z, from
# the suffix sums of `_suffix_sums!` (T1[g] = Σ_{l≥g} z[l], T2[g] =
# Σ_{l≥g} (n-l+1)·z[l], ranks counted from the top, so a short tail's moments
# never come from a difference of two large sums): O(1) for the threshold,
# O(log n) for the first value above it (bisection, ties included), O(1) for
# the moments. Every rule that falls back returns the no-fit result.
@inline function _tail_fit(rules::TailRules, z, lo::Int, T1, T2)
    n = length(z)
    zmax = z[n]
    n - lo + 1 >= 2 || return _no_tail_fit(zmax)
    tail = view(z, lo:n)
    u = _sorted_quantile(tail, rules.threshold_quantile)
    g = lo + searchsortedlast(tail, u)                  # the first value above u
    m = n - g + 1
    m >= rules.min_excess || return _no_tail_fit(zmax)
    sz = T1[g]                                          # Σ_{l=g}^{n} z[l]
    sjz = (m + 1) * sz - T2[g]                          # Σ_{j=1}^{m} j·z[g-1+j], ascending ranks
    shape, scale = gpd_pwm_fit(m, sz - m * u, sjz - u * m * (m + 1) / 2)
    (shape < rules.finite_shape_limit && scale > 0) || return _no_tail_fit(zmax)
    endpoint = u - scale / shape
    return (; outlier_halfwidth = max(endpoint, zmax), fitted_endpoint = endpoint, shape)
end

# The suffix sums of the sorted values z into T1 and T2 (length n + 1, T·[n+1] = 0).
function _suffix_sums!(T1, T2, z)
    n = length(z)
    T1[n+1] = 0.0; T2[n+1] = 0.0
    @inbounds for l in n:-1:1
        T1[l] = T1[l+1] + z[l]
        T2[l] = T2[l+1] + (n - l + 1) * z[l]
    end
    return T1, T2
end

_rules(m::Union{ExtremeValueHalfwidth, GeneralizedParetoOutliers}) = m.rules
_rules(r::TailRules) = r

"""
    tail_fit(rules, z) -> (; outlier_halfwidth, fitted_endpoint, shape)

The peaks-over-threshold fit of `rules` (a [`TailRules`](@ref) or an
extreme-value outlier model) to the sorted values `z`: `outlier_halfwidth` is
the uniform half-width (the fitted endpoint, raised to the largest value);
`fitted_endpoint` is `-Inf` and `shape` `NaN` wherever the rules fall back
(too few excesses, or no finite-endpoint fit). The sweep fits every prefix's
complement the same way, in O(log n) each, from suffix sums it builds once.
"""
function tail_fit(rules, z::AbstractVector{Float64})
    n = length(z)
    T1, T2 = _suffix_sums!(Vector{Float64}(undef, n + 1), Vector{Float64}(undef, n + 1), z)
    return _tail_fit(_rules(rules), z, 1, T1, T2)
end

"""
    gpd_support(fit, z, lo, hi) -> endpoint

The endpoint the generalized Pareto density scores with: the fitted endpoint
where it lies beyond every value (for `shape > -1` the density vanishes at the
endpoint, so the largest value cannot sit on it), else the largest value plus
the mean spacing of `z[lo:hi]` (positive unless every value ties) —
equivariant under a change of units.
"""
@inline function gpd_support(fit, z, lo::Int, hi::Int)
    fit.fitted_endpoint > z[hi] && return fit.fitted_endpoint
    spacing = hi > lo ? (z[hi] - z[lo]) / (hi - lo) : 0.0
    return spacing > 0 ? z[hi] + spacing : 2 * z[hi]
end

_is_pareto(shape) = isfinite(shape) && shape < 0

"""
    gpd_outlier_cost(z, endpoint, shape) -> Float64

`-log p(r)` for one signed scalar residual of absolute value `z` under the
generalized Pareto outlier density of endpoint `endpoint` and shape `shape`
(`GeneralizedPareto(0, σ, shape)` with `σ = |shape|·endpoint`, halved for the
sign): `log 2 + log σ + (1 + 1/shape)·log(1 - z/endpoint)`; the uniform
`log(2·endpoint)` where `shape` is not a finite-endpoint fit.
"""
@inline function gpd_outlier_cost(z::Float64, endpoint::Float64, shape::Float64)
    _is_pareto(shape) || return log(2endpoint)
    z < endpoint || return Inf
    return log(-2shape * endpoint) + (1 + 1 / shape) * log1p(-z / endpoint)
end

# The complement z[lo:hi]'s total cost under the density of `endpoint` and
# `shape`: the normalizer once, one log1p per point (the uniform level in O(1)).
@inline function _gpd_complement_cost(z, lo::Int, hi::Int, endpoint::Float64, shape::Float64)
    n = hi - lo + 1
    _is_pareto(shape) || return n * log(2endpoint)
    s = 0.0
    @inbounds for i in lo:hi
        s += log1p(-z[i] / endpoint)
    end
    return n * log(-2shape * endpoint) + (1 + 1 / shape) * s
end

# The uniform half-width whose density equals the generalized Pareto density at
# the model, f(0) = 1/(|ξ|·endpoint): `|ξ|·endpoint`; the endpoint itself for
# the uniform one.
_uniform_equivalent(endpoint::Float64, shape::Float64) = _is_pareto(shape) ? -shape * endpoint : endpoint
