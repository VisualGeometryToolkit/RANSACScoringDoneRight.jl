# =============================================================================
# src/evt.jl — the extreme-value pieces of the score, for the one-candidate
# simulations (d_g = 1): the generalized Pareto fit by probability-weighted
# moments, the estimators of the outlier half-width a from the complement of a
# prefix, the two outlier models the score can use, and the sweep over
# prefixes. No data constant lives here (those are configs/).
#
# The residual is a scalar, so the absolute coordinate of an outlier IS its
# distance and the pooled coordinates of the complement (Sec. 2.1, eq. pooled)
# are the sorted suffix of the distances.
# =============================================================================
using Statistics: quantile, mean, median
using SpecialFunctions: loggamma, logbeta
using Distributions: GeneralizedPareto, logpdf

"""
    gpd_pwm(y) -> (ξ, ω)

Hosking–Wallis probability-weighted-moment fit of a generalized Pareto law
`G(y) = 1 - (1 + ξ y/ω)^(-1/ξ)` to the excesses `y ≥ 0` (any order), with the
plotting positions `(j - 0.35)/n`. Hosking's `k` is `-ξ`.
"""
function gpd_pwm(y::AbstractVector{<:Real})
    ys = sort(y)
    n = length(ys)
    a0 = mean(ys)
    a1 = sum((1 - (j - 0.35) / n) * ys[j] for j in 1:n) / n
    k = a0 / (a0 - 2a1) - 2
    ω = 2a0 * a1 / (a0 - 2a1)
    return (-k, ω)
end

"""
    pot_endpoint(z, η; min_excess, xi_finite) -> (; a, a_fit, ξ, ω)

Peaks over threshold on the pooled coordinates `z` of a complement: threshold
`u = Q_η(z)` (linear interpolation), the GPD of the excesses above `u` by
`gpd_pwm`, and the endpoint `u - ω/ξ` as the half-width. The rules of the
experiments of record: fewer than `min_excess` excesses, or a shape
`ξ ≥ xi_finite` (no finite endpoint), fall back to `maximum(z)` with `ξ = NaN`
for too few excesses; an endpoint below `maximum(z)` is raised to it, so every
outlier lies in the support (`a`; `a_fit` is the endpoint before that rule,
`-Inf` where there is no finite-endpoint fit). `ω` is the fitted scale at `u`.
"""
function pot_endpoint(z::AbstractVector{<:Real}, η::Real; min_excess::Integer, xi_finite::Real)
    zmax = maximum(z)
    u = quantile(z, η; sorted = issorted(z))
    y = [v - u for v in z if v > u]
    length(y) < min_excess && return (; a = zmax, a_fit = -Inf, ξ = NaN, ω = NaN)
    ξ, ω = gpd_pwm(y)
    a_fit = ξ < xi_finite ? u + ω / (-ξ) : -Inf
    return (; a = max(a_fit, zmax), a_fit, ξ, ω)
end

# Block statistics of a random binning of the complement: `keys` are the
# observations' fixed random keys, so a point keeps its block label rank as the
# prefix grows.
function _block_labels(keys::AbstractVector{<:Real}, nb::Integer)
    r = invperm(sortperm(keys))                 # rank of each key, 1-based
    return [fld((ri - 1) * nb, length(keys)) + 1 for ri in r]
end

"""
    block_max_halfwidth(z, B, keys) -> a

The mean of the maxima of `min(B, length(z))` random blocks, times `(m+1)/m`
for the mean block size `m` (unbiased for a uniform on `[0, a]`).
"""
function block_max_halfwidth(z, B::Integer, keys)
    nb = min(B, length(z))
    lab = _block_labels(keys, nb)
    mx = fill(-Inf, nb)
    for (v, l) in zip(z, lab)
        mx[l] = max(mx[l], v)
    end
    m = length(z) / nb
    return mean(mx) * (m + 1) / m
end

"""
    block_min_halfwidth(z, B, keys) -> a

`(m+1)` times the mean of the block minima (unbiased for a uniform on
`[0, a]` when the complement holds only outliers; Sec. 2.3 shows why it
collapses inside the sweep).
"""
function block_min_halfwidth(z, B::Integer, keys)
    nb = min(B, length(z))
    lab = _block_labels(keys, nb)
    mn = fill(Inf, nb)
    for (v, l) in zip(z, lab)
        mn[l] = min(mn[l], v)
    end
    m = length(z) / nb
    return mean(mn) * (m + 1)
end

"""
    jeffreys_inlier_term(k, rss; dg = 1, nθ = 0)

The Jeffreys-limit inlier term `A(I)` (Sec. 2.2) up to its common constant:
`logΓ(ν/2) - (ν/2) log(π RSS)`, `ν = k dg - nθ`.
"""
function jeffreys_inlier_term(k::Integer, rss::Real; dg::Integer = 1, nθ::Integer = 0)
    ν = k * dg - nθ
    return loggamma(ν / 2) - ν / 2 * log(π * rss)
end

"""
    label_prior_term(prior, k, N; beta = (0.5, 0.5))

The log prior weight of a partition with `k` inliers of `N`, up to a constant
common to every partition (dimensionless, so the unit invariance is kept):

- `:half`      the fixed label prior γ = ½ of the profile-marginal score: 0;
- `:profile`   γ profiled out at γ̂ = k/N: `k log(k/N) + (N-k) log(1 - k/N)`;
- `:marginal`  γ integrated out under its conjugate prior Beta(α_γ, β_γ)
  (`beta`; Jeffreys' is (½, ½)), as `neurips2026` (Sec. 3, footnote) proposes:
  `∫ γ^k (1-γ)^(N-k) π(γ) dγ = B(k + α_γ, N - k + β_γ) / B(α_γ, β_γ)`. It
  depends only on `k`, so the sort-and-sweep stays exact.
"""
function label_prior_term(prior::Symbol, k::Integer, N::Integer; beta = (0.5, 0.5))
    prior === :half && return 0.0
    if prior === :profile
        nO = N - k
        return (k > 0 ? k * log(k / N) : 0.0) + (nO > 0 ? nO * log(nO / N) : 0.0)
    end
    αγ, βγ = beta
    prior === :marginal && return logbeta(k + αγ, N - k + βγ) - logbeta(αγ, βγ)
    throw(ArgumentError("label prior $(repr(prior)): :half, :profile or :marginal"))
end

"""
    outlier_loglik(model, z, a, ξ) -> Float64

The log-likelihood of the complement's residuals (d_g = 1, density of the
signed residual = half the density of its absolute value):

- `:uniform`  the Gaussian–uniform score: `-n_O log(2a)`;
- `:gpd`      the extended model: the generalized Pareto law of shape `ξ`
  carried down to zero (threshold stability), endpoint `a`, so its scale at
  zero is `ω₀ = |ξ| a` and the density of `|r|` is
  `(1/ω₀)(1 + ξ z/ω₀)^(-1/ξ-1)`. It is the uniform model at `ξ = -1`, and
  falls back to it where `ξ` is not a finite-endpoint fit. The density is
  Distributions' `GeneralizedPareto(0, ω₀, ξ)`.
"""
function outlier_loglik(model::Symbol, z, a::Real, ξ::Real)
    n = length(z)
    model === :uniform && return -n * log(2a)
    model === :gpd || throw(ArgumentError("outlier model $(repr(model)): :uniform or :gpd"))
    (isfinite(ξ) && ξ < 0) || return -n * log(2a)
    g = GeneralizedPareto(0.0, -ξ * a, ξ)       # location 0, scale ω₀ = |ξ| a, endpoint a
    s = -n * log(2.0)                           # the signed residual: half the density of |r|
    for v in z
        v < a || return -Inf                    # zero density at and beyond the endpoint
        s += logpdf(g, v)
    end
    return s
end

"""
    gpd_support(z, fit) -> a

The endpoint the extended model scores with: the fitted endpoint where it lies
beyond every coordinate (for `ξ > -1` the density vanishes AT the endpoint, so
the largest coordinate cannot sit on it), else the largest coordinate plus the
last spacing, `z₍ₙ₎ + (z₍ₙ₎ - z₍ₙ₋₁₎)` — scale-equivariant, like every rule of
the score. `z` is sorted.
"""
gpd_support(z, fit) = fit.a_fit > z[end] ? fit.a_fit : z[end] + (z[end] - z[end - 1])

"""
    sweep(d, inlier, est; model, kmin, kmax_margin, keys, pot) -> (k, tp, ξ)

The prefix sweep of one candidate (Sec. 2.4) on the distances `d` with the
true labels `inlier`: sort, accumulate RSS, estimate the half-width from each
prefix's complement with `est`, add the label `prior`'s term (a config table: `kind` = "fixed" — `scale`
times the trial's true support — | "fixed_abs" — the width `a` itself, one
value for every trial | "block_max" | "block_min" | "pot"), score with the
outlier `model`, keep the argmax.
Returns the selected prefix length, the number of true inliers in it, and the
median fitted shape over the sweep (NaN unless POT).
"""
function sweep(d::AbstractVector{<:Real}, inlier::AbstractVector{Bool}, est::AbstractDict;
               model::Symbol, prior::Symbol = :half, beta = (0.5, 0.5), kmin::Integer, kmax_margin::Integer, keys::AbstractVector{<:Real},
               true_halfwidth::Real, pot::AbstractDict)
    o = sortperm(d)
    ds, ls, ks = d[o], inlier[o], keys[o]
    N = length(ds)
    cs = cumsum(ds .^ 2)
    kind = est["kind"]
    best, kbest = -Inf, kmin
    ξs = Float64[]
    for k in kmin:(N - kmax_margin)
        z = @view ds[(k + 1):N]
        ξ = NaN
        a = if kind == "fixed"
            est["scale"] * true_halfwidth
        elseif kind == "fixed_abs"
            est["a"]
        elseif kind == "block_max"
            block_max_halfwidth(z, est["blocks"], @view ks[(k + 1):N])
        elseif kind == "block_min"
            block_min_halfwidth(z, est["blocks"], @view ks[(k + 1):N])
        elseif kind == "pot"
            fit = pot_endpoint(z, est["eta"]; min_excess = pot["min_excess"], xi_finite = pot["xi_finite"])
            ξ = fit.ξ
            push!(ξs, ξ)
            model === :gpd ? gpd_support(z, fit) : fit.a
        else
            throw(ArgumentError("estimator kind $(repr(kind))"))
        end
        s = jeffreys_inlier_term(k, cs[k]) + outlier_loglik(model, z, a, ξ) + label_prior_term(prior, k, N; beta)
        if s > best
            best, kbest = s, k
        end
    end
    ξv = filter(isfinite, ξs)
    return (kbest, count(@view ls[1:kbest]), isempty(ξv) ? NaN : median(ξv))
end

"""
    draw_case(rng, case) -> (d, inlier, A)

One candidate's distances: `n_in` inlier distances `|N(0, 1)|` and `n_out`
outlier distances on `[0, A]`, uniform (`outliers = "uniform"`), with the
density `2(A - x)/A²` decreasing linearly to zero (`"linear"`), or
`"mixed"`: each trial draws its own shape (uniform or linear, equally likely)
and its own extent `A`, log-uniform on `[A_min, A_max]` — the heterogeneous
setting in which one calibrated width cannot fit every trial. Returns the
trial's extent with the draws.
"""
function draw_case(rng, case::AbstractDict)
    n_in, n_out = case["n_in"], case["n_out"]
    shape = case["outliers"]
    A = if shape == "mixed"
        lo, hi = log(case["A_min"]), log(case["A_max"])
        shape = rand(rng, ("uniform", "linear"))
        exp(lo + (hi - lo) * rand(rng))
    else
        float(case["A"])
    end
    din = abs.(randn(rng, n_in))
    dout = if shape == "uniform"
        A .* rand(rng, n_out)
    elseif shape == "linear"
        A .* (1 .- sqrt.(rand(rng, n_out)))
    else
        throw(ArgumentError("outliers $(repr(case["outliers"])): uniform, linear or mixed"))
    end
    return vcat(din, dout), vcat(trues(n_in), falses(n_out)), A
end

"""
    partition_metrics(k, tp, n_in) -> (; precision, recall, f1)

Label agreement of a selected prefix of length `k` holding `tp` true inliers.
"""
function partition_metrics(k::Integer, tp::Integer, n_in::Integer)
    precision = tp / k
    recall = tp / n_in
    f1 = 2tp / (k + n_in)
    return (; precision, recall, f1)
end

# -----------------------------------------------------------------------------
# The soft-label formulation: the labels summed out per residual instead of
# profiled over partitions. The inlier scale (Jeffreys) and the mixing
# parameter (its conjugate Beta prior) are integrated out numerically, on a
# grid of log σ relative to the residuals' median (so the score stays
# equivariant under a change of units) and Beta-weighted cells of γ.
# -----------------------------------------------------------------------------
using Distributions: Beta, cdf
using Statistics: median as _median

_logaddexp(a, b) = (m = max(a, b); m == -Inf ? -Inf : m + log1p(exp(-abs(a - b))))

"""
    soft_mixture(d, inlier; model, prior, beta, soft, pot) -> (k, tp, ξ)

The soft-label score of one candidate (d_g = 1): every residual is a mixture
`γ N(r; 0, σ²) + (1 - γ) p_out(r)`, the labels summed out, σ² integrated out
under Jeffreys' prior and γ under Beta(`beta`) (`prior = :marginal`), fixed at
½ (`:half`) or profiled on the γ grid (`:profile`). The outlier density is
fitted once, to the upper tail of ALL residuals (peaks over threshold at
`soft["eta"]`, the rules of `pot`): the uniform level of its endpoint
(`model = :uniform`) or the generalized Pareto density carried down to zero
(`:gpd`). This assumes the top `1 - η` of the residuals are outliers.

Returns the size of the partition the posterior labels (P(inlier | r) > ½),
the true inliers in it, and the fitted shape.
"""
function soft_mixture(d::AbstractVector{<:Real}, inlier::AbstractVector{Bool};
                      model::Symbol, prior::Symbol, beta, soft::AbstractDict, pot::AbstractDict)
    ds = sort(d)
    fit = pot_endpoint(ds, soft["eta"]; min_excess = pot["min_excess"], xi_finite = pot["xi_finite"])
    a = model === :gpd ? gpd_support(ds, fit) : fit.a
    logO = [outlier_loglik(model, [v], a, fit.ξ) for v in d]     # one residual at a time

    lo, hi = soft["log_sigma_range"]
    σs = _median(d) .* exp.(range(lo, hi; length = soft["n_sigma"]))   # Jeffreys on σ²: uniform in log σ
    γs, logwγ = if prior === :half
        [0.5], [0.0]
    else
        M = soft["n_gamma"]
        edges = range(0, 1; length = M + 1)
        mids = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
        B = Beta(beta...)
        w = [cdf(B, edges[m + 1]) - cdf(B, edges[m]) for m in 1:M]
        mids, (prior === :marginal ? log.(w) : zeros(M))
    end
    n, J, M = length(d), length(σs), length(γs)
    logN = [-0.5 * log(2π) - log(σ) - v^2 / (2σ^2) for v in d, σ in σs]
    L = Matrix{Float64}(undef, J, M)                       # log p(r | σ_j, γ_m)
    for m in 1:M, j in 1:J
        lg, l1g = log(γs[m]), log1p(-γs[m])
        L[j, m] = sum(_logaddexp(lg + logN[i, j], l1g + logO[i]) for i in 1:n)
    end
    # The posterior over the grid: marginal (prior weights) or profiled (argmax in γ).
    logpost = L .+ logwγ'
    if prior === :profile
        mbest = argmax(vec(maximum(logpost; dims = 1)))
        logpost = fill(-Inf, J, M); logpost[:, mbest] .= L[:, mbest]
    end
    w = exp.(logpost .- maximum(logpost)); w ./= sum(w)
    pin = zeros(n)
    for m in 1:M, j in 1:J
        w[j, m] > 0 || continue
        lg, l1g = log(γs[m]), log1p(-γs[m])
        for i in 1:n
            li = lg + logN[i, j]
            pin[i] += w[j, m] * exp(li - _logaddexp(li, l1g + logO[i]))
        end
    end
    lab = pin .> 0.5
    k = count(lab)
    return (max(k, 1), count(lab .& inlier), fit.ξ)
end
