# =============================================================================
# src/simulation.jl — the one-candidate simulations of Sec. 4.1: the
# generative model of the draws, the label metrics, the research baselines
# that are not scorers (block statistics of the complement, scored through the
# package's `OutlierTerm`), and the soft-label formulation. The score itself —
# the sort-and-sweep, the outlier models, the label priors, the generalized
# Pareto fit — is RANSACScoringDoneRight's StudentTUniformScoring; nothing
# here re-implements it. No data constant lives here (those are configs/).
# =============================================================================
using RANSACScoringDoneRight: StudentTUniformScoring, TailRules, ExtremeValueHalfwidth,
    GeneralizedParetoOutliers, OutlierTerm, FixedHalfwidth, FlatLabelPrior, ProfiledLabelPrior,
    BetaLabelPrior, outlier_model, label_prior, alloc_scoring_workspace, score_residuals!,
    tail_fit, gpd_support, gpd_outlier_cost
using VisualGeometryToolkitCore: consensus_order, is_map_inlier, scale_prior
using LogExpFunctions: logaddexp
using Distributions: Beta, cdf
using Statistics: mean, quantile

"""
    Trial(rng, case)

One candidate's draw and what every estimator shares: the signed scalar
residuals `r` (`n_in` inliers from `N(0, 1)`, `n_out` outliers on `[-H, H]`
whose absolute values are uniform — `outliers = "uniform"` — or have the
density `2(H - x)/H²` decreasing linearly to zero — `"linear"` — or
`"mixed"`: each trial its own shape, uniform or linear, and its own `H`,
log-uniform on `[outlier_halfwidth_min, outlier_halfwidth_max]`), the true
labels, `H`, the absolute residuals `z` in the sweep's order (ascending `r²`,
stable), and each point's random block key rank in that order (for the block
baselines).
"""
struct Trial
    r::Vector{Float64}
    inlier::BitVector
    outlier_halfwidth::Float64
    z::Vector{Float64}
    block_rank::Vector{Int}      # in the order of z
end

function Trial(rng, case::AbstractDict)
    n_in, n_out = case["n_in"], case["n_out"]
    shape = case["outliers"]
    H = if shape == "mixed"
        lo, hi = log(case["outlier_halfwidth_min"]), log(case["outlier_halfwidth_max"])
        shape = rand(rng, ("uniform", "linear"))
        exp(lo + (hi - lo) * rand(rng))
    else
        float(case["outlier_halfwidth"])
    end
    rin = randn(rng, n_in)
    zout = if shape == "uniform"
        H .* rand(rng, n_out)
    elseif shape == "linear"
        H .* (1 .- sqrt.(rand(rng, n_out)))
    else
        throw(ArgumentError("outliers $(repr(case["outliers"])): uniform, linear or mixed"))
    end
    r = vcat(rin, zout .* rand(rng, (-1, 1), n_out))
    keys = rand(rng, length(r))
    o = sortperm(r .^ 2)                 # the sweep's own sort key and order (stable)
    return Trial(r, vcat(trues(n_in), falses(n_out)), H, abs.(r)[o], invperm(sortperm(keys))[o])
end

n_inliers(t::Trial) = count(t.inlier)

"""
    partition_metrics(k, tp, n_in) -> (; precision, recall, f1)

Label agreement of a selected partition of `k` observations holding `tp`
true inliers.
"""
partition_metrics(k::Integer, tp::Integer, n_in::Integer) =
    (; precision = tp / max(k, 1), recall = tp / n_in, f1 = 2tp / (k + n_in))

"""
    label_prior_of(name, beta) -> AbstractLabelPrior

The package's label prior for a config name: `half` (γ = ½), `profile`
(γ̂ = k/N), `marginal` (Beta(`beta`)).
"""
label_prior_of(name::AbstractString, beta) =
    name == "half" ? FlatLabelPrior() : name == "profile" ? ProfiledLabelPrior() :
    name == "marginal" ? BetaLabelPrior(beta...) :
    throw(ArgumentError("label prior $(repr(name)): half, profile or marginal"))

"""
    tail_rules(est, pot) -> TailRules

The package's tail rules: the estimator's `threshold_quantile` and the
`[pot]` rules.
"""
tail_rules(est::AbstractDict, pot::AbstractDict) =
    TailRules(; threshold_quantile = est["threshold_quantile"], min_excess = pot["min_excess"],
              finite_shape_limit = pot["finite_shape_limit"])

"""
    tail_model(est, pot; model, fit_per = :prefix) -> AbstractOutlierModel

The package's outlier model for a peaks-over-threshold estimator:
`ExtremeValueHalfwidth` (`model = :uniform`, component I) or
`GeneralizedParetoOutliers` (`:gpd`, component II).
"""
function tail_model(est::AbstractDict, pot::AbstractDict; model::Symbol, fit_per::Symbol = :prefix)
    rules = tail_rules(est, pot)
    model === :uniform && return ExtremeValueHalfwidth(rules)
    model === :gpd && return GeneralizedParetoOutliers{fit_per}(rules)
    throw(ArgumentError("outlier model $(repr(model)): uniform or gpd"))
end

"""
    WorkspaceCache()

Scoring workspaces reused across the estimators and grid widths of a run: a
workspace depends on the data count, the scale prior, the label prior and
whether the outlier model estimates the support, not on a supplied width.
"""
struct WorkspaceCache
    workspaces::Dict{Any, Any}
end
WorkspaceCache() = WorkspaceCache(Dict{Any, Any}())
workspace!(c::WorkspaceCache, s, n::Int) =
    get!(() -> alloc_scoring_workspace(n, 1, s), c.workspaces,
         (n, scale_prior(s), label_prior(s), outlier_model(s) isa FixedHalfwidth))

"""
    hard_labels(trial, scoring, cache) -> (k, tp)

The partition the package's sort-and-sweep selects with `scoring`, and its
true inliers (read from the scorer's own consensus order).
"""
function hard_labels(t::Trial, scoring, cache::WorkspaceCache)
    n = length(t.r)
    sws = workspace!(cache, scoring, n)
    _, k = score_residuals!(sws, zeros(n), scoring, t.r, n)
    return k, count(i -> t.inlier[i], view(consensus_order(scoring, sws), 1:k))
end

"""
    block_term(trial, stat, blocks) -> OutlierTerm

The research baseline of Sec. 2.4: each prefix's complement pays the uniform
level of a block-statistic half-width — `stat = :max`, `(m+1)/m` times the
mean of the block maxima; `:min`, `(m+1)` times the mean of the block minima
(m the mean block size) — over a random binning of the observations into
`blocks` blocks by their key rank, fixed as the prefix grows.
"""
function block_term(t::Trial, stat::Symbol, blocks::Integer)
    lab = [mod(r - 1, blocks) + 1 for r in t.block_rank]
    v = Vector{Float64}(undef, blocks)
    c = Vector{Int}(undef, blocks)
    return OutlierTerm() do k, z
        n = length(z)
        fill!(v, stat === :max ? -Inf : Inf); fill!(c, 0)
        @inbounds for i in k+1:n
            l = lab[i]
            v[l] = stat === :max ? max(v[l], z[i]) : min(v[l], z[i]); c[l] += 1
        end
        nb = count(>(0), c)
        mval = sum(v[l] for l in 1:blocks if c[l] > 0) / nb
        m = (n - k) / nb
        a = stat === :max ? mval * (m + 1) / m : mval * (m + 1)
        return (n - k) * log(2a)
    end
end

"""
    soft_labels(trial; model, prior, beta, soft, pot) -> (k, tp, shape)

The soft-label formulation (Sec. 3.2): every residual is a mixture
`γ N(r; 0, σ²) + (1 - γ) p_out(r)`, the labels summed out, σ² integrated out
under Jeffreys' prior on a grid of log σ placed relative to the median
|residual| (so the score moves with the units), and γ fixed at ½ (`half`),
profiled on its grid (`profile`) or integrated out under Beta(`beta`) on
Beta-weighted cells (`marginal`). The outlier density — the package's tail
fit, support rule and outlier cost — is fitted once, to the upper tail of all
residuals. Returns the partition the posterior labels (`is_map_inlier`), its
true inliers and the fitted shape.
"""
function soft_labels(t::Trial; model::Symbol, prior::AbstractString, beta, soft::AbstractDict, pot::AbstractDict)
    r, z = t.r, t.z
    n = length(r)
    fit = tail_fit(tail_rules(soft, pot), z)
    endpoint = model === :gpd ? gpd_support(fit, z, 1, n) : fit.outlier_halfwidth
    shape = model === :gpd ? fit.shape : NaN
    logO = [-gpd_outlier_cost(abs(v), endpoint, shape) for v in r]

    lo, hi = soft["log_sigma_range"]
    logσ = log(quantile(z, 0.5; sorted = true)) .+ range(lo, hi; length = soft["n_sigma"])   # Jeffreys: uniform in log σ
    γs, logwγ = if prior == "half"
        [0.5], [0.0]
    else
        M = soft["n_gamma"]
        edges = range(0, 1; length = M + 1)
        B = Beta(beta...)
        collect((edges[1:end-1] .+ edges[2:end]) ./ 2),
            prior == "marginal" ? [log(cdf(B, edges[m+1]) - cdf(B, edges[m])) for m in 1:M] : zeros(M)
    end
    J, M = length(logσ), length(γs)
    logN = [-0.5 * log(2π) - ls - v^2 / (2 * exp(2ls)) for v in r, ls in logσ]
    lγ, l1γ = log.(γs), log1p.(-γs)
    L = [sum(logaddexp(lγ[m] + logN[i, j], l1γ[m] + logO[i]) for i in 1:n) for j in 1:J, m in 1:M]
    logpost = L .+ logwγ'
    if prior == "profile"
        mbest = argmax(vec(maximum(logpost; dims = 1)))
        logpost = fill(-Inf, J, M); logpost[:, mbest] .= L[:, mbest]
    end
    w = exp.(logpost .- maximum(logpost)); w ./= sum(w)
    pin = zeros(n)
    for m in 1:M, j in 1:J
        w[j, m] > 0 || continue
        for i in 1:n
            li = lγ[m] + logN[i, j]
            pin[i] += w[j, m] * exp(li - logaddexp(li, l1γ[m] + logO[i]))
        end
    end
    lab = is_map_inlier.(pin)
    return max(count(lab), 1), count(lab .& t.inlier), fit.shape
end
