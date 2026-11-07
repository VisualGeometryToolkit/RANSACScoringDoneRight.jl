@testitem "extreme_value" begin
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: alloc_scoring_workspace, tail_fit, gpd_pwm_fit, gpd_support,
        gpd_outlier_cost, _build_prior_table, _tail_fit, _suffix_sums!, _sorted_quantile
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: IGPrior, check_workspace, with_width, OutlierHalfwidth, outlier_halfwidth
    using Distributions: GeneralizedPareto, logpdf
    using SpecialFunctions: loggamma, logbeta
    using Statistics: quantile
    using Random: Xoshiro

    # An independent reference for one complement z (sorted): the textbook
    # peaks-over-threshold fit, no prefix sums.
    function naive_tail(z, η; min_excess = 5, finite_shape_limit = -1e-3)
        u = quantile(z, η; sorted = true)
        y = sort([v - u for v in z if v > u])
        length(y) < min_excess && return (; outlier_halfwidth = z[end], fitted_endpoint = -Inf, shape = NaN)
        m = length(y)
        a0 = sum(y) / m
        a1 = sum((1 - (j - 0.35) / m) * y[j] for j in 1:m) / m
        shape = -(a0 / (a0 - 2a1) - 2)
        scale = 2a0 * a1 / (a0 - 2a1)
        (shape < finite_shape_limit && scale > 0) || return (; outlier_halfwidth = z[end], fitted_endpoint = -Inf, shape = NaN)
        e = u - scale / shape
        return (; outlier_halfwidth = max(e, z[end]), fitted_endpoint = e, shape)
    end
    jeffreys(k, m, rss) = loggamma((k - m) / 2) - (k - m) / 2 * log(2π) - (k - m) / 2 * log(rss / 2)

    draw(rng; n_in = 120, n_out = 280, A = 50.0) = vcat(randn(rng, n_in), A .* (2 .* rand(rng, n_out) .- 1))

    @testset "the prefix-sum tail fit == the textbook fit on every suffix" begin
        rng = Xoshiro(1)
        z = sort(abs.(draw(rng)))
        n = length(z)
        T1, T2 = _suffix_sums!(zeros(n + 1), zeros(n + 1), z)
        rules = TailRules(; threshold_quantile = 0.9)
        @test tail_fit(ExtremeValueHalfwidth(), z) == _tail_fit(rules, z, 1, T1, T2)
        for lo in (1, 2, 50, 200, 330, n - 30, n - 3)
            got = _tail_fit(rules, z, lo, T1, T2)
            want = naive_tail(z[lo:n], 0.9)
            @test got.outlier_halfwidth ≈ want.outlier_halfwidth rtol = 1e-10
            @test isequal(isnan(got.shape), isnan(want.shape))
            isnan(want.shape) || @test got.shape ≈ want.shape rtol = 1e-9
        end
    end

    @testset "the sorted quantile is Statistics' default" begin
        v = sort(abs.(draw(Xoshiro(4))))
        for p in (0.0, 0.1, 0.5, 0.9, 0.95, 1.0), w in (v, v[1:1], v[1:2], v[end-6:end], [1.0, 1.0, 1.0, 2.0])
            @test _sorted_quantile(w, p) == quantile(w, p; sorted = true)
        end
    end

    @testset "gpd_pwm_fit recovers Distributions' GeneralizedPareto" begin
        rng = Xoshiro(2)
        for shape in (-1.0, -0.5, -0.25), scale in (1.0, 40.0)
            ŝ, ω̂ = gpd_pwm_fit(rand(rng, GeneralizedPareto(0.0, scale, shape), 200_000))
            @test ŝ ≈ shape atol = 0.02
            @test ω̂ ≈ scale rtol = 0.02
        end
    end

    @testset "the extreme-value sweep == the brute-force max of its partition score" begin
        m = 2
        for seed in 1:4, label_prior in (FlatLabelPrior(), BetaLabelPrior())
            r = draw(Xoshiro(seed))
            n = length(r)
            s = StudentTUniformScoring(m, ExtremeValueHalfwidth(); label_prior)
            sws = alloc_scoring_workspace(n, 1, s)
            S, k = score_residuals!(sws, zeros(n), s, r, n)
            z = sort(abs.(r))
            pt = _build_prior_table(n, label_prior)
            Sk = map(m+1:n) do k
                a = k == n ? z[n] : naive_tail(z[k+1:n], 0.9).outlier_halfwidth
                jeffreys(k, m, sum(abs2, z[1:k])) - (n - k) * log(2a) + (isempty(pt) ? 0.0 : pt[k])
            end
            @test S ≈ maximum(Sk) rtol = 1e-10
            @test k == m + argmax(Sk)
        end
    end

    @testset "generalized Pareto outliers: per prefix and per hypothesis == brute force" begin
        m = 2
        r = draw(Xoshiro(7); A = 80.0)
        n = length(r)
        z = sort(abs.(r))
        cost(v, a, ξ) = (isfinite(ξ) && ξ < 0) ? log(2.0) - logpdf(GeneralizedPareto(0.0, -ξ * a, ξ), v) : log(2a)
        support(f, zs) = f.fitted_endpoint > zs[end] ? f.fitted_endpoint :
                         (length(zs) > 1 && zs[end] > zs[1] ? zs[end] + (zs[end] - zs[1]) / (length(zs) - 1) : 2 * zs[end])
        for fit_per in (:prefix, :hypothesis)
            s = StudentTUniformScoring(m, GeneralizedParetoOutliers(; fit_per))
            sws = alloc_scoring_workspace(n, 1, s)
            S, k = score_residuals!(sws, zeros(n), s, r, n)
            whole = naive_tail(z, 0.9)
            Sk = map(m+1:n) do k
                k == n && return jeffreys(k, m, sum(abs2, z))
                f = fit_per === :prefix ? naive_tail(z[k+1:n], 0.9) : whole
                zs = fit_per === :prefix ? z[k+1:n] : z
                a = support(f, zs)
                jeffreys(k, m, sum(abs2, z[1:k])) - sum(cost(v, a, f.shape) for v in z[k+1:n])
            end
            @test S ≈ maximum(Sk) rtol = 1e-9
            @test k == m + argmax(Sk)
        end
    end

    @testset "the generalized Pareto density is the uniform one at shape -1" begin
        for v in (0.1, 3.0, 9.9)
            @test gpd_outlier_cost(v, 10.0, -1.0) ≈ log(20.0)
        end
        @test gpd_outlier_cost(10.0, 10.0, -0.5) == Inf
        # the closed form is Distributions' GeneralizedPareto from zero, halved for the sign
        for (v, ξ) in ((0.3, -0.5), (7.0, -0.8), (2.0, -1.3))
            @test gpd_outlier_cost(v, 10.0, ξ) ≈ log(2.0) - logpdf(GeneralizedPareto(0.0, -ξ * 10.0, ξ), v)
        end
        @test gpd_outlier_cost(3.0, 10.0, NaN) == log(20.0)
    end

    @testset "every outlier model and label prior is invariant to the residuals' units" begin
        for model in (ExtremeValueHalfwidth(), GeneralizedParetoOutliers(),
                      GeneralizedParetoOutliers(; fit_per = :hypothesis)),
            label_prior in (FlatLabelPrior(), ProfiledLabelPrior(), BetaLabelPrior()), seed in 1:3
            r = draw(Xoshiro(seed))
            n = length(r)
            s = StudentTUniformScoring(2, model; label_prior)
            k1 = score_residuals!(alloc_scoring_workspace(n, 1, s), zeros(n), s, r, n)[2]
            k2 = score_residuals!(alloc_scoring_workspace(n, 1, s), zeros(n), s, 1000 .* r, n)[2]
            @test k1 == k2
        end
    end

    @testset "OutlierTerm with a constant density is the supplied width" begin
        r = draw(Xoshiro(9))
        n = length(r)
        fixed = StudentTUniformScoring(2, 40.0)
        term = StudentTUniformScoring(2, OutlierTerm((k, z) -> (length(z) - k) * log(80.0)))
        @test score_residuals!(alloc_scoring_workspace(n, 1, term), zeros(n), term, r, n) ==
              score_residuals!(alloc_scoring_workspace(n, 1, fixed), zeros(n), fixed, r, n)
    end

    @testset "every outlier model scores without allocating, type-stably" begin
        r = draw(Xoshiro(10))
        n = length(r)
        q = zeros(n)
        for model in (40.0, ExtremeValueHalfwidth(), GeneralizedParetoOutliers(),
                      GeneralizedParetoOutliers(; fit_per = :hypothesis)),
            label_prior in (FlatLabelPrior(), BetaLabelPrior())
            s = StudentTUniformScoring(2, model; label_prior)
            sws = alloc_scoring_workspace(n, 1, s)
            score_residuals!(sws, q, s, r, n)
            @test (@allocated score_residuals!(sws, q, s, r, n)) == 0
            @test (@inferred score_residuals!(sws, q, s, r, n)) isa Tuple{Float64, Int}
        end
    end

    @testset "regressions: fallbacks, ties, non-finite residuals, empty problems, large pools" begin
        rules = TailRules()
        # a shape at or above the finite-shape limit falls back to the uniform support
        heavy = sort(rand(Xoshiro(21), GeneralizedPareto(0.0, 1.0, 0.3), 2000))
        f = tail_fit(rules, heavy)
        @test isnan(f.shape) && f.fitted_endpoint == -Inf && f.outlier_halfwidth == heavy[end]
        # tied maxima: the support lies beyond them, so every cost is finite
        z = [collect(0.1:0.1:5.0); 9.0; 9.0]
        sup = gpd_support((; fitted_endpoint = -Inf), z, 1, length(z))
        @test sup > 9.0 && isfinite(gpd_outlier_cost(9.0, sup, -0.5))
        # a non-finite residual is an outlier, never poisons the other partitions
        r = draw(Xoshiro(22))
        n = length(r)
        for model in (ExtremeValueHalfwidth(), GeneralizedParetoOutliers(), GeneralizedParetoOutliers(; fit_per = :hypothesis))
            s = StudentTUniformScoring(2, model)
            S0, k0 = score_residuals!(alloc_scoring_workspace(n, 1, s), zeros(n), s, r, n)
            rbad = [r; Inf]
            S1, k1 = score_residuals!(alloc_scoring_workspace(n + 1, 1, s), zeros(n + 1), s, rbad, n + 1)
            @test isfinite(S1) && 0 < k1 <= n
            # an empty problem scores no partition, as the supplied-width sweep does
            @test score_residuals!(alloc_scoring_workspace(0, 1, s), Float64[], s, Float64[], 0) == (-Inf, 0)
        end
        # OutlierTerm has no inlier threshold
        @test !VisualGeometryToolkitCore.has_inlier_threshold(StudentTUniformScoring(2, OutlierTerm((k, z) -> 0.0)))
        @test VisualGeometryToolkitCore.has_inlier_threshold(StudentTUniformScoring(2, ExtremeValueHalfwidth()))
        # a large pool's short tail: the suffix-sum moments match the textbook fit
        zl = sort(abs.(draw(Xoshiro(23); n_in = 3000, n_out = 7000)))
        T1, T2 = _suffix_sums!(zeros(length(zl) + 1), zeros(length(zl) + 1), zl)
        for lo in (length(zl) - 60, length(zl) - 200)
            @test _tail_fit(rules, zl, lo, T1, T2).outlier_halfwidth ≈ naive_tail(zl[lo:end], 0.9).outlier_halfwidth rtol = 1e-12
        end
    end

    @testset "label priors" begin
        n = 50
        @test _build_prior_table(n, FlatLabelPrior()) == Float64[]
        @test _build_prior_table(n, BetaLabelPrior(0.5, 0.5))[20] ≈ logbeta(20.5, 30.5) - logbeta(0.5, 0.5)
        @test_throws ArgumentError BetaLabelPrior(0.0, 1.0)
    end

    @testset "guards" begin
        @test_throws ArgumentError StudentTUniformScoring(2, ExtremeValueHalfwidth(); codimension = 2)
        @test_throws ArgumentError StudentTUniformScoring{Predictive}(2, ExtremeValueHalfwidth())
        @test_throws ArgumentError FixedHalfwidth(0.0)
        @test_throws ArgumentError ExtremeValueHalfwidth(; threshold_quantile = 1.0)
        @test_throws ArgumentError GeneralizedParetoOutliers(; fit_per = :pair)
        s = StudentTUniformScoring(2, ExtremeValueHalfwidth())
        @test outlier_halfwidth(s) === nothing              # the scoring contract: no supplied width
        @test_throws ArgumentError with_width(s, OutlierHalfwidth(1.0))
        fixed = StudentTUniformScoring(2, 1.0)
        @test_throws ArgumentError check_workspace(s, alloc_scoring_workspace(10, 1, fixed))
        @test check_workspace(s, alloc_scoring_workspace(10, 1, s)) === nothing
        @test outlier_model(fixed) === FixedHalfwidth(1.0) && label_prior(fixed) === FlatLabelPrior()
        @test StudentTUniformScoring(2, FixedHalfwidth(1.0)) === fixed
    end
end
