# =============================================================================
# evt_tests.jl — the extreme-value pieces of the score (src/evt.jl) against
# Distributions' GeneralizedPareto: the probability-weighted-moment fit
# recovers the shape and scale of its draws, the endpoint rule, the extended
# outlier model reduces to the uniform one at ξ = -1, and the sweep is
# invariant to the units of the residuals.
# =============================================================================
using Test
using Random: Xoshiro
using Distributions: GeneralizedPareto, logpdf, maximum as dmaximum

@testset "evt" begin
    @testset "gpd_pwm recovers Distributions' GeneralizedPareto" begin
        rng = Xoshiro(11)
        for ξ in (-1.0, -0.5, -0.25), ω in (1.0, 37.0)
            y = rand(rng, GeneralizedPareto(0.0, ω, ξ), 200_000)
            ξ̂, ω̂ = gpd_pwm(y)
            @test ξ̂ ≈ ξ atol = 0.02
            @test ω̂ ≈ ω rtol = 0.02
        end
    end

    @testset "pot_endpoint: the endpoint of a uniform, and its rules" begin
        z = sort(rand(Xoshiro(3), 50_000) .* 100)
        fit = pot_endpoint(z, 0.9; min_excess = 5, xi_finite = -1e-3)
        @test fit.ξ ≈ -1 atol = 0.05
        @test fit.a ≈ 100 rtol = 0.01
        @test fit.a >= maximum(z)
        few = pot_endpoint(z[1:20], 0.9; min_excess = 5, xi_finite = -1e-3)
        @test isnan(few.ξ) && few.a == maximum(z[1:20])
    end

    @testset "the extended model is the uniform one at ξ = -1" begin
        z = [0.5, 10.0, 60.0, 99.0]
        @test outlier_loglik(:gpd, z, 100.0, -1.0) ≈ outlier_loglik(:uniform, z, 100.0, NaN)
        # and its density is Distributions' GPD from zero, halved for the sign
        g = GeneralizedPareto(0.0, 0.5 * 100.0, -0.5)
        @test outlier_loglik(:gpd, z, 100.0, -0.5) ≈ sum(logpdf(g, v) - log(2) for v in z)
        @test dmaximum(g) ≈ 100.0
        @test outlier_loglik(:gpd, [100.0], 100.0, -0.5) == -Inf
    end

    @testset "label prior: marginal and profiled differ by a near-constant" begin
        N = 1000
        δ = [label_prior_term(:marginal, k, N) - label_prior_term(:profile, k, N) for k in (30, 300, 500)]
        @test maximum(δ) - minimum(δ) < 0.05
        @test label_prior_term(:half, 300, N) == 0
    end

    @testset "soft labels: invariant to units, near-perfect on separated data" begin
        pot = Dict("min_excess" => 5, "xi_finite" => -1e-3)
        soft = Dict("eta" => 0.9, "log_sigma_range" => [-10.0, 3.0], "n_sigma" => 40, "n_gamma" => 25)
        case = Dict("n_in" => 300, "n_out" => 700, "A" => 100.0, "outliers" => "uniform")
        rng = Xoshiro(17)
        for model in (:uniform, :gpd), prior in (:half, :marginal)
            d, inl, _ = draw_case(rng, case)
            r1 = soft_mixture(d, inl; model, prior, beta = (0.5, 0.5), soft, pot)
            r2 = soft_mixture(1000 .* d, inl; model, prior, beta = (0.5, 0.5), soft, pot)
            @test r1[1] == r2[1]
            @test partition_metrics(r1[1], r1[2], 300).f1 > 0.9
        end
    end

    @testset "the sweep is invariant to the units of the residuals" begin
        rng = Xoshiro(5)
        pot = Dict("min_excess" => 5, "xi_finite" => -1e-3)
        est = Dict("kind" => "pot", "eta" => 0.9)
        case = Dict("n_in" => 300, "n_out" => 700, "A" => 100.0, "outliers" => "uniform")
        for _ in 1:5, model in (:uniform, :gpd)
            d, inl, A = draw_case(rng, case)
            keys = rand(rng, length(d))
            kw = (; model, kmin = 3, kmax_margin = 10, keys, pot)
            @test sweep(d, inl, est; true_halfwidth = A, kw...)[1] ==
                  sweep(1000 .* d, inl, est; true_halfwidth = 1000A, kw...)[1]
        end
    end
end
