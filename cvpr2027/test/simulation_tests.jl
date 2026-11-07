# =============================================================================
# simulation_tests.jl — the project's own code (src/simulation.jl): the
# generative model of the draws, the block baselines scored through the
# package's OutlierTerm, the soft-label formulation, and the config mappings.
# The score itself is RANSACScoringDoneRight's and is tested there.
# =============================================================================
using Test
using StableRNGs: StableRNG
using RANSACScoringDoneRight: StudentTUniformScoring, ExtremeValueHalfwidth, GeneralizedParetoOutliers,
    FlatLabelPrior, ProfiledLabelPrior, BetaLabelPrior

@testset "simulation" begin
    uniform = Dict("n_in" => 300, "n_out" => 700, "outlier_halfwidth" => 100.0, "outliers" => "uniform")

    @testset "the draws" begin
        t = Trial(StableRNG(1), uniform)
        @test length(t.r) == 1000 && n_inliers(t) == 300
        @test issorted(t.z) && t.z == sort(abs.(t.r))
        @test maximum(abs, t.r[.!t.inlier]) <= 100.0
        mixed = Dict("n_in" => 30, "n_out" => 70, "outliers" => "mixed",
                     "outlier_halfwidth_min" => 10.0, "outlier_halfwidth_max" => 1000.0)
        @test 10.0 <= Trial(StableRNG(2), mixed).outlier_halfwidth <= 1000.0
    end

    @testset "config mappings" begin
        @test label_prior_of("half", (0.5, 0.5)) === FlatLabelPrior()
        @test label_prior_of("profile", (0.5, 0.5)) === ProfiledLabelPrior()
        @test label_prior_of("marginal", (1.0, 2.0)) === BetaLabelPrior(1.0, 2.0)
        pot = Dict("min_excess" => 5, "finite_shape_limit" => -1e-3)
        est = Dict("threshold_quantile" => 0.8)
        @test tail_model(est, pot; model = :uniform) isa ExtremeValueHalfwidth
        @test tail_model(est, pot; model = :gpd, fit_per = :hypothesis) isa GeneralizedParetoOutliers{:hypothesis}
        @test partition_metrics(320, 300, 300).f1 ≈ 600 / 620
    end

    @testset "estimators: separated data is labelled well, and units do not matter" begin
        t = Trial(StableRNG(3), uniform)
        c = 1000.0
        tc = Trial(c .* t.r, t.inlier, c * t.outlier_halfwidth, c .* t.z, t.block_rank)
        cache = WorkspaceCache()
        pot = Dict("min_excess" => 5, "finite_shape_limit" => -1e-3)
        soft = Dict("threshold_quantile" => 0.9, "log_sigma_range" => [-10.0, 3.0], "n_sigma" => 40, "n_gamma" => 25)
        for prior in ("half", "marginal")
            lp = label_prior_of(prior, (0.5, 0.5))
            s = StudentTUniformScoring(0, ExtremeValueHalfwidth(); label_prior = lp)
            k, tp = hard_labels(t, s, cache)
            @test partition_metrics(k, tp, 300).f1 > 0.9
            @test hard_labels(tc, s, cache)[1] == k
            b = hard_labels(t, StudentTUniformScoring(0, block_term(t, :max, 20); label_prior = lp), cache)
            @test b[1] == hard_labels(tc, StudentTUniformScoring(0, block_term(tc, :max, 20); label_prior = lp), cache)[1]
            for model in (:uniform, :gpd)
                ks, tps, _ = soft_labels(t; model, prior, beta = (0.5, 0.5), soft, pot)
                @test partition_metrics(ks, tps, 300).f1 > 0.9
                @test soft_labels(tc; model, prior, beta = (0.5, 0.5), soft, pot)[1] == ks
            end
        end
    end
end
