# The Student-t + uniform scorer's consensus set as a seed: `score!` sweeps,
# `consensus_order`'s first `inlier_count` entries ARE the swept partition, and
# VisualGeometryCore's toolkit seeds from it — `PosteriorWeightState(…; order,
# prior)`, `posterior_weights!`/`posterior_inlier_probs!`, `consensus_s2` — under
# the scorer's OWN scale prior (the one the sweep scored the partition under).
# The seeds themselves are pinned in VisualGeometryCore (fit_scale_tests.jl).

@testitem "consensus_seed" begin
    using Test
    using Random: MersenneTwister
    import VisualGeometryCore as VGC
    using VisualGeometryCore: IGPrior, score!, inlier_count, consensus_order, consensus_s2, scale_prior,
        partition_student_t, partition_rss, PosteriorWeightState, s_hat_sq, student_t_dof, sq_residuals,
        inlier_prior, posterior_weights!, posterior_inlier_probs!, inlier_threshold, ScoreBuffers,
        GauInlier, decision_threshold, check_workspace, with_width, OutlierHalfwidth, outlier_halfwidth
    using RansacScoringDoneRight
    using RansacScoringDoneRight: alloc_scoring_workspace, sweep_order

    rng = MersenneTwister(7)
    n_in, n_out = 80, 20
    # codimension 2: residual NORMS — Rayleigh inliers at σ = 0.5, uniform gross
    # outliers out to the half-width.
    halfwidth = 40.0
    r = vcat([0.5 * hypot(randn(rng), randn(rng)) for _ in 1:n_in], halfwidth .* rand(rng, n_out))
    n = length(r)

    @testset "the seed reads the scorer's partition under its prior" begin
        for prior in (IGPrior(0.0, 0.0), IGPrior(1.0, 0.25))
            s = StudentTUniformScoring(4, halfwidth; prior, codimension = 2)
            @test scale_prior(s) === prior
            W = VGC.scoring_workspace(s, n, 2)
            q = Vector{Float64}(undef, n)
            S = score!(W, s, q, r, n)
            k = inlier_count(s, W)
            order = consensus_order(s, W)
            @test isfinite(S) && order === sweep_order(W)
            @test q == r .^ 2
            model_dof = 5
            @test model_dof < k < n
            # the partition is the inliers: every gross outlier past 10σ is out
            @test all(order[j] <= n_in || r[order[j]] < 5.0 for j in 1:k)

            pws = PosteriorWeightState(q, n, model_dof, k; order, prior = scale_prior(s), w = zeros(n),
                                       inlier_posterior = zeros(n), sq_r = zeros(n),
                                       log_two_outlier_halfwidth = log(2 * halfwidth), codimension = 2)
            ν, s2 = partition_student_t(prior, partition_rss(order, r, k), k; model_dof, codimension = 2)
            @test s_hat_sq(pws) === s2 && student_t_dof(pws) === ν
            @test sq_residuals(pws) == r .^ 2
            @test inlier_prior(pws) === 0.5
            @test consensus_s2(s, r, order, k; model_dof, d_g = 2) === s2

            w = zeros(n); p = zeros(n)
            kw = (; model_dof, log_two_outlier_halfwidth = log(2 * halfwidth), codimension = 2, prior)
            posterior_weights!(w, q, order, k; kw...)
            posterior_inlier_probs!(p, q, order, k; kw...)
            @test w ≈ pws.w rtol = 1e-12
            @test p ≈ pws.inlier_posterior rtol = 1e-12
        end
    end

    @testset "inlier_threshold: the consensus Student-t's crossover" begin
        s = StudentTUniformScoring(4, halfwidth; codimension = 2)
        W = VGC.scoring_workspace(s, n, 2)
        buf = ScoreBuffers{Float64}(n, W)
        buf.residuals .= r
        score!(W, s, buf.scores, buf.residuals, n)
        k = inlier_count(s, W)
        s2 = consensus_s2(s, r, consensus_order(s, W), k; model_dof = 5, d_g = 2)
        @test inlier_threshold(s, buf) === decision_threshold(GauInlier(s2; d_g = 2), halfwidth)
        @test 0 < inlier_threshold(s, buf) < 5.0
    end

    @testset "check_workspace and with_width" begin
        s = StudentTUniformScoring(4, halfwidth; codimension = 2)
        @test check_workspace(s, VGC.scoring_workspace(s, n, 2)) === nothing
        @test_throws ArgumentError check_workspace(StudentTUniformScoring(4, halfwidth; codimension = 2,
                                                   prior = IGPrior(1.0, 0.25)), VGC.scoring_workspace(s, n, 2))
        @test_throws ArgumentError check_workspace(StudentTUniformScoring(4, halfwidth; codimension = 2,
                                                   profile_prior = true), VGC.scoring_workspace(s, n, 2))
        s2 = with_width(s, OutlierHalfwidth(3.0))
        @test s2 isa typeof(s) && outlier_halfwidth(s2) === 3.0
        @test VGC.minimal_sample_size(s2) == 4 && VGC.codimension(s2) == 2
    end
end
