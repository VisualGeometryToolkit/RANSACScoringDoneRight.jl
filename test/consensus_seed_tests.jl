# R22: contract test — the scoring contract (`scoring_workspace`, `score!`) and pool selection (`select_on_grid!`) of this package's scorers
# The Student-t + uniform scorer's consensus set as a seed: `score!` sweeps,
# `consensus_order`'s first `inlier_count` entries ARE the swept partition, and
# VisualGeometryToolkitCore's toolkit seeds from it — `PosteriorWeightState(…; order,
# prior)`, `consensus_s2` and the scoring contract's `inlier_probabilities!` — under
# the scorer's OWN scale prior (the one the sweep scored the partition under).
# Every check goes through Core's PUBLIC entries (the partition's Student-t is
# `ScaleMAP` for ŝ² and the documented ν rule, its mixture `StudentTMixture`);
# the seeds themselves are pinned in VisualGeometryToolkitCore (its ScaleMAP tests).

@testitem "consensus_seed" begin
    using VisualGeometryToolkitCore: StudentTUniformScoring, FlatLabelPrior, ProfiledLabelPrior, label_prior, sweep_order, JeffreysPrior, IGPrior   # the Student-t scorer (Core's)
    using Test
    using Random: MersenneTwister
    import VisualGeometryToolkitCore as VGC
    using VisualGeometryToolkitCore: IGPrior, score!, inlier_count, consensus_order, consensus_s2, scale_prior,
        partition_rss, PosteriorWeightState, s_hat_sq, student_t_dof, sq_distances,
        inlier_prior, inlier_probabilities!, StudentTMixture, posterior_weight, inlier_posterior,
        EstimationProblem, SquaredScale, ScaleMAP, solve, decision_boundary, ScoreBuffers,
        GauInlier, check_workspace, with_width, OutlierHalfwidth, outlier_halfwidth
    using RANSACScoringDoneRight

    rng = MersenneTwister(7)
    n_in, n_out = 80, 20
    # codimension 2: residual NORMS — Rayleigh inliers at σ = 0.5, uniform gross
    # outliers out to the half-width.
    halfwidth = 40.0
    r = vcat([0.5 * hypot(randn(rng), randn(rng)) for _ in 1:n_in], halfwidth .* rand(rng, n_out))
    n = length(r)

    @testset "the seed reads the scorer's partition under its prior" begin
        for prior in (IGPrior(0.0, 0.0), IGPrior(1.0, 0.25))
            s = StudentTUniformScoring(4, halfwidth; scale_prior = prior, codimension = 2, label_prior = FlatLabelPrior())
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
            # the partition's Student-t: ŝ² the inverse-gamma MAP of its RSS over
            # ν_data = (k − model_dof)·d_g components, ν = max(ν_data + 2α₀, 2)
            ν_data = (k - model_dof) * 2
            s2 = solve(EstimationProblem(SquaredScale, (; rss = partition_rss(order, r, k), dof = ν_data)), ScaleMAP(prior)).u
            ν = max(ν_data + 2 * prior.alpha0, 2.0)
            @test s_hat_sq(pws) === s2 && student_t_dof(pws) === ν
            @test sq_distances(pws) == r .^ 2
            @test inlier_prior(pws) === 0.5
            @test consensus_s2(s, r, order, k; model_dof, d_g = 2) === s2

            # the seed's first E-step is the partition's Student-t + uniform mixture
            m = StudentTMixture(ν, s2, log(2 * halfwidth), 2)
            w = [posterior_weight(m, q[i]) for i in 1:n]
            p = [inlier_posterior(m, q[i]) for i in 1:n]
            @test w ≈ pws.w rtol = 1e-12
            @test p ≈ pws.inlier_posterior rtol = 1e-12
            # and the scoring contract's per-point inlier probabilities of the consensus set
            pc = inlier_probabilities!(zeros(n), s, q, order, k; model_dof, d_g = 2)
            @test pc ≈ pws.inlier_posterior rtol = 1e-12
        end
    end

    @testset "decision_boundary: the consensus Student-t's crossover" begin
        s = StudentTUniformScoring(4, halfwidth; codimension = 2, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
        W = VGC.scoring_workspace(s, n, 2)
        buf = ScoreBuffers{Float64}(n, W)
        buf.residuals .= r
        score!(W, s, buf.scores, buf.residuals, n)
        k = inlier_count(s, W)
        s2 = consensus_s2(s, r, consensus_order(s, W), k; model_dof = 5, d_g = 2)
        @test decision_boundary(s, buf) === decision_boundary(GauInlier(s2; d_g = 2), halfwidth)
        @test 0 < decision_boundary(s, buf) < 5.0
    end

    @testset "check_workspace and with_width" begin
        s = StudentTUniformScoring(4, halfwidth; codimension = 2, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
        @test check_workspace(s, VGC.scoring_workspace(s, n, 2)) === nothing
        @test_throws ArgumentError check_workspace(StudentTUniformScoring(4, halfwidth; codimension = 2,
                                                   scale_prior = IGPrior(1.0, 0.25), label_prior = FlatLabelPrior()), VGC.scoring_workspace(s, n, 2))
        @test_throws ArgumentError check_workspace(StudentTUniformScoring(4, halfwidth; codimension = 2,
                                                   label_prior = ProfiledLabelPrior(), scale_prior = JeffreysPrior()), VGC.scoring_workspace(s, n, 2))
        s2 = with_width(s, OutlierHalfwidth(3.0))
        @test s2 isa typeof(s) && outlier_halfwidth(s2) === 3.0
        @test VGC.minimal_sample_size(s2) == 4 && VGC.codimension(s2) == 2
    end
end
