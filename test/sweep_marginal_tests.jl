# `sweep!` — the per-hypothesis sort-and-sweep — against a brute force of its
# own partition score S(θ, I_k) over every prefix k, profiled-γ term included.
#
# (This file used to test `sweep_marginal!`, the logsumexp over the same nested
#  partitions, against the brute force; that soft sweep and the `:marginal_soft`
#  grid it served are deleted — no caller anywhere. What stays pinned is the
#  hard sweep, including the noise-free case where the running RSS sits below
#  its eps floor: the floor must clamp a COPY, so the best score is the brute
#  force's maximum exactly.)

@testitem "sweep_marginal" begin
    using VisualGeometryToolkitCore: StudentTUniformScoring, FlatLabelPrior, ProfiledLabelPrior, BetaLabelPrior, label_prior, alloc_scoring_workspace, sweep!, partition_score, score_residuals!, JeffreysPrior, IGPrior   # the Student-t scorer (Core's)
    using Test
    using RANSACScoringDoneRight
    using VisualGeometryToolkitCore: IGPrior

    function _brute_Sk(r, model_dof, dg, halfwidth, prior, sws, label_prior)
        n = length(r)
        sq = sort(r .^ 2)
        Sk = Float64[]
        rss = 0.0
        for k in 1:n
            rss += sq[k]
            k <= model_dof && continue
            S = partition_score(prior, sws.lg_table[k - model_dof], dg * (k - model_dof) / 2,
                             max(rss, eps()), (n - k) * dg * log(2 * halfwidth))
            label_prior isa FlatLabelPrior || (S += sws.prior_table[k])
            push!(Sk, S)
        end
        return Sk
    end

    @testset "sweep! == the brute-force max of its partition score" begin
        model_dof, dg, halfwidth = 4, 1, 2.0
        r = Float64[0.01, -0.03, 0.02, 0.05, -0.04, 0.06, 0.015, -0.025,
                    1.3, -1.7, 0.9, 1.9, -1.1, 0.7]
        # NOISE-FREE: after the sample points every inlier has r² < eps, so the
        # running RSS sits below the floor for many partitions.
        r0 = [zeros(model_dof); 1e-10 .* (1:14); [1.3, -1.7, 0.9, 1.9, -1.1, 0.7]]
        @test maximum(abs2, r0[1:model_dof+14]) < eps()
        for data in (r, r0), label_prior in (FlatLabelPrior(), ProfiledLabelPrior(), BetaLabelPrior(0.5, 0.5)),
            prior in (IGPrior(), IGPrior(1.5, 0.01))
            n = length(data)
            s = StudentTUniformScoring(model_dof, halfwidth; scale_prior = prior, label_prior)
            sws = alloc_scoring_workspace(n, dg, s)
            S_hard, k_hard = score_residuals!(sws, zeros(n), s, data, n)
            Sk = _brute_Sk(data, model_dof, dg, halfwidth, prior, sws, label_prior)
            @test S_hard ≈ maximum(Sk) rtol = 1e-14
            @test k_hard == model_dof + argmax(Sk)
        end

        # The profiled prior changes the score.
        n = length(r)
        s0 = StudentTUniformScoring(model_dof, halfwidth; label_prior = FlatLabelPrior(), scale_prior = JeffreysPrior())
        s1 = StudentTUniformScoring(model_dof, halfwidth; label_prior = ProfiledLabelPrior(), scale_prior = JeffreysPrior())
        S0, _ = score_residuals!(alloc_scoring_workspace(n, dg, s0), zeros(n), s0, r, n)
        S1, _ = score_residuals!(alloc_scoring_workspace(n, dg, s1), zeros(n), s1, r, n)
        @test S0 != S1
    end
end
