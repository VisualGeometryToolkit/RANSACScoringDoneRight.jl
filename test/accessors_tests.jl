# Accessors for the structs downstream code used to read by field name, and
# `resize!(::ScoringWorkspace, n)`, which replaced VisualGeometryToolkitEstimation's
# `_set_n!` reaching into seven workspace fields.

@testitem "accessors" begin
    using Test
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: sweep_order, penalties, sort_key, lg_table,
        alloc_scoring_workspace, default_scoring_workspace, _build_prior_table
    using VisualGeometryToolkitCore: IGPrior, outlier_halfwidth, scale_prior, minimal_sample_size, codimension,
        score!, inlier_count, consensus_order, partition_rss

    @testset "accessors" begin
        prior = IGPrior(1.5, 0.25)
        s = StudentTUniformScoring(4, 3.0; prior, codimension = 2)
        @test outlier_halfwidth(s) === s.outlier_halfwidth === 3.0
        @test minimal_sample_size(s) === s.model_dof === 4
        @test scale_prior(s) === s.scale_prior === prior
        @test codimension(s) === 2

        sws = alloc_scoring_workspace(20, 2, s)
        @test sweep_order(sws) === sws.perm
        @test penalties(sws) === sws.penalties
        @test sort_key(sws) === sws.sort_key
        @test lg_table(sws) === sws.lg_table
        @test scale_prior(sws) === prior

        # the scoring contract reads the workspace through its accessors
        n = 12
        r = sqrt.(Float64[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 9.0, 16.0, 25.0, 36.0])
        s1 = StudentTUniformScoring(4, 10.0)
        W = alloc_scoring_workspace(n, 1, s1)
        score!(W, s1, zeros(n), r, n)
        @test inlier_count(s1, W) === W.k
        @test consensus_order(s1, W) === sweep_order(W)
    end

    @testset "resize!(::ScoringWorkspace, n) == a fresh workspace for n" begin
        dg, m, halfwidth = 1, 4, 2.0
        r = Float64[0.01, -0.03, 0.02, 0.05, -0.04, 0.06, 0.015, -0.025,
                    1.3, -1.7, 0.9, 1.9, -1.1, 0.7, 0.012, -0.018, 2.2, 0.03]
        for profile_prior in (false, true), prior in (IGPrior(), IGPrior(1.5, 0.01))
            s = StudentTUniformScoring(m, halfwidth; prior, profile_prior)
            big = alloc_scoring_workspace(40, dg, s)
            for n in (length(r), 9, length(r))            # trim, trim further, regrow
                @test resize!(big, n) === big
                @test length(sweep_order(big)) == length(big.scratch) == n
                @test length(penalties(big)) == length(sort_key(big)) == n
                @test all(iszero, penalties(big))
                @test length(big.radix.keys) == length(big.radix.keys2) ==
                      length(big.radix.perm2) == n
                # the profiled-γ table is REBUILT for n, not truncated from N = 40
                @test big.prior_table == (profile_prior ? _build_prior_table(n) : Float64[])

                fresh = alloc_scoring_workspace(n, dg, s)
                got  = score_residuals!(big,   zeros(n), s, r[1:n], n)
                want = score_residuals!(fresh, zeros(n), s, r[1:n], n)
                @test got === want
                @test sweep_order(big) == sweep_order(fresh)
            end
        end
        # the table cannot outgrow its build size: the workspace does not record dg
        sws = default_scoring_workspace(10)
        @test_throws ArgumentError resize!(sws, 11)
        @test_throws ArgumentError resize!(sws, -1)
        # n = 0 keeps a profiled table (its non-emptiness IS the profile flag)
        sp = alloc_scoring_workspace(10, 1, StudentTUniformScoring(4, 2.0; profile_prior = true))
        resize!(sp, 0)
        @test !isempty(sp.prior_table)
        resize!(sp, 7)
        @test sp.prior_table == _build_prior_table(7)
    end

    @testset "partition_rss: the swept partition's RSS from plain residuals" begin
        n, m = 14, 4
        r = Float32[0.01, -0.03, 0.02, 0.05, -0.04, 0.06, 0.015, -0.025, 1.3, -1.7, 0.9, 1.9, -1.1, 0.7]
        s = StudentTUniformScoring(m, 2.0)
        sws = alloc_scoring_workspace(n, 1, s)
        _, k = score_residuals!(sws, zeros(n), s, r, n)
        @test m < k < n
        # the loop VisualGeometryToolkitEstimation's seed partition hand-rolled, verbatim
        RSS = 0.0
        for j in 1:k
            RSS += Float64(r[sweep_order(sws)[j]])^2
        end
        @test partition_rss(sweep_order(sws), r, k) === RSS
        @test partition_rss(sweep_order(sws), r, k) ≈ sum(abs2, Float64.(r[sweep_order(sws)[1:k]])) rtol = 1e-14
        @test partition_rss(sweep_order(sws), r, 0) === 0.0
    end
end
