# The scorer fact a grid reads — the width axis its grid is native on
# (VisualGeometryCore's `width_axis(scoring)`, with this package's methods for
# its grid families) — the scale multiple → native maps every grid of these
# families is cut with (`to_native`, `native_map`: VisualGeometryCore's since
# K+, pinned here on the Student-t marginal's crossover they serve), and the
# marginal's method of the one grid verb against the per-pair selection.

@testitem "native_widths" setup = [PoolFixtures] begin
    using Test, Random
    using KernelAbstractions
    using VisualGeometryCore: LossWidth, OutlierHalfwidth, ScaleMultiple, to_native, native_map, decision_width,
        WelschShape, GauInlier, decision_threshold, IGPrior, RansacScoring, MsacScoring, width_axis,
        best_pool_model
    using RansacScoringDoneRight
    import VisualGeometryCore
    using RansacScoringDoneRight: LossSpec, loss_spec, stage_ig_shared_from_pairs

    @testset "to_native(::OutlierHalfwidth): the mixture's crossover" begin
        for d_g in (1, 2), scale in (0.01, 1.0, 3.0), k in (0.8, 2.0, 6.0)
            h = to_native(OutlierHalfwidth(), k, scale; d_g)
            @test h == decision_width(GauInlier(scale^2; d_g), k * scale)
            # the MAP rule of that mixture flips at the residual k·scale
            @test decision_threshold(GauInlier(scale^2; d_g), h) ≈ k * scale rtol = 1e-10
        end
        # d_g is the outlier density's exponent: it moves the native width by decades
        @test to_native(OutlierHalfwidth(), 6.0, 1.0; d_g = 1) ≈ 8.2293e7 rtol = 1e-4
        @test to_native(OutlierHalfwidth(), 6.0, 1.0; d_g = 2) ≈ 1.0156e4 rtol = 1e-4
        @test_throws UndefKeywordError to_native(OutlierHalfwidth(), 2.0, 1.0)
        # the loss-width axis is VisualGeometryCore's: the truncation flips at its width
        @test to_native(LossWidth(), 2.5, 0.4; d_g = 1) == 2.5 * 0.4
    end

    @testset "native_map: the curried to_native" begin
        for axis in (OutlierHalfwidth(), LossWidth()), d_g in (1, 2)
            f = native_map(axis; d_g)
            @test f(3.0, 0.2) == to_native(axis, 3.0, 0.2; d_g)
        end
        @test_throws MethodError native_map(ScaleMultiple(); d_g = 1)
    end

    @testset "width_axis of the grid families" begin
        m = StudentTUniformScoring(5, 0.1; codimension = 1)
        spec = loss_spec(WelschShape(), IGPrior(); Δ = 0.5, t_lo = -10.0, t_hi = 0.0, NG = 33)
        @test width_axis(m) === OutlierHalfwidth()
        @test width_axis(spec) === OutlierHalfwidth()
        for sc in (RansacScoring(), MsacScoring(2), MlesacScoring(), MagsacScoring(),
                   WelschCostScoring(), SmoothTLSScoring(), GemanScoring())
            @test width_axis(sc) === LossWidth()
        end
        @test width_axis(WelschMarginal(; prior_loss_width = 0.1, outlier_halfwidth = 1.0)) === LossWidth()
        @test width_axis(SimFitPP()) === LossWidth()
        # the axis is the one `with_width` rebuilds the marginal on
        @test VisualGeometryCore.with_width(m, OutlierHalfwidth(0.7)) == StudentTUniformScoring(5, 0.7; codimension = 1)
        # the retired per-type trait is gone, not aliased
        @test !isdefined(RansacScoringDoneRight, Symbol("native", "_axis"))   # in fragments: a retired name
    end

    @testset "grid_evaluate(::StudentTUniformScoring) selects what best_pool_model selects" begin
        rng = MersenneTwister(3)
        pool = map(1:6) do _
            n, M = 90, 7
            D = mixture_residuals(rng, n, M)
            S = reduce(hcat, [sort(randperm(rng, n)[1:5]) for _ in 1:M])
            (identity_pool(D; sample_indices = S, errors_r = 10 .* rand(rng, M), errors_t = zeros(M)), D)
        end
        sh = stage_ig_shared_from_pairs(first.(pool), last.(pool); backend = KernelAbstractions.CPU())
        hw = [0.5, 2.0, 8.0, 40.0]
        for prior in (IGPrior(), IGPrior(2.0, 0.05)), dg in (1, 2)
            sc = StudentTUniformScoring(5, 1.0; prior, codimension = dg)
            errs, idxs = grid_evaluate(sc, sh, hw; return_idxs = true)
            ref = [best_pool_model(StudentTUniformScoring(5, h; prior, codimension = dg), p, D) for (p, D) in pool, h in hw]
            @test idxs == ref
            @test all(!iszero, ref)
            @test errs == [pool[i][1].errors_r[ref[i, t]] for i in eachindex(pool), t in eachindex(hw)]
        end
        # what the device sweep cannot score is refused, never scored as something else
        @test_throws ArgumentError grid_evaluate(StudentTUniformScoring{Predictive}(5, 1.0; codimension = 1), sh, hw)
        @test_throws ArgumentError grid_evaluate(StudentTUniformScoring(5, 1.0; codimension = 1, profile_prior = true), sh, hw)
        # the mixture marginal needs the problem's minimal sample size and codimension
        spec = loss_spec(WelschShape(), IGPrior(); Δ = 0.5, t_lo = -10.0, t_hi = 0.0, NG = 33)
        @test_throws UndefKeywordError grid_evaluate(spec, sh, hw)
    end
end
