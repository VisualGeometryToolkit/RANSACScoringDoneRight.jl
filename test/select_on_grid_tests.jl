# The grid families on VisualGeometryCore's pool-selection contract
# (src/selection.jl): every family stages through the one `:rsdr` staging, and
# `select_on_grid!` is THE grid verb's selection (`grid_evaluate(…;
# return_idxs = true)`) on that staging — bit-identical to the evaluation
# engine's former per-scoring dispatch. SIMFIT++ is its host loop over the
# pairs' pixel residuals with the caller's per-(pair, column) generator. And
# VisualGeometryCore's generic CPU fallback (per column `with_width` then the
# generic `best_pool_model`) on its `:pool` staging selects exactly what the
# Exact grid of the width-free family selects — the path of VisualGeometryCore's
# threshold scorers (MSAC/RANSAC at a threshold are no grid family).

@testitem "select_on_grid" setup = [PoolFixtures] begin
    using Test, Random, LinearAlgebra
    using StaticArrays
    using KernelAbstractions
    using RansacScoringDoneRight
    using RansacScoringDoneRight: stage_ig_shared_from_pairs, loss_spec, Exact, LossSpec, pair_chunk_budget,
        simfitpp_select, pixel_residuals, residual_unit_scale
    using VisualGeometryCore: AbstractConstraintProblem, RansacScoring, MsacScoring, IGPrior, WelschShape,
        FundMat, sampson_distance, select_on_grid!, selection_staging, stage_selection
    using VisualGeometryCore.Pools: PoolPair
    import VisualGeometryCore as VGC

    struct _SGProblem <: AbstractConstraintProblem end
    VGC.codimension(::Type{_SGProblem}) = 1
    VGC.minimal_sample_size(::Type{_SGProblem}) = 5

    const be = KernelAbstractions.CPU()

    function _sg_pool(rng, P)
        map(1:P) do _
            n, M = rand(rng, 40:120), rand(rng, 1:9)
            D = mixture_residuals(rng, n, M)
            S = reduce(hcat, [sort(randperm(rng, n)[1:5]) for _ in 1:M])
            (identity_pool(D; sample_indices = S, errors_r = 10 .* rand(rng, M), errors_t = zeros(M)), D)
        end
    end

    # select_on_grid! through the contract: the family's staging key, its staged
    # batch, the selections.
    function _select(sc, grid, pairs, Rs; kw...)
        st = stage_selection(selection_staging(sc), pairs, Rs; backend = be)
        idxs = fill(Int32(-1), length(pairs), length(grid))
        out = select_on_grid!(idxs, _SGProblem, sc, grid, st; kw...)
        @test out === idxs
        return idxs
    end

    rng = MersenneTwister(20261009)
    pool = _sg_pool(rng, 9)
    pairs, Rs = first.(pool), last.(pool)
    sh = stage_ig_shared_from_pairs(pairs, Rs; backend = be)

    @testset "the one :rsdr staging" begin
        fams = (StudentTUniformScoring(5, 1.0; codimension = 1), MsacScoring(), RansacScoring(), MlesacScoring(),
                MagsacScoring(), WelschCostScoring(), SmoothTLSScoring(), GemanScoring(),
                loss_spec(WelschShape(), IGPrior(); Δ = 0.5, t_lo = -10.0, t_hi = 0.0, NG = 33),
                WelschMarginal(; prior_loss_width = 0.1, outlier_halfwidth = 1.0), SimFitPP())
        @test all(sc -> selection_staging(sc) === :rsdr, fams)
        st = stage_selection(:rsdr, pairs, Rs; backend = be)
        @test st.pairs === pairs
        @test st.staged.K_per_pair == sh.K_per_pair && st.staged.sorted_sq_flat == sh.sorted_sq_flat
    end

    @testset "each grid family ≡ grid_evaluate's selections (bitwise)" begin
        hw = [0.5, 2.0, 8.0, 40.0]
        lw = [0.3, 1.0, 3.0, 8.0]
        for prior in (IGPrior(), IGPrior(2.0, 0.05)), dg in (1, 2)
            sc = StudentTUniformScoring(5, 1.0; prior, codimension = dg)
            @test _select(sc, hw, pairs, Rs) == last(grid_evaluate(sc, sh, hw; return_idxs = true))
        end
        for sc in (MsacScoring(), RansacScoring(), MlesacScoring(), MagsacScoring(),
                   WelschCostScoring(), SmoothTLSScoring(), GemanScoring())
            @test _select(sc, lw, pairs, Rs) == last(grid_evaluate(sc, sh, lw; return_idxs = true))
        end
        # a threshold scorer is VisualGeometryCore's, not a grid family: it
        # stages `:pool` and selects through the generic fallback (below)
        for sc in (MsacScoring(1; threshold = 0.7), RansacScoring(2; threshold = 0.7))
            @test selection_staging(sc) === :pool
        end
        spec = loss_spec(WelschShape(), IGPrior(); Δ = 0.5, t_lo = -10.0, t_hi = 0.0, NG = 33)
        @test _select(spec, hw, pairs, Rs) ==
              last(grid_evaluate(Float64, spec, sh, hw; model_dof = 5, dg = 1, return_idxs = true))
        # the mixture marginal reads the problem's minimal sample size and codimension
        st = stage_selection(:rsdr, pairs, Rs; backend = be)
        @test_throws MethodError select_on_grid!(Matrix{Int32}(undef, length(pairs), 4), spec, hw, st)
        m = WelschMarginal(; prior_loss_width = 0.1, outlier_halfwidth = 1.0)
        @test _select(m, [0.05, 0.2, 1.0], pairs, Rs) == last(grid_evaluate(m, sh, [0.05, 0.2, 1.0]; return_idxs = true))
    end

    @testset "pair_chunk_budget" begin
        @test pair_chunk_budget(10, 10) == 256
        @test pair_chunk_budget(5500, 600) == clamp(fld(6_000_000_000, 5500 * 600 * 16), 1, 256)
        @test pair_chunk_budget(10^9, 10^3) == 1
        @test pair_chunk_budget(0, 0) == 256
    end

    @testset "SIMFIT++: the host loop over pixel residuals, one generator per (pair, column)" begin
        # pool-frame pairs whose maps are similarities: the pixel residuals differ
        # from the stored ones, and the initial guess crosses frames
        sim(s, tx, ty) = SMatrix{3,3,Float64,9}(s, 0.0, 0.0, 0.0, s, 0.0, tx, ty, 1.0)
        Kc = SMatrix{3,3,Float64,9}(400.0, 0.0, 0.0, 0.0, 400.0, 0.0, 320.0, 240.0, 1.0)
        I3s = SMatrix{3,3,Float64,9}(I)
        r = MersenneTwister(77)
        sp = map(1:4) do k
            n, K = 120 + 20k, 6
            csponds = [SVector(2rand(r) - 1, 2rand(r) - 1) => SVector(2rand(r) - 1, 2rand(r) - 1) for _ in 1:n]
            models = [FundMat(Tuple(SMatrix{3,3,Float64,9}(randn(r, 9)...))) for _ in 1:K]
            D = Float32[sampson_distance(models[j], csponds[i].first, csponds[i].second) for i in 1:n, j in 1:K]
            S = reduce(hcat, [sort(randperm(r, n)[1:7]) for _ in 1:K])
            s = 1 / (300.0 + 100k)
            pair = PoolPair(; csponds, K1 = Kc, K2 = Kc, R_gt = I3s, t_gt = SVector(1.0, 0.0, 0.0), models,
                            sample_indices = S, errors_r = fill(NaN, K), errors_t = fill(NaN, K),
                            gt_inliers = falses(n), gt_variance = 1.0e-4 * k,
                            source_map = sim(s, -0.9, -0.7), target_map = sim(s, -0.95, -0.65))
            (pair, D)
        end
        spairs, sRs = first.(sp), last.(sp)
        grid = [0.002, 0.01, 0.05]
        cfg = SimFitPP()
        rng_of(fp, t) = MersenneTwister(hash((fp, t)))
        got = _select(cfg, grid, spairs, sRs; rng_of)
        want = [Int32(first(simfitpp_select(pixel_residuals(p), grid[t] / residual_unit_scale(p), cfg;
                                            rng = rng_of((VGC.Pools.num_points(p), VGC.Pools.num_models(p), p.gt_variance), t))))
                for p in spairs, t in eachindex(grid)]
        @test got == want
        @test all(j -> 0 <= j <= 6, got)
        # the generator is the caller's: no default
        st = stage_selection(:rsdr, spairs, sRs; backend = be)
        @test_throws UndefKeywordError select_on_grid!(Matrix{Int32}(undef, 4, 3), _SGProblem, cfg, grid, st)
        # thread-schedule independent: a rerun is identical
        @test _select(cfg, grid, spairs, sRs; rng_of) == got
    end

    @testset "VisualGeometryCore's generic fallback ≡ the Exact grid of the width-free family" begin
        # The generic method on the `:pool` staging: the threshold scorers'
        # own path; the marginal's through `invoke` (it is a grid family).
        generic!(idxs, sc, grid, st) =
            invoke(select_on_grid!, Tuple{AbstractMatrix{Int32}, Type, Any, AbstractVector, Any}, idxs, _SGProblem, sc, grid, st)
        st = stage_selection(:pool, pairs, Rs)
        @test st == (; pairs, residuals = Rs)
        lw = [0.05, 0.3, 1.0, 3.0, 8.0, 25.0]
        for dg in (1, 2), (sc, wf) in ((MsacScoring(dg; threshold = 1.0), MsacScoring(dg)),
                                       (RansacScoring(dg; threshold = 1.0), RansacScoring(dg)))
            idxs = fill(Int32(-1), length(pairs), length(lw))
            @test _select(sc, lw, pairs, Rs) == last(grid_evaluate(Float64, wf, sh, lw; aggregation = Exact(), return_idxs = true))
            generic!(idxs, sc, lw, st)
            @test idxs == last(grid_evaluate(Float64, wf, sh, lw; aggregation = Exact(), return_idxs = true))
        end
        # and for the marginal on its own axis: the per-pair generic selection
        hw = [0.5, 2.0, 8.0, 40.0]
        sc = StudentTUniformScoring(5, 1.0; codimension = 1)
        idxs = fill(Int32(-1), length(pairs), length(hw))
        generic!(idxs, sc, hw, st)
        @test idxs == last(grid_evaluate(sc, sh, hw; return_idxs = true))
    end
end
