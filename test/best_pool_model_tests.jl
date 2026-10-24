# `best_pool_model` is the per-pair model selection over the quantity the grids
# evaluate. For a width-free loss-width family at a loss width (this package's
# 4-argument method) it is the Exact grid's per-model score on one pair (the sum
# of `residual_quality`, fed as staging feeds it); a scorer AT its width — the
# Student-t marginal, MSAC/RANSAC at a threshold — goes through
# VisualGeometryToolkitCore's generic 3-argument method over `score!`, and must pick
# exactly what the width-free family picks at that width. The sentinel for "no
# model" is 0, never a valid index. (The typed pool-scorer handles are retired.)

@testitem "best_pool_model" setup = [PoolFixtures] begin
    using Test, Random
    using KernelAbstractions
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: stage_ig_shared_from_pairs,
        Exact, exact_pm_score_kernel!, _support_sq, pool_model_score
    using VisualGeometryToolkitCore: AbstractConstraintProblem, RansacScoring, MsacScoring, best_pool_model
    import VisualGeometryToolkitCore as VGC

    const _BP_FAMILIES = (RansacScoring(), MsacScoring(), MlesacScoring(), MagsacScoring(),
                          WelschCostScoring(), SmoothTLSScoring(), GemanScoring())

    function _bp_pair(rng; n = 120, M = 8, sample = :valid)
        D = mixture_residuals(rng, n, M)
        S = zeros(Int, sample === :none ? 0 : 5, M)
        if sample === :valid
            for j in 1:M
                S[:, j] = sort(randperm(rng, n)[1:5])
            end
        end
        (identity_pool(D; sample_indices = S, errors_r = 10 .* rand(rng, M), errors_t = zeros(M)), D)
    end

    struct _BPProblem <: AbstractConstraintProblem end
    VGC.codimension(::Type{_BPProblem}) = 1
    VGC.minimal_sample_size(::Type{_BPProblem}) = 5

    @testset "best_pool_model: one loop, the grids' quantity" begin
        rng = MersenneTwister(23)
        loss_widths = [0.3, 1.0, 3.0, 8.0]
        be = KernelAbstractions.CPU()

        @testset "loss-width family == the Exact kernel on one pair, per model" begin
            pair, D = _bp_pair(rng)
            sh = stage_ig_shared_from_pairs([pair], [D]; backend = be)
            M = Int(sh.M_per_pair[1]); T = length(loss_widths)
            for sc in _BP_FAMILIES
                Sx = zeros(1, M, T)
                exact_pm_score_kernel!(be)(Sx, sh.sorted_sq_flat, sh.pair_offset, sh.K_per_pair,
                    sh.M_per_pair, loss_widths, Int32(T), Int32(0), sc, _support_sq(sc, maximum(loss_widths));
                    ndrange = (1, M))
                KernelAbstractions.synchronize(be)
                for j in 1:M, (t, loss_width) in enumerate(loss_widths)
                    # same terms, summed in index order here and in sorted order there
                    @test pool_model_score(sc, pair, D, j, loss_width) ≈ Sx[1, j, t] rtol = 1e-12 atol = 1e-12
                end
            end
        end

        @testset "loss-width family picks what the Exact grid picks" begin
            pool = [_bp_pair(rng) for _ in 1:10]
            sh = stage_ig_shared_from_pairs(first.(pool), last.(pool); backend = be)
            for sc in _BP_FAMILIES
                _, jx = grid_evaluate(Float64, sc, sh, loss_widths; aggregation = Exact(), return_idxs = true)
                @test [best_pool_model(sc, p, D, loss_width) for (p, D) in pool, loss_width in loss_widths] == jx
            end
        end

        @testset "sentinel 0 = no model" begin
            empty, De = _bp_pair(rng; M = 0)
            nanpair, _ = _bp_pair(rng; M = 3)
            for sc in _BP_FAMILIES
                @test best_pool_model(sc, empty, De, 1.0) == 0
            end
            @test best_pool_model(StudentTUniformScoring(5, 0.1; codimension = 1), empty, De) == 0
            @test best_pool_model(StudentTUniformScoring(5, 0.1; codimension = 1), nanpair, fill(NaN32, 120, 3)) == 0
            for sc in (MsacScoring(1; threshold = 1.0), RansacScoring(1; threshold = 1.0),
                       StudentTUniformScoring(_BPProblem, 0.1))
                @test best_pool_model(sc, empty, De) == 0
            end
        end

        @testset "a scorer at its width ≡ the width-free family at that width (bitwise)" begin
            # VisualGeometryToolkitCore's generic form over `score!` against this
            # package's 4-argument loop over `residual_quality`: the same index,
            # over several pairs, widths and codimensions, minimal samples and all.
            for _ in 1:12
                pair, D = _bp_pair(rng; n = rand(rng, 20:150), M = rand(rng, 1:12))
                for loss_width in (0.05, loss_widths..., 25.0), dg in (1, 2)
                    @test best_pool_model(MsacScoring(dg; threshold = loss_width), pair, D) ===
                          best_pool_model(MsacScoring(), pair, D, loss_width)
                    @test best_pool_model(RansacScoring(dg; threshold = loss_width), pair, D) ===
                          best_pool_model(RansacScoring(), pair, D, loss_width)
                end
            end
            # the retired handles are gone, not aliased
            for n in ("Msac", "Ransac", "Marginal")     # spelled in fragments: retired names
                @test !isdefined(RANSACScoringDoneRight, Symbol(n, "PoolScorer"))
            end
            # the marginal at its width: the generic form with a supplied workspace
            # and scratch is the allocating one
            pair, D = _bp_pair(rng)
            for halfwidth in (0.05, 0.5, 5.0)
                sc = StudentTUniformScoring(_BPProblem, halfwidth)
                ws = VGC.scoring_workspace(sc, VGC.Pools.num_points(pair), 1)
                @test best_pool_model(sc, pair, D) ==
                      best_pool_model(sc, pair, D; ws, scores = Vector{Float64}(undef, VGC.Pools.num_points(pair)))
            end
        end

        @testset "an invalid sample index throws" begin
            bad, Db = _bp_pair(rng; sample = :zero)
            @test_throws BoundsError best_pool_model(MsacScoring(), bad, Db, 1.0)
            @test_throws BoundsError best_pool_model(StudentTUniformScoring(5, 0.1; codimension = 1), bad, Db)
        end

        @testset "a residual matrix of another pair is refused" begin
            pair, D = _bp_pair(rng)
            @test_throws DimensionMismatch best_pool_model(MsacScoring(), pair, D[1:end-1, :], 1.0)
            @test_throws DimensionMismatch best_pool_model(StudentTUniformScoring(5, 0.1; codimension = 1), pair, D[:, 1:end-1])
            @test_throws DimensionMismatch stage_ig_shared_from_pairs([pair], [D[:, 1:end-1]])
            @test_throws DimensionMismatch stage_ig_shared_from_pairs([pair], [D, D])
        end
    end
end
