# VisualGeometryToolkitCore's scoring contract on this package's scorer and on the
# baseline scorers it adds grid methods for: `test_scoring_contract` (score!
# type-stable, allocation-free and deterministic; inlier_count in 0:n;
# consensus_order a permutation; the derived consensus set and scale inferred;
# the bounded-sum early exit tie-safe) on StudentTUniformScoring — model-certain,
# codimension 1 and 2, with and without a minimal sample — and on MSAC/RANSAC at
# a threshold. Then the loop MSAC score IS the pool score, bit for bit.

@testitem "scoring_contract" setup = [PoolFixtures] begin
    using Test, Random
    import VisualGeometryToolkitCore as VGC
    using VisualGeometryToolkitCore: test_scoring_contract, scoring_workspace, score!, inlier_count,
        consensus_order, MsacScoring, RansacScoring
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: pool_model_score

    rng = MersenneTwister(41)
    n = 60
    # inlier/outlier mixtures, signed scalars and non-negative norms, plus a
    # vector with a non-finite (degenerate) residual
    vecs = [[rand(rng) < 0.6 ? 0.3 * randn(rng) : 12 * (rand(rng) - 0.5) for _ in 1:n],
            [rand(rng) < 0.4 ? abs(0.2 * randn(rng)) : 8 * rand(rng) for _ in 1:n],
            Float32[rand(rng) < 0.7 ? 0.1f0 * randn(rng, Float32) : 5 * rand(rng, Float32) for _ in 1:n]]
    vecs[1][7] = Inf
    sample = [3, 11, 29, 40]

    @testset "StudentTUniformScoring (d_g = $dg)" for dg in (1, 2)
        for r in vecs, smp in (nothing, sample), s in (StudentTUniformScoring(4, 6.0; codimension = dg),
                                                       StudentTUniformScoring(4, 6.0; codimension = dg,
                                                                              prior = VGC.IGPrior(1.5, 0.01)))
            test_scoring_contract(s, r; d_g = dg, sample = smp)
        end
    end

    @testset "MSAC and RANSAC at a threshold" begin
        for r in vecs, smp in (nothing, sample), τ in (0.25, 1.0)
            test_scoring_contract(MsacScoring(1; threshold = τ), r; sample = smp)
            test_scoring_contract(RansacScoring(1; threshold = τ), r; sample = smp)
        end
    end

    # The loop scorer's score on one model's residual column is the pool score of
    # that model, bitwise: Σ residual_quality in index order, the minimal
    # sample's points entering at zero residual.
    @testset "loop MSAC == pool_model_score(MsacScoring(), …), bitwise" begin
        for M in (1, 6), τ in (0.3, 1.0, 4.0)
            D = mixture_residuals(rng, 120, M)
            D[5, 1] = NaN32
            S = reduce(hcat, [sort(randperm(rng, 120)[1:5]) for _ in 1:M])
            pair = identity_pool(D; sample_indices = S, errors_r = zeros(M), errors_t = zeros(M))
            sc = MsacScoring(1; threshold = τ)
            W = scoring_workspace(sc, 120, 1)
            scores = zeros(120)
            for j in 1:M
                col = D[:, j]
                @test score!(W, sc, scores, col, 120; sample = S[:, j]) === pool_model_score(MsacScoring(), pair, D, j, τ)
                k = inlier_count(sc, W)
                @test Set(consensus_order(sc, W)[1:k]) ==
                      Set(i for i in 1:120 if i in S[:, j] || (isfinite(col[i]) && Float64(col[i] * col[i]) < τ^2))
            end
        end
    end
end
