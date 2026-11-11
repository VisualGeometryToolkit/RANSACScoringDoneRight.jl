# VisualGeometryToolkitCore's scoring contract on this package's scorer and on the
# baseline scorers it adds grid methods for: `test_scoring_contract` (score!
# type-stable, allocation-free and deterministic; inlier_count in 0:n;
# consensus_order a permutation; the derived consensus set and scale inferred;
# the bounded-sum early exit tie-safe) on StudentTUniformScoring — model-certain,
# codimension 1 and 2, with and without a minimal sample — and on MSAC/RANSAC at
# a threshold. Then every loss-width family at a loss width (MSAC, RANSAC and this
# package's) scores by the one summed quality, bit for bit.

@testitem "scoring_contract" setup = [PoolFixtures] begin
    using VisualGeometryToolkitCore: StudentTUniformScoring, FlatLabelPrior, label_prior, JeffreysPrior   # the Student-t scorer (Core's)
    using Test, Random
    import VisualGeometryToolkitCore as VGC
    using VisualGeometryToolkitCore: test_scoring_contract, scoring_workspace, score!, inlier_count,
        consensus_order, MsacScoring, RansacScoring
    using RANSACScoringDoneRight

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
        for r in vecs, smp in (nothing, sample), s in (StudentTUniformScoring(4, 6.0; codimension = dg, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()),
                                                       StudentTUniformScoring(4, 6.0; codimension = dg,
                                                                              scale_prior = VGC.IGPrior(1.5, 0.01), label_prior = FlatLabelPrior()))
            test_scoring_contract(s, r; d_g = dg, sample = smp)
        end
    end

    families = (MsacScoring, RansacScoring, MlesacScoring, MagsacScoring,
                WelschCostScoring, SmoothTLSScoring, GemanScoring)

    # Every family at a loss width is a VisualGeometryToolkitCore threshold
    # scorer: the whole contract, the early exit included, from Core's one loop.
    @testset "every family at a loss width" begin
        for F in families, r in vecs, smp in (nothing, sample), τ in (0.25, 1.0)
            sc = F(1; threshold = τ)
            @test sc isa VGC.AbstractThresholdScoring{Float64}
            test_scoring_contract(sc, r; sample = smp)
        end
        for F in families      # the width-free form implements no scoring contract
            @test F() isa VGC.AbstractThresholdScoring{Nothing}
            @test !VGC.has_decision_boundary(F()) && VGC.score_bound(F()) === nothing
            @test_throws MethodError scoring_workspace(F(), n, 1)
        end
    end

    # Every loss-width family at a loss width scores one model's residual column
    # by the same arithmetic, bitwise: Σ residual_quality in index order (a
    # non-finite square an outlier, quality 0: a NaN, an Inf and a finite d whose
    # Float32 square overflows), then the minimal sample's points corrected to
    # zero residual — written out here, as this package's own loop computed it
    # before Core's one loop replaced it.
    function _summed_quality(sc, col, smp, τ)
        q(d) = (d2 = Float64(d * d); isfinite(d2) ? VGC.residual_quality(sc, d2, τ) : 0.0)
        s = 0.0
        for d in col
            s += q(d)
        end
        if smp !== nothing
            for i in smp
                s += VGC.residual_quality(sc, 0.0, τ) - q(col[i])
            end
        end
        return s
    end
    @testset "every family's score! is the summed quality, bitwise" begin
        for M in (1, 6), τ in (0.3, 1.0, 4.0)
            D = mixture_residuals(rng, 120, M)
            D[5, 1] = NaN32; D[6, 1] = Inf32; D[8, 1] = 1.0f30      # 1f30² overflows Float32
            S = reduce(hcat, [sort(randperm(rng, 120)[1:5]) for _ in 1:M])
            S[1:3, 1] = [5, 6, 8]                                    # sample points among them
            for F in families
                sc = F(1; threshold = τ)
                W = scoring_workspace(sc, 120, 1)
                scores = zeros(120)
                for j in 1:M, smp in (S[:, j], nothing)
                    col = D[:, j]
                    @test score!(W, sc, scores, col, 120; sample = smp) === _summed_quality(sc, col, smp, τ)
                    k = inlier_count(sc, W)
                    @test Set(consensus_order(sc, W)[1:k]) ==
                          Set(i for i in 1:120 if (smp !== nothing && i in smp) ||
                                                     (isfinite(col[i]) && Float64(col[i] * col[i]) < τ^2))
                end
            end
        end
    end

    # The early exit never changes the selection: a running best, each model
    # scored against it, picks the full scores' first strict best.
    @testset "the early exit keeps the selection" begin
        for F in families, τ in (0.3, 1.0, 4.0)
            sc = F(1; threshold = τ)
            D = mixture_residuals(rng, 120, 200)
            S = reduce(hcat, [randperm(rng, 120)[1:4] for _ in 1:200])
            W = scoring_workspace(sc, 120, 1)
            full = [score!(W, sc, zeros(120), view(D, :, j), 120; sample = view(S, :, j)) for j in 1:200]
            best, best_j = -Inf, 0
            for j in 1:200
                s = score!(W, sc, zeros(120), view(D, :, j), 120; sample = view(S, :, j), bound = best)
                s > best && (best = s; best_j = j)
            end
            @test best_j == findfirst(==(maximum(full)), full) && best === maximum(full)
        end
    end
end
