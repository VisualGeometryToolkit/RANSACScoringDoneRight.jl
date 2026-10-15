# The minimal-sample rule, one for every scorer: a model's minimal-sample points
# are fit exactly by construction and carry no information. They enter at ZERO
# residual (`zero_sample!`, whatever was stored for them) and the model's `m`
# dof are subtracted. Each scorer is checked against the same problem with the
# sample points REMOVED:
#   * the marginal (`score_residuals!`) and the predictive marginal
#     (`score_sq_residuals!`) at model_dof = m score exactly what they score on
#     the n − m remaining points at model_dof = 0 — ν = dg·(k − m) and the
#     outlier count n − k agree term by term;
#   * a loss-width family's pool score (`pool_model_score`, hence `best_pool_model`, the
#     Exact grid and the typed pool scorers) is Σ over the remaining points plus
#     the model-independent constant m·q(0), so the sample points never change a
#     ranking;
#   * in every scorer the stored values at the sample indices are ignored.

@testitem "sample_point_rule" setup = [PoolFixtures] begin
    using Test, Random, StaticArrays
    import VisualGeometryCore as VGC
    using RansacScoringDoneRight
    using RansacScoringDoneRight: alloc_scoring_workspace,
        score_residuals!, score_sq_residuals!
    using VisualGeometryCore: AbstractConstraintProblem, JeffreysPrior, IGPrior, RansacScoring, MsacScoring,
        residual_quality, best_pool_model
    const _SP = VGC

    # A scalar-constraint problem: rᵢ = xᵢ − 0.6θ₁ − 0.8θ₂.
    struct _SPProblem <: AbstractConstraintProblem
        x::Vector{Float64}
    end
    _SP.data_size(p::_SPProblem) = length(p.x)
    VGC.constraint_with_jacobians(p::_SPProblem, θ, i::Int) =
        (p.x[i] - 0.6 * θ[1] - 0.8 * θ[2], SVector(1.0, 0.0), SVector(-0.6, -0.8))

    rng = MersenneTwister(71)
    n, m, halfwidth = 60, 4, 6.0
    r = [rand(rng) < 0.6 ? 0.3 * randn(rng) : 12 * (rand(rng) - 0.5) for _ in 1:n]
    S = sort(randperm(rng, n)[1:m]); rest = setdiff(1:n, S)
    r_junk = copy(r); r_junk[S] .= 50 .* rand(rng, m)       # stored sample values: ignored

    @testset "marginal: zero residual, m dof subtracted" begin
        for prior in (JeffreysPrior(), IGPrior(2.0, 0.5))
            s_m = StudentTUniformScoring(m, halfwidth; prior); s_0 = StudentTUniformScoring(0, halfwidth; prior)
            sws = alloc_scoring_workspace(n, 1, s_m); q = zeros(n)
            S_m, k_m = score_residuals!(sws, q, s_m, r, n; sample_idx = S)
            @test all(q[S] .== 0)
            S_j, k_j = score_residuals!(sws, q, s_m, r_junk, n; sample_idx = S)
            @test (S_j, k_j) == (S_m, k_m)
            nr = n - m
            sws0 = alloc_scoring_workspace(nr, 1, s_0); q0 = zeros(nr)
            S_0, k_0 = score_residuals!(sws0, q0, s_0, r[rest], nr)
            @test S_m ≈ S_0 rtol = 1e-12
            @test k_m == k_0 + m
        end
    end

    @testset "predictive marginal: zero residual and penalty, m dof subtracted" begin
        Σ̃ = SMatrix{2,2,Float64,4}(2e-3, 3e-4, 3e-4, 1e-3)
        θ = SVector(0.1, -0.2)
        s_m = StudentTUniformScoring{Predictive}(m, halfwidth; codimension = 1)
        s_0 = StudentTUniformScoring{Predictive}(0, halfwidth; codimension = 1)
        sws = alloc_scoring_workspace(n, 1, s_m); q = zeros(n)
        S_m, k_m = score_sq_residuals!(sws, q, s_m, _SPProblem(r), θ, Σ̃, S, n)
        @test all(q[S] .== 0) && all(sws.penalties[S] .== 0)
        @test score_sq_residuals!(sws, q, s_m, _SPProblem(r_junk), θ, Σ̃, S, n) == (S_m, k_m)
        nr = n - m
        sws0 = alloc_scoring_workspace(nr, 1, s_0); q0 = zeros(nr)
        S_0, k_0 = score_sq_residuals!(sws0, q0, s_0, _SPProblem(r[rest]), θ, Σ̃, nothing, nr)
        @test S_m ≈ S_0 rtol = 1e-10
        @test k_m == k_0 + m
    end

    @testset "loss-width families and pool scorers: sample points add only the constant m·q(0)" begin
        M = 6
        D = mixture_residuals(rng, n, M)
        SI = reduce(hcat, [sort(randperm(rng, n)[1:m]) for _ in 1:M])
        pair = identity_pool(D; sample_indices = SI, errors_r = rand(rng, M), errors_t = zeros(M))
        Dj = copy(D); for j in 1:M; Dj[SI[:, j], j] .= 100 .* rand(rng, Float32, m); end
        pair_j = identity_pool(Dj; sample_indices = SI, errors_r = pair.errors_r, errors_t = pair.errors_t)
        for sc in (RansacScoring(), MsacScoring(), MlesacScoring(), MagsacScoring(),
                   WelschCostScoring(), SmoothTLSScoring(), GemanScoring()), loss_width in (0.5, 2.0, 6.0)
            q0 = residual_quality(sc, 0.0, loss_width)
            for j in 1:M
                others = setdiff(1:n, SI[:, j])
                ref = sum(i -> (d = D[i, j]; isfinite(d) ? residual_quality(sc, Float64(d * d), loss_width) : 0.0), others)
                @test pool_model_score(sc, pair, D, j, loss_width) - m * q0 ≈ ref rtol = 1e-12 atol = 1e-12
                @test pool_model_score(sc, pair_j, Dj, j, loss_width) ≈ pool_model_score(sc, pair, D, j, loss_width) rtol = 1e-12
            end
            @test best_pool_model(sc, pair_j, Dj, loss_width) == best_pool_model(sc, pair, D, loss_width)
        end
        sm = StudentTUniformScoring(m, halfwidth)
        @test best_pool_model(sm, pair_j, Dj) == best_pool_model(sm, pair, D)
    end
end
