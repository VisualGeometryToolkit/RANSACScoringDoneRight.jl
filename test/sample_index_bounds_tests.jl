# The sample-index writes are bounds-checked: an invalid index throws instead of
# writing outside the buffer. Under `@inbounds`, a pool whose `sample_indices`
# held 0 made every `score_residuals!` (hence the marginal `best_pool_model`)
# write `q[0]`, corrupting the heap into later hangs and crashes far from the
# cause.

@testitem "sample_index_bounds" setup = [PoolFixtures] begin
    using Test, StaticArrays
    import VisualGeometryCore as VGC
    using RansacScoringDoneRight
    using RansacScoringDoneRight: alloc_scoring_workspace,
        score_residuals!, score_sq_residuals!, precompute_pair
    using VisualGeometryCore: AbstractConstraintProblem, JeffreysPrior, IGPrior
    const _SB = VGC

    struct _SBProblem <: AbstractConstraintProblem
        x::Vector{Float64}
    end
    _SB.data_size(p::_SBProblem) = length(p.x)
    VGC.constraint_with_jacobians(p::_SBProblem, θ, i::Int) =
        (p.x[i] - 0.6 * θ[1] - 0.8 * θ[2], SVector(1.0, 0.0), SVector(-0.6, -0.8))
    _SB.solver_covariance(::_SBProblem, _θ, ::AbstractVector{Int}) =
        SMatrix{2,2,Float64,4}(1e-3, 0.0, 0.0, 1e-3)

    @testset "sample indices are bounds-checked" begin
        n = 8
        r = collect(range(0.01, 0.1; length = n))
        s = StudentTUniformScoring(2, 5.0)
        sws = alloc_scoring_workspace(n, 1, s); q = zeros(n)
        @test_throws BoundsError score_residuals!(sws, q, s, r, n; sample_idx = [0, 1])
        @test_throws BoundsError score_residuals!(sws, q, s, r, n; sample_idx = [1, n + 1])
        S, k = score_residuals!(sws, q, s, r, n; sample_idx = [1, 2])
        @test isfinite(S) && q[1] == 0.0 && q[2] == 0.0

        sp = StudentTUniformScoring{Predictive}(2, 5.0; codimension = 1)
        swp = alloc_scoring_workspace(n, 1, sp); qp = zeros(n)
        p = _SBProblem(r); θ = SVector(0.0, 0.0)
        @test_throws BoundsError score_sq_residuals!(swp, qp, sp, p, θ, [0, 1], n)
        @test isfinite(first(score_sq_residuals!(swp, qp, sp, p, θ, [1, 2], n)))
    end

    @testset "pool staging throws on an invalid sample index too" begin
        # `precompute_pair` silently SKIPPED an out-of-range index until
        # 2026-09-30, where every other scorer threw; the one sample-zeroing
        # loop throws everywhere. "No minimal sample" is a 0-row matrix.
        n, M = 6, 2
        D = rand(Float32, n, M)
        pool(S) = identity_pool(D; sample_indices = S, errors_r = zeros(M), errors_t = zeros(M))
        @test_throws BoundsError precompute_pair(pool(zeros(Int, 2, M)), D)
        @test_throws BoundsError precompute_pair(pool(fill(n + 1, 2, M)), D)
        pp = precompute_pair(pool([1 2; 3 4]), D)
        @test all(pp.sorted_sq[1:2, :] .== 0) && issorted(pp.sorted_sq[:, 1])
        @test precompute_pair(pool(zeros(Int, 0, M)), D).sorted_sq[1, 1] > 0
    end
end
