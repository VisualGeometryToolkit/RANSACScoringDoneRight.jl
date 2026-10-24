# `pool_imc` is the ONE stored selection pose error of a pool model, by model
# type: rotation-only for a homography pool, max(Δrot, Δtrans) for an epipolar
# (F/E) pool. `precompute_pair` stages exactly it, so every grid reports on it.

@testitem "pool_imc" setup = [PoolFixtures] begin
    using Test, Random, StaticArrays
    using VisualGeometryToolkitCore: HomographyMat, EssentialMat, FundMat
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: precompute_pair
    using VisualGeometryToolkitCore.Pools: pool_imc

    rng = MersenneTwister(5)
    n, M = 20, 4
    D = mixture_residuals(rng, n, M)
    er = [1.0, 4.0, 2.0, 0.5]; et = [3.0, 1.0, 2.0, 9.0]
    H = [HomographyMat(I3) for _ in 1:M]
    F = [FundMat(SMatrix{3,3,Float64,9}(0, 0, 0, 0, 0, -1, 0, 1, 0)) for _ in 1:M]
    E = [EssentialMat(SMatrix{3,3,Float64,9}(0, 0, 0, 0, 0, -1, 0, 1, 0)) for _ in 1:M]

    hpool = identity_pool(D; errors_r = er, errors_t = et, models = H)
    @test [pool_imc(hpool, j) for j in 1:M] == er                       # Δrot alone
    @test precompute_pair(hpool, D).errors_imc == er
    for ms in (F, E)
        pool = identity_pool(D; errors_r = er, errors_t = et, models = ms)
        @test [pool_imc(pool, j) for j in 1:M] == max.(er, et)
        @test precompute_pair(pool, D).errors_imc == max.(er, et)
    end
end
