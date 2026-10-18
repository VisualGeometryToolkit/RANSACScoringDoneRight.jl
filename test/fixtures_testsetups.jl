# Shared test fixtures. ReTestItems discovers `@testsetup` modules in
# `*_testsetups.jl` files; an item takes one with `setup = [Name]`.

# A light pool record built directly in the pool frame, so its maps are the
# identity, supplied explicitly (a pool record without maps is refused), plus
# the inlier/outlier mixture residuals most pool tests score: the record and its
# residual matrix travel separately, as they do from a pool's two files.
@testsetup module PoolFixtures
    using Random
    using StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore.Pools: PoolPair
    export I3, identity_pool, mixture_residuals

    const I3 = SMatrix{3,3,Float64,9}(I)
    const _KC = SMatrix{3,3,Float64,9}(400.0, 0.0, 0.0, 0.0, 400.0, 0.0, 320.0, 240.0, 1.0)

    """
        identity_pool(D; sample_indices, errors_r, errors_t, models) -> PoolPair

    The light record of a pool of `size(D, 2)` identity models over `size(D, 1)`
    zero correspondences — the shape of the residual matrix `D` it is scored
    with — with a dummy pinhole calibration and identity maps. `sample_indices` defaults to none (a 0-row matrix: no minimal-sample
    points), the errors to NaN, the models to plain identity matrices (supply
    `models` — e.g. `HomographyMat`s — to type the pool).
    """
    function identity_pool(D::AbstractMatrix; sample_indices = zeros(Int, 0, size(D, 2)),
                           errors_r = fill(NaN, size(D, 2)), errors_t = fill(NaN, size(D, 2)),
                           models = [I3 for _ in 1:size(D, 2)])
        n, M = size(D)
        return PoolPair{eltype(models)}(;
            csponds = [SVector(0.0, 0.0) => SVector(0.0, 0.0) for _ in 1:n], K1 = _KC, K2 = _KC,
            R_gt = I3, t_gt = SVector(1.0, 0.0, 0.0), models, sample_indices, errors_r, errors_t,
            gt_inliers = falses(n), gt_variance = NaN, source_map = I3, target_map = I3)
    end

    """
        mixture_residuals(rng, n, M) -> Matrix{Float32}

    `n` residual magnitudes for each of `M` models: per model an inlier fraction
    `f ∈ [0.3, 0.8)` and scale `σ ∈ [0.2, 1)`, inliers `|σ·N(0,1)|`, outliers
    uniform on `[0, 20)`.
    """
    function mixture_residuals(rng::AbstractRNG, n::Integer, M::Integer)
        D = Matrix{Float32}(undef, n, M)
        for j in 1:M
            f = 0.3 + 0.5 * rand(rng); σ = 0.2 + 0.8 * rand(rng)
            for i in 1:n
                D[i, j] = rand(rng) < f ? abs(σ * randn(rng)) : 20 * rand(rng)
            end
        end
        return D
    end
end
