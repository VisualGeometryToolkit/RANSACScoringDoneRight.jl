# The pool staging defaults to the stack's device (VisualGeometryToolkitCore's `default_device`), never a pinned CPU:
# the GPU-first rule. The resolver is swapped for a distinguishable CPU backend, so the default is seen to follow it.
@testitem "default_device: the pool staging follows the stack's device resolver" begin
    using Test, StaticArrays, Random, LinearAlgebra
    import KernelAbstractions
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore.Pools: PoolPair
    using RANSACScoringDoneRight: stage_ig_shared_from_pairs
    rng = MersenneTwister(1)
    I3 = SMatrix{3, 3, Float64, 9}(I); KC = SMatrix{3, 3, Float64, 9}(400.0, 0, 0, 0, 400.0, 0, 320.0, 240.0, 1.0)
    n, M = 40, 3
    pair = PoolPair{typeof(I3)}(; correspondences = [SVector(0.0, 0.0) => SVector(0.0, 0.0) for _ in 1:n], K1 = KC, K2 = KC, R_gt = I3,
                                t_gt = SVector(1.0, 0.0, 0.0), models = [I3 for _ in 1:M],
                                sample_indices = reduce(hcat, [sort(randperm(rng, n)[1:5]) for _ in 1:M]),
                                errors_r = rand(rng, M), errors_t = zeros(M), gt_inliers = falses(n), gt_variance = NaN,
                                source_map = I3, destination_map = I3)
    R = rand(rng, Float32, n, M)
    marked = KernelAbstractions.CPU(; static = true)            # a CPU backend the plain default is not
    old = VgtkCore._DEVICE_RESOLVER[]
    try
        VgtkCore._DEVICE_RESOLVER[] = () -> marked
        VgtkCore.reset_default_device!()
        @test stage_ig_shared_from_pairs([pair], [R]).backend === marked
    finally
        VgtkCore._DEVICE_RESOLVER[] = old
        VgtkCore.reset_default_device!()
    end
end
