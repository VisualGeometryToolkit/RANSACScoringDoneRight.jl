# The scorers read the problem protocol's whitened residuals: a DEGENERATE
# measurement (VisualGeometryToolkitCore's one Sampson degeneracy rule — residual 0
# when the constraint vanishes, ±Inf otherwise) reaches the sweep as an outlier,
# and the covariance-given predictive scorer reports the SAME signed residuals
# `residuals!` writes. The protocol itself (`whitened_residual`,
# `residual_context`, the rounding scale) is VisualGeometryToolkitCore's and is tested
# there (constraint_problem_tests.jl).

@testitem "whitened_residuals" begin
    using Test, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VGC
    using VisualGeometryToolkitCore: AbstractConstraintProblem
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: alloc_scoring_workspace, score_sq_residuals!, sweep_order, penalties,
        Predictive

    # A problem that IS its list of constraint tuples (g, ∂ₓg, ∂_θg[, Σ̃_x]).
    struct _CDProblem{C} <: AbstractConstraintProblem
        cds::Vector{C}
    end
    VGC.data_size(p::_CDProblem) = length(p.cds)
    VGC.constraint_with_jacobians(p::_CDProblem, _model, i::Int) = p.cds[i]
    VGC.solver_covariance(::_CDProblem, _model, ::AbstractVector{Int}) =
        SMatrix{2,2,Float64,4}(1e-3, 0.0, 0.0, 1e-3)

    _scalar(g, gx) = (g, SVector{2,Float64}(gx...), SVector(1.0, 2.0))

    # The epipolar constraint of VGC's FundMat at each pair, as a 5-tuple with the
    # model's rounding scale.
    struct _F5Problem <: AbstractConstraintProblem
        cs::Vector{Pair{SVector{2,Float64},SVector{2,Float64}}}
    end
    VGC.data_size(p::_F5Problem) = length(p.cs)
    function VGC.constraint_with_jacobians(p::_F5Problem, F, i::Int)
        u, v = p.cs[i]
        g, Gx, Gθ = VGC.constraint_with_jacobians(F, u, v)
        return (g, Gx, Gθ, I, VGC.constraint_rounding_scale(F, u, v))
    end

    @testset "the sweeps read a degenerate point as an outlier" begin
        n = 12
        cds = [_scalar(0.01 * i, (0.6, 0.8)) for i in 1:n]
        cds[5] = _scalar(0.02, (0.0, 0.0))              # degenerate, unsatisfied
        cds[n] = _scalar(3.0, (0.6, 0.8))               # an ordinary outlier
        p = _CDProblem(cds)

        s = StudentTUniformScoring(2, 5.0)
        sws = alloc_scoring_workspace(n, 1, s); q = zeros(n)
        S, k = score_sq_residuals!(sws, q, s, p, nothing, n)
        @test q[5] === Inf
        @test isfinite(S)
        @test !(5 in sweep_order(sws)[1:k])
        @test k == n - 2                                 # both outliers out

        # predictive: whiten's zero Jacobian row gives V = 1, so q = Inf, ℓ = 0
        sp = StudentTUniformScoring{Predictive}(2, 5.0)
        swp = alloc_scoring_workspace(n, 1, sp); qp = zeros(n)
        Sp, kp = score_sq_residuals!(swp, qp, sp, p, nothing, [1, 2], n)
        @test qp[5] === Inf
        @test penalties(swp)[5] === 0.0
        @test isfinite(Sp)
        @test !(5 in sweep_order(swp)[1:kp])
    end

    @testset "the predictive scorer reports residuals!'s signed residuals" begin
        n = 8
        cds = [_scalar((-1.0)^i * 0.05 * i, (0.6, 0.8)) for i in 1:n]
        p = _CDProblem(cds)
        r = zeros(n); VGC.residuals!(r, p, nothing)
        @test any(<(0), r)                               # signed, not a norm
        sp = StudentTUniformScoring{Predictive}(2, 5.0)
        swp = alloc_scoring_workspace(n, 1, sp); qp = zeros(n); rp = zeros(n)
        Σ̃ = VGC.solver_covariance(p, nothing, [1, 2])
        score_sq_residuals!(swp, qp, sp, p, nothing, Σ̃, nothing, n; residuals = rp)
        @test rp == r
    end

    @testset "a 5-tuple constraint: the scorer squares sampson_distance" begin
        S2 = SVector{2,Float64}
        F1 = VGC.FundMat{Float64}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        F2 = VGC.FundMat{Float64}(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cases = (
            (F1, [S2(0.0, 5.0) => S2(0.0, -3.0), S2(1e-18, 5.0) => S2(0.0, -3.0), S2(0.7, 2.0) => S2(-0.4, 1.0)]),
            (F2, [S2(0.0, 0.0) => S2(0.0, 0.0), S2(1e-17, 0.0) => S2(0.0, 1e-17), S2(0.3, -0.2) => S2(0.5, 0.9)]))
        for (F, cs) in cases
            p = _F5Problem(cs)
            n = length(cs)
            d = [VGC.sampson_distance(F, c) for c in cs]
            s = StudentTUniformScoring(2, 5.0)
            sws = alloc_scoring_workspace(n, 1, s); q = zeros(n)
            score_sq_residuals!(sws, q, s, p, F, n)
            @test q == d .^ 2
        end
    end
end
