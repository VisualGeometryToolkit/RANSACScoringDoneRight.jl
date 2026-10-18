# Predictive scoring refuses a problem with no solver covariance, and SKIPS a
# degenerate sample (a provider's `nothing`) like a failed solve. It never falls
# back to model-certain scoring, which is how RVG's AbsolutePoseProblem and
# HandEyeProblem ran "predictive" without it doing anything (until 2026-09-22).

@testitem "predictive_provider" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VGC
    using RansacScoringDoneRight
    using RansacScoringDoneRight: Predictive, alloc_scoring_workspace, score_sq_residuals!, penalties, sort_key
    using VisualGeometryToolkitCore: AbstractConstraintProblem, JeffreysPrior, scored_model, solver_covariance,
        provides_solver_covariance
    const _PV = VGC

    # Scalar constraints g = x − ⟨(0.6, 0.8), θ⟩ over 1-D data, θ ∈ ℝ².
    abstract type _PVBase <: AbstractConstraintProblem end
    struct _Providerless <: _PVBase
        x::Vector{Float64}
    end
    struct _Provider <: _PVBase
        x::Vector{Float64}
        degenerate::Bool      # the solver Jacobian does not exist for this sample
    end
    _PV.data_size(p::_PVBase) = length(p.x)
    _PV.codimension(::Union{_PVBase, Type{<:_PVBase}}) = 1
    _PV.minimal_sample_size(::Union{_PVBase, Type{<:_PVBase}}) = 2
    VGC.constraint_with_jacobians(p::_PVBase, θ, i::Int) =
        (p.x[i] - 0.6 * θ[1] - 0.8 * θ[2], SVector(1.0, 0.0), SVector(-0.6, -0.8))
    _PV.solver_covariance(p::_Provider, _θ, ::AbstractVector{Int}) =
        p.degenerate ? nothing : SMatrix{2,2,Float64,4}(1e-3, 0.0, 0.0, 1e-3)
    _PV.provides_solver_covariance(::Type{<:_Provider}) = true

    _message(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError" end

    @testset "predictive scoring: refuse a missing provider, skip a degenerate sample" begin
        n = 10
        x = collect(range(0.0, 0.2; length = n)); x[n] = 5.0     # one outlier
        θ = SVector(0.0, 0.0); idx = [1, 2]
        none = _Providerless(x); prov = _Provider(x, false); degen = _Provider(x, true)

        @testset "the trait" begin
            @test provides_solver_covariance(_Providerless) === false
            @test provides_solver_covariance(_Provider) === true
        end

        @testset "construction refuses a problem type with no provider" begin
            @test_throws ArgumentError StudentTUniformScoring{Predictive}(none, 5.0)
            @test_throws ArgumentError StudentTUniformScoring{Predictive}(none, 5.0; prior = JeffreysPrior())
            @test_throws ArgumentError StudentTUniformScoring{Predictive}(_Providerless, 5.0)
            msg = _message(() -> StudentTUniformScoring{Predictive}(none, 5.0))
            @test occursin("_Providerless provides none", msg)
            @test occursin("solver_covariance(::_Providerless, model, sample_indices)", msg)
            @test occursin("provides_solver_covariance(::Type{<:_Providerless}) = true", msg)
            @test occursin("or score _Providerless model-certain", msg)
            # the model-certain spec needs no provider; a provider constructs
            @test StudentTUniformScoring(none, 5.0) isa StudentTUniformScoring{Nothing}
            @test StudentTUniformScoring(_Providerless, 5.0) isa StudentTUniformScoring{Nothing}
            @test StudentTUniformScoring{Predictive}(prov, 5.0) isa StudentTUniformScoring{Predictive}
            @test StudentTUniformScoring{Predictive}(_Provider, 5.0) isa StudentTUniformScoring{Predictive}
        end

        # The model_dof constructors carry no problem: the fallback method refuses
        # at the first scored sample, with the same message.
        sp = StudentTUniformScoring{Predictive}(2, 5.0; codimension = 1)

        @testset "the fallback solver_covariance refuses too" begin
            sws = alloc_scoring_workspace(n, 1, sp); q = zeros(n)
            @test_throws ArgumentError solver_covariance(none, θ, idx)
            @test_throws ArgumentError scored_model(sp, none, θ, idx)
            @test_throws ArgumentError score_sq_residuals!(sws, q, sp, none, θ, idx, n)
            @test _message(() -> solver_covariance(none, θ, idx)) ==
                  _message(() -> StudentTUniformScoring{Predictive}(none, 5.0))
            # the model-certain wrap never consults a provider
            @test scored_model(StudentTUniformScoring(2, 5.0), none, θ, idx) === θ
        end

        @testset "a degenerate sample is no hypothesis" begin
            sws = alloc_scoring_workspace(n, 1, sp); q = fill(7.0, n)
            pen = copy(penalties(sws)); key = copy(sort_key(sws))
            @test score_sq_residuals!(sws, q, sp, degen, θ, idx, n) === (-Inf, 0)
            @test q == fill(7.0, n)                               # untouched
            @test penalties(sws) == pen && sort_key(sws) == key
            @test scored_model(sp, degen, θ, idx) === nothing
        end

        @testset "a provided covariance scores predictively" begin
            sws = alloc_scoring_workspace(n, 1, sp); q = zeros(n)
            S, k = score_sq_residuals!(sws, q, sp, prov, θ, idx, n)
            @test isfinite(S) && k == n - 1                       # the outlier is out
            @test any(>(0), penalties(sws))                        # the covariance entered
            w = scored_model(sp, prov, θ, idx)
            @test w isa VGC.Uncertain
            @test VGC.cov(w) == SMatrix{2,2,Float64,4}(1e-3, 0.0, 0.0, 1e-3)
        end

        @testset "a given covariance: the provider's score, residual norms on request" begin
            Σ̃ = SMatrix{2,2,Float64,4}(1e-3, 0.0, 0.0, 1e-3)
            sws1 = alloc_scoring_workspace(n, 1, sp); q1 = zeros(n)
            sws2 = alloc_scoring_workspace(n, 1, sp); q2 = zeros(n); r = zeros(n)
            @test score_sq_residuals!(sws1, q1, sp, prov, θ, idx, n) ===
                  score_sq_residuals!(sws2, q2, sp, prov, θ, Σ̃, idx, n; residuals = r)
            @test q1 == q2 && penalties(sws1) == penalties(sws2)
            @test q2[idx] == zeros(2) && penalties(sws2)[idx] == zeros(2)   # the sample is exact
            @test r ≈ abs.(x)                                    # ‖rᵢ‖ at θ = 0, unit Σ_x
            # no minimal sample (a refit's covariance): nothing is zeroed
            sws3 = alloc_scoring_workspace(n, 1, sp); q3 = zeros(n)
            S3, k3 = score_sq_residuals!(sws3, q3, sp, prov, θ, Σ̃, nothing, n)
            @test isfinite(S3) && q3[2] > 0 && penalties(sws3)[2] > 0
        end
    end
end
