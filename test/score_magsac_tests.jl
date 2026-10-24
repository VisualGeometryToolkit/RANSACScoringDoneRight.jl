# The exact per-pair MAGSAC++ selection, `best_pool_model(MagsacScoring(), pair,
# loss_width)` — the sum of `residual_quality` — and the two device aggregations of
# that quality, all pinned against MAGSAC++'s loss ρ written out below from the
# paper (§2.2), independently of the package's expression. The constants are not
# pinned: both sides read `_MAGSAC_SCALE_MULTIPLE`, so a change to it moves both. The pair has
# no minimal-sample points, which the per-pair scorer would score as exact
# inliers.

@testitem "score_magsac" setup = [PoolFixtures] begin
    using Test, Random
    using SpecialFunctions: gamma, gamma_inc
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: MagsacScoring, _MAGSAC_DOF, _MAGSAC_SCALE_MULTIPLE
    using VisualGeometryToolkitCore: best_pool_model   # the 4-argument method is this package's

    # MAGSAC++'s quality of a residual r at loss_width: ρ(κ·σ_max) − ρ(r), with
    # κ = _MAGSAC_SCALE_MULTIPLE, σ_max = loss_width/κ, R = r²/(2σ_max²), n = dof = 4, and C(n) = (2^{n/2}·Γ(n/2)·0.99)⁻¹
    # (0.99 is MAGSAC's normaliser):
    #   ρ(r)         = C·2^{(n+1)/2}/σ_max · [σ_max²/2·γ((n+1)/2, R)
    #                                        + r²/4·(Γ((n−1)/2, R) − Γ((n−1)/2, κ²/2))]
    #   ρ(κ·σ_max)   = σ_max·C·2^{(n−1)/2}·γ((n+1)/2, κ²/2)      (saturated)
    # γ, Γ the UNNORMALIZED incomplete gammas: Γ(o)·P and Γ(o)·Q. Zero from the
    # cutoff on.
    const _DOF = _MAGSAC_DOF; const _SCALE_MULTIPLE = _MAGSAC_SCALE_MULTIPLE
    _γ(o, x) = gamma(o) * gamma_inc(o, x)[1]
    _Γ(o, x) = gamma(o) * gamma_inc(o, x)[2]
    const _C = 1 / (2^(_DOF / 2) * gamma(_DOF / 2) * 0.99)
    function _paper_ρ(r, loss_width)
        σ = loss_width / _SCALE_MULTIPLE
        return _C * 2^((_DOF + 1) / 2) / σ *
               (σ^2 / 2 * _γ((_DOF + 1) / 2, r^2 / (2σ^2)) +
                r^2 / 4 * (_Γ((_DOF - 1) / 2, r^2 / (2σ^2)) - _Γ((_DOF - 1) / 2, _SCALE_MULTIPLE^2 / 2)))
    end
    _paper_ρ_cut(loss_width) = (σ = loss_width / _SCALE_MULTIPLE; σ * _C * 2^((_DOF - 1) / 2) * _γ((_DOF + 1) / 2, _SCALE_MULTIPLE^2 / 2))
    _paper_quality(r, loss_width) = r < loss_width ? _paper_ρ_cut(loss_width) - _paper_ρ(r, loss_width) : 0.0

    # The per-pair pick from the paper's quality: the best summed quality over the
    # finite residuals, the first model on a tie.
    function _magsac_reference(D::AbstractMatrix, loss_width::Float64)
        best_s = -Inf; best_j = 1
        for j in axes(D, 2)
            s = sum(_paper_quality(Float64(d), loss_width) for d in view(D, :, j) if isfinite(d))
            s > best_s && (best_s = s; best_j = j)
        end
        return best_j
    end

    @testset "best_pool_model(MagsacScoring(), pair, D, loss_width) == the paper's summed quality" begin
        rng = MersenneTwister(11)
        n, K = 80, 12
        D = Matrix{Float32}(undef, n, K)
        for j in 1:K
            σj = 0.2 + 0.1 * rand(rng)                   # every model a plausible inlier scale
            for i in 1:n
                D[i, j] = rand(rng) < 0.6 ? abs(σj * randn(rng)) : 5 * rand(rng)
            end
        end
        D[3, 4] = NaN32                                   # a non-finite residual is skipped
        pair = identity_pool(D)                           # no minimal-sample points
        picks = Int[]
        for loss_width in 10.0 .^ range(-1.0, 0.7; length = 15)
            j = best_pool_model(MagsacScoring(), pair, D, loss_width)
            @test j == _magsac_reference(D, loss_width)
            push!(picks, j)
        end
        @test length(unique(picks)) > 1                  # the grid actually moves the choice
    end

    # -----------------------------------------------------------------------------
    # MAGSAC++'s per-residual quality is the PAPER's loss, and the two device
    # aggregations of it agree. The device bin weight used to drop the Γ(o) factors
    # that turn Julia's regularized `gamma_inc` into the paper's unnormalized
    # incomplete gammas — not a constant factor (Γ(5/2)/Γ(3/2) = 1.5 re-weights the
    # r²/4 term against the σ²/2 term) — and agreed with the (since deleted) lookup
    # table on only ~90% of the (pair, loss width) cells of the pool below.
    #
    # Exact and LogBinned now evaluate ONE scalar, `residual_quality`, so their
    # agreement (2) cannot catch a wrong formula: a Γ-less quality moves both alike.
    # (1) is the guard — the scalar against the paper's ρ above.
    # -----------------------------------------------------------------------------
    using KernelAbstractions
    using RANSACScoringDoneRight: residual_quality, stage_ig_shared_from_pairs,
        Exact, LogBinned

    @testset "MAGSAC++: the paper's quality; Exact and LogBinned agree" begin
        # (1) Per residual
        for loss_width in (0.3, 1.0, 4.0)
            @test _paper_ρ(loss_width, loss_width) ≈ _paper_ρ_cut(loss_width) rtol = 1e-12   # ρ(κ·σ_max): the two forms meet
            for u in (0.0, 0.05, 0.2, 0.5, 0.8, 0.99)
                r = u * loss_width
                @test residual_quality(MagsacScoring(), r^2, loss_width) ≈ _paper_quality(r, loss_width) rtol = 1e-10
            end
            @test residual_quality(MagsacScoring(), loss_width^2, loss_width) == 0.0         # zero from the cutoff on
            @test residual_quality(MagsacScoring(), (3 * loss_width)^2, loss_width) == 0.0
        end

        # (2) Per selection, on a pool: LogBinned (200 bins/decade) picks Exact's
        # model. Binning discretises, so a near-tie may split; the bound leaves room
        # for that (240/240 agree at both precisions when written).
        rng = MersenneTwister(1)
        pairs = map(1:20) do _
            n, M = 200, 20
            D = mixture_residuals(rng, n, M)
            (identity_pool(D; errors_r = 10 .* rand(rng, M), errors_t = zeros(M)), D)
        end
        loss_widths = 10.0 .^ range(log10(0.3), 1.0; length = 12)
        shared = stage_ig_shared_from_pairs(first.(pairs), last.(pairs); backend = KernelAbstractions.CPU())
        _, exact_pick = grid_evaluate(Float64, MagsacScoring(), shared, loss_widths; aggregation = Exact(),
                                      return_idxs = true)
        for S in (Float64, Float32)
            _, binned_pick = grid_evaluate(S, MagsacScoring(), shared, loss_widths; aggregation = LogBinned(),
                                           return_idxs = true)
            @test count(binned_pick .== exact_pick) / length(exact_pick) >= 0.99
        end
    end
end
