# =============================================================================
# The Gaussian-Uniform boundary as the loss width <-> outlier half-width maps
# read it, and the loss-width families' per-residual quality.
#
# `decision_threshold` and its inverse `decision_width(::GauInlier, ·)` — the ONE
# place the Eq. (2) MAP rule lives — are VisualGeometryCore's and pinned there.
# =============================================================================

@testitem "boundary" begin
    using Test
    # VisualGeometryCore's since K+ (width_axes.jl); pinned here too, where the
    # Student-t marginal and the loss-width families they convert between live.
    using VisualGeometryCore: outlier_halfwidth_to_loss_width, loss_width_to_outlier_halfwidth
    using VisualGeometryCore: GauInlier, decision_width

    # (GauInlier's own two directions are VisualGeometryCore's:
    # gau_inlier_boundary_tests.jl there.)
    @testset "the loss width <-> outlier half-width maps ride the GauInlier boundary" begin
        @testset "outlier half-width <-> loss width round trip at the protocol scale multiple" begin
            for operating_scale_multiple in (1.0, 2.5, 4.0), outlier_halfwidth in (0.7, 5.0, 120.0), d_g in (1, 2)
                loss_width = outlier_halfwidth_to_loss_width(outlier_halfwidth; operating_scale_multiple, d_g)
                @test loss_width_to_outlier_halfwidth(loss_width; operating_scale_multiple, d_g) ≈
                      outlier_halfwidth rtol = 1.0e-12
            end
        end

        @testset "outlier half-width <-> loss width honour the codimension" begin
            # d_g = 1 (F/E) is BIT-identical to the form that hard-coded it
            uh1(scale_multiple) = decision_width(GauInlier(1.0; d_g = 1), Float64(scale_multiple))
            for operating_scale_multiple in (1.0, 2.5, 4.0), x in (0.004, 0.05, 0.7, 120.0)
                @test outlier_halfwidth_to_loss_width(x; operating_scale_multiple, d_g = 1) ===
                      Float64(operating_scale_multiple) * x / uh1(operating_scale_multiple)
                @test loss_width_to_outlier_halfwidth(x; operating_scale_multiple, d_g = 1) ===
                      uh1(operating_scale_multiple) * x / Float64(operating_scale_multiple)
            end
            # d_g = 2 (H) is the codim-2 crossover: the loss width is
            # operating_scale_multiple·ŝ where the outlier half-width is the codim-2
            # half-width at scale ŝ — the boundary itself, not a codim-1 stand-in
            for operating_scale_multiple in (1.0, 2.5, 4.0), ŝ in (0.01, 0.3, 2.0)
                outlier_halfwidth = decision_width(GauInlier(ŝ^2; d_g = 2), operating_scale_multiple * ŝ)
                @test outlier_halfwidth_to_loss_width(outlier_halfwidth; operating_scale_multiple, d_g = 2) ≈
                      operating_scale_multiple * ŝ rtol = 1.0e-12
            end
            @test_throws UndefKeywordError outlier_halfwidth_to_loss_width(1.0; operating_scale_multiple = 2.5)   # no default d_g
        end
    end

    # =============================================================================
    # The loss-width families' per-residual quality (`residual_quality`, the ONE scalar both the
    # Exact and the LogBinned aggregation evaluate) comes from VisualGeometryCore's
    # rho instead of hand-written closed forms. Pin BOTH the agreement and the
    # assumption it rests on.
    # =============================================================================
    using RansacScoringDoneRight: WelschCostScoring, SmoothTLSScoring, GemanScoring,
        MlesacScoring, MagsacScoring
    using VisualGeometryCore: MsacScoring, RansacScoring, residual_quality
    import VisualGeometryCore as VGC
    using VisualGeometryCore: loss_shape      # the owner's accessor, which RSDR extends

    @testset "loss bin weights: a quality in [0, 1], 1 at d = 0, scale-free" begin
        # The shapes' closed forms are VGC's (its tests own them); what the
        # loss-width families need of the quality is checked here, not restated: 1
        # at a zero residual, non-increasing in |d| within [0, 1], a function of
        # d/loss_width only, and — smooth-TLS being compactly supported — exactly 0
        # from the loss width on.
        rs = range(0.0, 12.0; length = 121)
        for sc in (WelschCostScoring(), SmoothTLSScoring(), GemanScoring()),
            loss_width in (0.3, 1.0, 4.7)
            q = [residual_quality(sc, r * r, loss_width) for r in rs]
            @test q[1] == 1.0
            @test all(x -> 0.0 <= x <= 1.0, q)
            @test issorted(q; rev = true)
            for r in rs, λ in (0.25, 3.0)
                @test residual_quality(sc, (λ * r)^2, λ * loss_width) ≈
                      residual_quality(sc, r * r, loss_width) atol = 1.0e-14
            end
        end
        @test all(residual_quality(SmoothTLSScoring(), r * r, 1.0) == 0.0
                  for r in range(1.0, 12.0; length = 23))

        # It is the QUALITY, not the IRLS weight. Welsch coincides; the other two
        # must NOT — if they ever did, `VGC.weight` would have been the right verb
        # and this test is the thing that says otherwise.
        @testset "quality ≠ IRLS weight for GM and smooth-TLS" begin
            r, loss_width = 2.0, 2.5
            @test residual_quality(WelschCostScoring(), r^2, loss_width) ≈
                VGC.weight(VGC.WelschShape(), r^2, loss_width^2)
            @test residual_quality(GemanScoring(), r^2, loss_width)^2 ≈
                VGC.weight(VGC.GemanMcclureShape(), r^2, loss_width^2)
            @test !isapprox(residual_quality(GemanScoring(), r^2, loss_width),
                            VGC.weight(VGC.GemanMcclureShape(), r^2, loss_width^2))
        end

        # `residual_quality` divides by the squared loss width as the well depth.
        # That holds because VGC scales rho by it, so it saturates there — assert it
        # rather than trusting it, since a shape added later could break the
        # convention.
        @testset "rho saturates at the squared loss width (the depth the quality divides by)" begin
            for sh in (VGC.WelschShape(), VGC.GemanMcclureShape(), VGC.SmoothTLSShape())
                for sq_loss_width in (0.09, 1.0, 22.0)
                    @test VGC.ρ(sh, 1.0e12 * sq_loss_width, sq_loss_width) ≈ sq_loss_width rtol = 1.0e-9
                end
            end
        end
    end

    # =============================================================================
    # The scorings that PROFILE their labels reduce to a rho of one width; the ones
    # that marginalise them do not. Assert both sides.
    # =============================================================================
    @testset "loss_shape follows the label elimination" begin
        # MSAC's quality is 1 − ρ(TLSShape)/loss_width² — the shape's rho, so there
        # is one definition (the dedicated early-out MSAC device kernel that summed
        # min(d², loss_width²) instead is gone with the one exact device kernel over
        # `residual_quality`).
        @testset "MSAC's cost is ρ(TLSShape)" begin
            sh = loss_shape(MsacScoring())
            @test sh === VGC.TLSShape()
            for loss_width in (0.4, 1.0, 6.0), d in range(0.0, 15.0; length = 151)
                sq_loss_width = loss_width^2
                @test VGC.ρ(sh, d^2, sq_loss_width) == min(d^2, sq_loss_width)
                # and the stated identity, max(0, 1 - d²/loss_width²)
                @test 1.0 - VGC.ρ(sh, d^2, sq_loss_width) / sq_loss_width ≈ max(0.0, 1.0 - d^2 / sq_loss_width) atol = 1.0e-14
                @test residual_quality(MsacScoring(), d^2, loss_width) ≈ max(0.0, 1.0 - d^2 / sq_loss_width) atol = 1.0e-14
            end
        end

        # RANSAC is the SAME shape under the other reduction: its count is the
        # indicator `d² < loss_width²`, which is exactly `weight(TLSShape) = dρ/ds`
        # of `min(s, loss_width²)`. An RSDR-local `CountShape` used to state this
        # with a ρ of its own; deleted, because VGC's TLS shape already says it and a
        # shape that implements only ρ inherits the rest of the loss interface by
        # accident.
        @testset "RANSAC's count is weight(TLSShape)" begin
            for loss_width in (0.4, 1.0, 6.0), d in range(0.0, 15.0; length = 151)
                sq_loss_width = loss_width^2
                @test VGC.weight(VGC.TLSShape(), d^2, sq_loss_width) == (d^2 < sq_loss_width ? 1.0 : 0.0)
                @test residual_quality(RansacScoring(), d^2, loss_width) == (d^2 < sq_loss_width ? 1.0 : 0.0)
            end
            # RANSAC's quality is that weight directly, so it has no `loss_shape`
            # at all — nothing would consume one.
            @test_throws MethodError loss_shape(RansacScoring())
        end

        # Marginalising the labels gives a log-sum-exp in (sigma, outlier half-width)
        # jointly, which
        # a one-width rho cannot express — so these two must have NO shape.
        @testset "the label-marginalising members have no shape" begin
            @test_throws MethodError loss_shape(MlesacScoring())
            @test_throws MethodError loss_shape(MagsacScoring())
        end
    end

    # VGC's own shape-math invariants — totality, ψ = r·w, w = dρ/ds, and the lift's
    # refusal — used to be pinned here. They are gone: all four now live in
    # VisualGeometryCore's test/lo_schedules_tests.jl, which is where a VGC
    # regression should be reported. `irls_weight` in particular no longer exists to
    # assert about (it was an exact alias of `weight` and VGC deleted it), and ψ is
    # now DEFINED as `r * weight(...)`, so pinning that relation here would restate a
    # definition. What RSDR actually depends on is pinned above: that ρ saturates at
    # the squared loss width, which is the depth `residual_quality` divides by.
end
