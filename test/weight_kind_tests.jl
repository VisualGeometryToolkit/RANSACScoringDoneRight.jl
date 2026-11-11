# weight_kind_tests.jl — what a robust result's per-point weights mean under
# each scorer this package defines (VisualGeometryToolkitCore's `weight_kind`
# trait, the robust-estimation contract §12). The Overview states it: every
# loss-width family is an `AbstractThresholdScoring`, so its weights are the
# 0/1 consensus membership (`BinaryWeights`), and none defines an
# `outlier_halfwidth` that would give it posterior values under that kind; the
# Student-t + uniform scorer, with or without this package's outlier models,
# gives posterior inlier probabilities (`PosteriorWeights`).

@testitem "weight_kind" begin
    using Test, RANSACScoringDoneRight
    using VisualGeometryToolkitCore: weight_kind, BinaryWeights, PosteriorWeights, AbstractThresholdScoring,
        StudentTUniformScoring, JeffreysPrior, FlatLabelPrior, outlier_halfwidth

    # every loss-width family, width-free and at a loss width, through its tag and its type
    for tag in ("ransac", "msac", "mlesac", "magsac", "welsch_cost", "smooth_tls", "geman"), dg in (1, 2)
        s = loss_width_scoring(tag, dg)
        @test s isa AbstractThresholdScoring
        @test @inferred(weight_kind(typeof(s))) === BinaryWeights()
    end
    for F in (MlesacScoring, MagsacScoring, WelschCostScoring, SmoothTLSScoring, GemanScoring)
        s = F(2; threshold = 0.5)
        @test weight_kind(typeof(s)) === BinaryWeights()
        @test outlier_halfwidth(s) === nothing   # no half-width: the weights stay the membership
    end

    # the Student-t + uniform scorer, at a half-width and with each outlier model
    priors = (; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
    for model in (6.0, ExtremeValueHalfwidth(), GeneralizedParetoOutliers(), OutlierTerm((k, z) -> 0.0))
        @test weight_kind(typeof(StudentTUniformScoring(2, model; priors...))) === PosteriorWeights()
    end
end
