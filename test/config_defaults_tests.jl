# The defaults of record (configs/default.toml, the stack's defaults mechanism): every field of
# `TailRules`, the scale marginals' log-scale span and every redescending shape's canonical depth are
# read there — the file covers each, and the code holds no second copy of the values.
@testitem "config_defaults: configs/default.toml covers TailRules, the scale span and the depths" begin
    using Test
    using TOML: TOML
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: TailRules, ExtremeValueHalfwidth
    using VisualGeometryToolkitCore: WelschShape, GemanMcclureShape, SmoothTLSShape
    import RANSACScoringDoneRight as RSDR
    toml = TOML.parsefile(joinpath(pkgdir(RANSACScoringDoneRight), "configs", "default.toml"))
    # completeness: every TailRules field is a key, and the default TailRules is the file's
    @test Set(keys(toml["tail_rules"])) == Set(String.(fieldnames(TailRules)))
    t = TailRules()
    @test all(getfield(t, f) == toml["tail_rules"][String(f)] for f in fieldnames(TailRules))
    @test ExtremeValueHalfwidth().rules == t
    @test (t.threshold_quantile, t.min_excess, t.finite_shape_limit) === (0.9, 5, -1.0e-3)   # the values the code held
    # the scale marginals' span (t0 ± span) and the canonical depths, by the shapes' loss tags
    @test toml["scale_marginal"]["log_scale_span"] === 7.0
    depth = toml["inlier_potentials"]["depth"]
    @test Set(keys(depth)) == Set(["welsch", "geman_mcclure", "smooth_tls"])
    for (K, tag, v) in ((WelschShape, "welsch", 0.5), (GemanMcclureShape, "geman_mcclure", 0.5), (SmoothTLSShape, "smooth_tls", 1.0))
        @test depth[tag] === v
    end
    # the code reads the file (the mechanism's lookup, which throws on a missing key)
    @test RSDR._config_default("tail_rules", "min_excess") == 5
    @test_throws Exception RSDR._config_default("tail_rules", "no_such_key")
end
