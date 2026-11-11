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
    @test (t.pot_threshold_quantile, t.min_excess, t.finite_shape_limit) === (0.9, 5, -1.0e-3)   # the values the code held
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

# The owner's ruling: TailRules' `threshold_quantile` is `pot_threshold_quantile`, the quantile level of the complement's
# residuals at which the peaks-over-threshold threshold u is set (Coles 2001, §4.3). The old name is refused, naming the
# new one, as a keyword (TailRules and the two extreme-value models that forward to it) and as a TOML key; the new one is
# accepted, and the default TailRules is the file's.
@testitem "config_defaults: pot_threshold_quantile; the retired threshold_quantile refused, as keyword and TOML key" begin
    using Test
    using TOML: TOML
    using RANSACScoringDoneRight: TailRules, ExtremeValueHalfwidth, GeneralizedParetoOutliers
    import RANSACScoringDoneRight as RSDR
    @test TailRules(; pot_threshold_quantile = 0.8).pot_threshold_quantile === 0.8
    @test TailRules().pot_threshold_quantile === 0.9
    @test ExtremeValueHalfwidth(; pot_threshold_quantile = 0.85).rules.pot_threshold_quantile === 0.85
    @test hasfield(TailRules, :pot_threshold_quantile) && !hasfield(TailRules, :threshold_quantile)
    for f in (() -> TailRules(; threshold_quantile = 0.9), () -> ExtremeValueHalfwidth(; threshold_quantile = 0.9),
              () -> GeneralizedParetoOutliers(; threshold_quantile = 0.9))
        @test_throws r"pot_threshold_quantile" f()
        @test_throws ArgumentError f()
    end
    toml = TOML.parsefile(joinpath(pkgdir(RSDR), "configs", "default.toml"))
    @test haskey(toml["tail_rules"], "pot_threshold_quantile") && !haskey(toml["tail_rules"], "threshold_quantile")
    @test_throws r"pot_threshold_quantile" RSDR._check_retired_keys(Dict("tail_rules" => Dict("threshold_quantile" => 0.9)))
    @test RSDR._check_retired_keys(toml) === toml
end
