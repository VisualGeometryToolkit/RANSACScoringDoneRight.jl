# The loss-width families' two aggregations, Exact and LogBinned(200 per decade), are two
# ALGORITHMS over ONE per-residual quality `residual_quality(scoring, d², loss_width)`:
#
#   * Exact is the plain sum of the quality over every finite residual — pinned
#     against a host loop in the same (sorted) order, bit for bit;
#   * LogBinned evaluates the same quality at log-bin centres, so per residual
#     it is off by at most the quality's variation across that residual's bin.
#     Every family's quality is non-increasing in d², so that variation is
#     q(bin's low edge) − q(bin's high edge); summed over a model's residuals it
#     bounds |S_binned − S_exact| (plus Float32 rounding of the weight table);
#   * on a pool the two pick the same model almost everywhere.

@testitem "loss_width_aggregation" setup = [PoolFixtures] begin
    using Test, Random
    using KernelAbstractions
    using RansacScoringDoneRight
    using RansacScoringDoneRight: stage_ig_shared_from_pairs,
        Exact, LogBinned, RansacScoring, MsacScoring, MlesacScoring, MagsacScoring,
        WelschCostScoring, SmoothTLSScoring, GemanScoring, residual_quality, _support_sq,
        exact_pm_score_kernel!, histogram_pm_score_kernel!, _build_score_weight_matrix

    const _AGG_FAMILIES = (RansacScoring(), MsacScoring(), MlesacScoring(), MagsacScoring(),
                           WelschCostScoring(), SmoothTLSScoring(), GemanScoring())

    _agg_pool(rng; P = 12, n = 150, M = 16) = map(1:P) do _
        D = mixture_residuals(rng, n, M)
        (identity_pool(D; errors_r = 10 .* rand(rng, M), errors_t = zeros(M)), D)
    end

    @testset "loss-width families: Exact and LogBinned over one quality" begin
        rng = MersenneTwister(17)
        loss_widths = collect(10.0 .^ range(log10(0.3), 1.0; length = 7))
        be = KernelAbstractions.CPU()

        @testset "per model score (one pair)" begin
            pair, D = only(_agg_pool(rng; P = 1, n = 120, M = 6))
            sh = stage_ig_shared_from_pairs([pair], [D]; backend = be)
            flat = sh.sorted_sq_flat; K = Int(sh.K_per_pair[1]); M = Int(sh.M_per_pair[1])
            T = length(loss_widths)
            # the LogBinned grid exactly as grid_evaluate builds it (Float64 bins)
            d2_min, d2_max = 0.01 * minimum(loss_widths)^2, 100 * maximum(loss_widths)^2
            nb = max(64, ceil(Int, 200 * log10(d2_max / d2_min)))
            lspacing = log(d2_max / d2_min) / nb
            edge(b) = exp(log(d2_min) + b * lspacing)           # upper edge of bin b
            for sc in _AGG_FAMILIES
                Sx = zeros(1, M, T)
                exact_pm_score_kernel!(be)(Sx, flat, sh.pair_offset, sh.K_per_pair, sh.M_per_pair,
                    loss_widths, Int32(T), Int32(0), sc, _support_sq(sc, maximum(loss_widths)); ndrange = (1, M))
                W = _build_score_weight_matrix(sc, loss_widths, nb, d2_min, d2_max)
                Sb = zeros(1, M, T)
                histogram_pm_score_kernel!(be)(Sb, flat, sh.pair_offset, sh.K_per_pair, sh.M_per_pair,
                    vec(W), Int32(nb), log(d2_min), nb / log(d2_max / d2_min), Int32(T), Int32(0);
                    ndrange = (1, M))
                KernelAbstractions.synchronize(be)
                for m in 1:M, (t, loss_width) in enumerate(loss_widths)
                    col = Float64.(flat[(m - 1) * K + 1:m * K])     # sorted, as the kernel reads it
                    # Exact == the plain host sum, same order. EXACT on purpose, and
                    # a canary: two drivers (the KA kernel, this foldl) accumulate the
                    # SAME scalar in the SAME order, so bit-identity is the contract —
                    # held at max |Δ| 0.0 across the 33 CPU/CUDA parity configurations.
                    # If it ever fails (a reassociating compiler, a fused multiply-add),
                    # MEASURE the margin and loosen TO it; never re-run until green.
                    @test Sx[1, m, t] === foldl(+, (residual_quality(sc, d2, loss_width) for d2 in col); init = 0.0)
                    # LogBinned within the bins' quality variation
                    bound = 0.0; mag = 0.0
                    for d2 in col
                        b = d2 > 0 ? clamp(floor(Int, (log(d2) - log(d2_min)) * nb / log(d2_max / d2_min)) + 1, 1, nb) : 1
                        lo = b == 1 ? 0.0 : edge(b - 1)
                        hi = b == nb ? Inf : edge(b)
                        bound += residual_quality(sc, lo, loss_width) - (isfinite(hi) ? residual_quality(sc, hi, loss_width) : 0.0)
                        mag += abs(residual_quality(sc, d2, loss_width))
                    end
                    @test abs(Sb[1, m, t] - Sx[1, m, t]) <= bound + 1e-6 * (mag + 1)
                end
            end
        end

        @testset "per selection (a pool)" begin
            pool = _agg_pool(rng)
            sh = stage_ig_shared_from_pairs(first.(pool), last.(pool); backend = be)
            for sc in _AGG_FAMILIES
                _, jx = grid_evaluate(Float64, sc, sh, loss_widths; aggregation = Exact(), return_idxs = true)
                _, jb = grid_evaluate(Float64, sc, sh, loss_widths; aggregation = LogBinned(), return_idxs = true)
                _, jb32 = grid_evaluate(sc, sh, loss_widths; return_idxs = true)   # the default: Float32, LogBinned
                @test count(jx .== jb) / length(jx) >= 0.95
                @test count(jx .== jb32) / length(jx) >= 0.95
            end
        end
    end
end
