# A NaN or ±Inf residual is an OUTLIER on every scoring path.
#
# The exact host scorers (the per-pair `score_magsac`/`score_gau`/`score_msac`,
# now `best_pool_model(scoring, pair, loss_width)`; `_fill_grid_exact!`), the host
# histogram (`bin_residuals_stereo`) and the RANSAC/MSAC device kernels always
# treated it so. Two paths did not:
#   * the MAGSAC++ lookup-table scorer (since deleted) SKIPPED it — zero loss,
#     the loss of a perfect inlier, in a sum it minimised;
#   * the device histogram (`histogram_pm_score_kernel!`, every histogram
#     family of `grid_evaluate` on a `LossWidthScoring`) sent NaN to bin 1, the perfect-inlier
#     weight, and +Inf into `floor(Int, Inf)` (an InexactError on CPU()).

@testitem "nonfinite_residuals" setup = [PoolFixtures] begin
    using Test, Random
    using KernelAbstractions
    using RANSACScoringDoneRight
    using RANSACScoringDoneRight: MagsacScoring, MlesacScoring, WelschCostScoring, SmoothTLSScoring, GemanScoring,
        RansacScoring, MsacScoring,
        stage_ig_shared_from_pairs, Exact, LogBinned,
        histogram_pm_score_kernel!, exact_pm_score_kernel!, _build_score_weight_matrix,
        _support_sq
    using VisualGeometryToolkitCore: best_pool_model   # the 4-argument method is this package's

    # One pair whose M models share their residuals except that model j has
    # 3(j−1) of its INLIERS replaced by `bad(j)`. Model 1 keeps every inlier, so it
    # is the best model at every loss width (ties included: both paths keep the first).
    function _nonfinite_pair(bad; n = 200, M = 6, seed = 7)
        rng = MersenneTwister(seed)
        base = Float32[i <= 150 ? abs(0.3f0 * randn(rng)) : 20 + 20 * rand(rng) for i in 1:n]
        D = repeat(base, 1, M)
        for j in 1:M, i in 1:3(j - 1)
            D[i, j] = bad(j)
        end
        (identity_pool(D; errors_r = collect(1.0:M), errors_t = zeros(M)), D)
    end

    const _NF_LOSS_WIDTHS = [0.3, 1.0, 3.0, 10.0]
    _nonfinite(j) = isodd(j) ? NaN32 : Inf32        # both kinds, alternating by model
    _far(j) = 1f3                                    # a finite outlier far past every loss width

    @testset "non-finite residuals are outliers" begin
        pair_nf, pair_far = _nonfinite_pair(_nonfinite), _nonfinite_pair(_far)

        @testset "per-pair scorers, every loss-width family" begin
            # A non-finite residual scores as a far finite one does — so the picks
            # agree, and model 1 wins.
            for loss_width in _NF_LOSS_WIDTHS, sc in (RansacScoring(), MsacScoring(), MlesacScoring(),
                                      MagsacScoring(), WelschCostScoring(),
                                      SmoothTLSScoring(), GemanScoring())
                @test best_pool_model(sc, pair_nf..., loss_width) == best_pool_model(sc, pair_far..., loss_width) == 1
            end
        end

        @testset "device histogram, per model score (bit-exact)" begin
            # A column with trailing NaN/+Inf scores EXACTLY as the same column
            # without them: the skip adds nothing, and the finite terms are summed
            # in the same order. Checked for every histogram family, both precisions.
            rng = MersenneTwister(3)
            fin = sort!(Float32[abs(0.5f0 * randn(rng))^2 for _ in 1:40])
            cols_fin = [fin, fin .* 1.5f0]
            cols_bad = [vcat(c, Float32[Inf, NaN, NaN]) for c in cols_fin]   # sorted: Inf < NaN
            loss_widths = [0.3, 1.0, 3.0]
            nb = 400; d2_min = 0.01 * minimum(loss_widths)^2; d2_max = 100 * maximum(loss_widths)^2
            backend = KernelAbstractions.CPU()
            kern = histogram_pm_score_kernel!(backend)
            function scores(sc, cols, S)
                K, M = length(cols[1]), length(cols)
                W = _build_score_weight_matrix(sc, loss_widths, nb, d2_min, d2_max)
                out = zeros(1, M, length(loss_widths))
                kern(out, reduce(vcat, cols), Int64[0, K * M], Int32[K], Int32[M], vec(W),
                     Int32(nb), S(log(d2_min)), S(nb) / S(log(d2_max / d2_min)),
                     Int32(length(loss_widths)), Int32(0); ndrange = (1, M))
                KernelAbstractions.synchronize(backend)
                out
            end
            for sc in (MlesacScoring(), MagsacScoring(), WelschCostScoring(),
                       SmoothTLSScoring(), GemanScoring()), S in (Float64, Float32)
                @test scores(sc, cols_bad, S) == scores(sc, cols_fin, S)
            end
            # the Exact kernel likewise skips them — every loss-width family
            kx = exact_pm_score_kernel!(backend)
            function xscores(sc, cols)
                K, M = length(cols[1]), length(cols)
                out = zeros(1, M, length(loss_widths))
                kx(out, reduce(vcat, cols), Int64[0, K * M], Int32[K], Int32[M], loss_widths,
                   Int32(length(loss_widths)), Int32(0), sc, _support_sq(sc, maximum(loss_widths)); ndrange = (1, M))
                KernelAbstractions.synchronize(backend)
                out
            end
            for sc in (RansacScoring(), MsacScoring(), MlesacScoring(), MagsacScoring(),
                       WelschCostScoring(), SmoothTLSScoring(), GemanScoring())
                @test xscores(sc, cols_bad) == xscores(sc, cols_fin)
            end
        end

        @testset "device histogram, per selection (grid_evaluate)" begin
            # End to end through staging: NaN/Inf sort last and are skipped; MAGSAC++
            # gives a far finite residual exactly zero too, so the picks agree, and
            # model 1 wins at every loss width.
            for S in (Float64, Float32), agg in (LogBinned(), Exact())
                picks = map((pair_nf, pair_far)) do p
                    shared = stage_ig_shared_from_pairs([p[1]], [p[2]]; backend = KernelAbstractions.CPU())
                    last(grid_evaluate(S, MagsacScoring(), shared, _NF_LOSS_WIDTHS; aggregation = agg,
                                       return_idxs = true))
                end
                @test picks[1] == picks[2]
                @test all(==(1), picks[1])
            end
        end
    end
end
