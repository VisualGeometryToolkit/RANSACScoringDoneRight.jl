# =============================================================================
# selection.jl — this package's grid families on VisualGeometryCore's pool
# selection contract (`selection_staging`, `stage_selection`,
# `select_on_grid!`): one staged batch of pairs (`stage_ig_shared_from_pairs`)
# shared by every grid family, then THE grid verb `grid_evaluate` (or
# SIMFIT++'s host loop) per scoring. Bit-identical to the evaluation engine's
# former per-scoring dispatch.
# =============================================================================

# Every grid family of this package reads the one staging: the WIDTH-FREE
# loss-width families (RANSAC/MSAC at a threshold are VisualGeometryCore's
# scorers and select through its generic fallback), the marginals, SIMFIT++.
const _WidthFreeLossWidth = Union{RansacScoring{<:Any, Nothing}, MsacScoring{<:Any, Nothing}, MlesacScoring,
                                  MagsacScoring, WelschCostScoring, SmoothTLSScoring, GemanScoring}
const _RsdrGridScoring = Union{StudentTUniformScoring, _WidthFreeLossWidth, LossSpec, ScaleMarginal, SimFitPP}
selection_staging(::_RsdrGridScoring) = :rsdr

# The staged batch: the device staging of the pairs' residual columns, and the
# pairs themselves (SIMFIT++ reads their pixel residuals on the host).
stage_selection(::Val{:rsdr}, pairs, residuals; backend = KernelAbstractions.CPU()) =
    (; staged = stage_ig_shared_from_pairs(pairs, residuals; backend), pairs)

"""
    pair_chunk_budget(M_max, T) -> Int

Pairs per chunk of a device selection grid over `T` columns of pools with up
to `M_max` models: the chunk bounds the score block (Float64, pairs × M_max ×
T) plus argmax's broadcast temporary under ~6 GiB, clamped to [1, 256].
T-aware, so a wide miscalibration grid (T up to ~600) stays in device memory
and narrow validation grids keep the chunk large.
"""
pair_chunk_budget(M_max::Integer, T::Integer) =
    clamp(fld(6_000_000_000, max(1, M_max) * max(1, T) * 16), 1, 256)

# A device grid is ONE `grid_evaluate` over the staged batch. The mixture
# marginal of a redescending loss (`LossSpec`) carries neither the minimal
# sample size nor the codimension, so they are the problem's, and it is
# evaluated in Float64 (the protocol's precision for the marginal-Welsch sweep);
# a quadrature marginal sizes its own chunk.
function select_on_grid!(idxs::AbstractMatrix{Int32}, problem_type::Type, scoring::LossSpec, grid::AbstractVector,
                         staged; kw...)
    _, i = grid_evaluate(Float64, scoring, staged.staged, grid; model_dof = minimal_sample_size(problem_type),
                         dg = codimension(problem_type), return_idxs = true)
    return copyto!(idxs, i)
end
function select_on_grid!(idxs::AbstractMatrix{Int32}, ::Type, scoring::ScaleMarginal, grid::AbstractVector, staged;
                         kw...)
    _, i = grid_evaluate(scoring, staged.staged, grid; return_idxs = true)
    return copyto!(idxs, i)
end
function select_on_grid!(idxs::AbstractMatrix{Int32}, ::Type, scoring::Union{StudentTUniformScoring, _WidthFreeLossWidth},
                         grid::AbstractVector, staged; kw...)
    _, i = grid_evaluate(scoring, staged.staged, grid; return_idxs = true,
                         pair_chunk_size = pair_chunk_budget(staged.staged.M_max, length(grid)))
    return copyto!(idxs, i)
end

# SIMFIT++ over the initial-loss-width grid `grid` (the pool's residual unit):
# per pair the models' residuals in PIXELS (`pixel_residuals`, once per pair,
# shared by every column); for every initial guess `simfitpp_select`
# re-estimates the loss width (the reference's pixel bounds untouched) and
# selects the MSAC argmax at the estimate. Only the initial guess crosses
# frames, through the pair's mean pool-frame-per-pixel scale
# (`residual_unit_scale`). `rng_of(fingerprint, t)` is the caller's generator
# per (pair content, column) — `fingerprint = (num_points, num_models,
# gt_variance)` — so the grid is thread-schedule independent.
function select_on_grid!(idxs::AbstractMatrix{Int32}, ::Type, cfg::SimFitPP, grid::AbstractVector, staged;
                         rng_of, kw...)
    pairs = staged.pairs
    T = length(grid)
    Threads.@threads for p in eachindex(pairs)
        pair = pairs[p]
        Dpx = pixel_residuals(pair)
        s = residual_unit_scale(pair)
        fp = (num_points(pair), num_models(pair), pair.gt_variance)   # the pair's content fingerprint
        @inbounds for t in 1:T
            j, _, _ = simfitpp_select(Dpx, grid[t] / s, cfg; rng = rng_of(fp, t))
            idxs[p, t] = j
        end
    end
    return idxs
end
