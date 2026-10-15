# =============================================================================
# IG / Jeffreys marginal sweep — KernelAbstractions port (CPU + CUDA backends)
# =============================================================================
#
# `grid_evaluate(::StudentTUniformScoring, …)` scores a pool of pairs, staged once into
# the flat `IGSharedData` layout by `stage_ig_shared_from_pairs`, over a grid of
# outlier half-widths: `ig_sweep_kernel!` runs the sort-and-sweep marginal score
# (the same `_sweep_score` arithmetic as sweep.jl) for every (pair, model,
# outlier half-width),
# and a gather kernel pulls the chosen model's error.
#
# (The GPU bootstrap tuner that lived here — `tune_ka`/`evaluate_ka`, their
#  `IGTuneBuffers`/`IGEvalBuffers` and `alloc_*_buffers`, the
#  `reduce_argmax_and_gather!` kernel, `stage_ig_shared_ka(method)` and the
#  `_method_*` pool accessors — is deleted with the evaluation engine's
#  `evaluate_val` driver that was its only caller. The tuning constants come
#  from LocalOptimizationDoneRight's two-knob sidecars.)

# =============================================================================
# IGSharedData — flat layout of every pair in a pool
# =============================================================================
"""
    IGSharedData{Backend, F32A, F64A, IntA}

Backend-agnostic flat-layout view of every `PrecomputedPair` in a pool. Built
once per pool by [`stage_ig_shared_from_pairs`](@ref) and reused by every
grid evaluator.

Memory layout — all distinct pairs flattened:
- `sorted_sq_flat::F32A` (a vector of the pool's residual eltype on backend)
   contiguous concatenation of all (K_i × M_i) sorted_sq matrices in
   column-major order: pair 1 model 1 (length K_1), pair 1 model 2, …,
   pair 1 model M_1, pair 2 model 1, …
- `pair_offset::IntA` (length P+1)
   `pair_offset[p]` = absolute starting index of pair p in `sorted_sq_flat`.
   `pair_offset[P+1]` = total length (sentinel).
- `K_per_pair::IntA` (length P)  — # residuals for each pair
- `M_per_pair::IntA` (length P)  — # models for each pair
- `M_max::Int`                   — max # of models across all pairs (kernel grid sizing)
- `errors_imc_flat::F64A`          — concatenated per-model selection imc = `pool_imc` (H: Δrot; E/F: max(Δrot,Δtrans)), indexed by `errors_offset[p] + j`
- `errors_offset::IntA`          — `errors_offset[p]` = start of pair p's imc segment

P is the number of pairs in the pool.
"""
struct IGSharedData{Backend, F32A, F64A, I64A, I32A}
    backend::Backend
    sorted_sq_flat::F32A
    pair_offset::I64A            # length P+1, Int64 (total residuals can exceed 2.1G)
    K_per_pair::I32A             # length P, Int32
    M_per_pair::I32A             # length P, Int32
    M_max::Int
    errors_imc_flat::F64A
    errors_offset::I64A          # length P+1, Int64
end

# =============================================================================
# Kernel
# =============================================================================

"""
    ig_sweep_kernel!(scores_out, sorted_sq_flat, pair_offset, K_per_pair, M_per_pair,
                     lg_table, log_two_outlier_halfwidth_grid, model_dof, dg, prior, p_offset)

One work-item per (pair, model, grid-point): ONE sequential sweep over the
model's K sorted squared residuals, writing the best partition score
`max_k S(k)` to `scores_out[p_local, m, t]` (Float64). `S(k)` is sweep.jl's
`_partition_score` — the one per-partition formula every sweep shares, host and
device — at the grid point's `log(2·outlier_halfwidth)`. RSS accumulates in Float64. Work-items
beyond a pair's model count write nothing (the chunked driver pre-fills −Inf).
"""
@kernel function ig_sweep_kernel!(
        scores_out,              # (chunk, M_max, T) Float64
        @Const(sorted_sq_flat),
        @Const(pair_offset),
        @Const(K_per_pair),
        @Const(M_per_pair),
        @Const(lg_table),
        @Const(log_two_outlier_halfwidth_grid),   # (T,) log(2·outlier_halfwidth_t)
        model_dof::Int32, dg::Int32, prior::IGPrior, p_offset::Int32)
    p_local, m, t = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)
    inbounds_thread = p_local <= size(scores_out, 1) && t <= size(scores_out, 3) &&
                      p <= length(M_per_pair)
    M = inbounds_thread ? Int(M_per_pair[p]) : 0
    if m <= M
        K     = Int(K_per_pair[p])
        base  = pair_offset[p] + (Int64(m) - Int64(1)) * Int64(K)
        log_two_outlier_halfwidth = log_two_outlier_halfwidth_grid[t]
        dg_f  = Float64(dg)
        RSS = 0.0
        bS  = -Inf
        @inbounds for k in 1:K
            RSS += Float64(sorted_sq_flat[base + k])
            if k > model_dof
                S = _partition_score(prior, lg_table, RSS, k, model_dof, dg_f, K, log_two_outlier_halfwidth)
                S > bS && (bS = S)
            end
        end
        @inbounds scores_out[p_local, m, t] = bS
    end
end

# =============================================================================
# Marginal grid evaluation
# =============================================================================
#
# The marginal grid, on CPU() or a GPU: `ig_sweep_kernel!` fills each
# pair-chunk's (chunk, M_max, T) score block and the SHARED chunked driver
# (`_chunked_argmax_eval`, marg_sweep_ka.jl — the one every grid evaluator runs
# on) takes the model argmax and gathers the selected model's error.
#
# (The marginal family has no loss width: its grid is of outlier half-widths;
# `decision_threshold` converts one to the residual its MAP rule flips at.)

# (`free_ka_shared!` is deleted: the staged `IGSharedData` belongs to the
#  caller, which drops it when done. The grid evaluators free their OWN
#  function-local device buffers eagerly through `_free_local!`
#  (marg_sweep_ka.jl); the `_try_unsafe_free!` they used to call freed nothing
#  on any backend.)

"""
    grid_evaluate(scoring::StudentTUniformScoring, staged::IGSharedData, outlier_halfwidths;
                  pair_chunk_size = 256, return_idxs = false) -> Matrix{Float64}

The Student-t marginal grid (the IG/Jeffreys sort-and-sweep) on the staged
pool's backend. It returns a `(P, T)` matrix of each pair's error at the model `scoring`
selects at each outlier half-width in `outlier_halfwidths` (`NormalizedImageFrame`
units). The grid REPLACES the scoring's own outlier half-width; its minimal
sample size, codimension and scale prior are the scoring's. With
`return_idxs`, also the `(P, T)` Int32 selected model indices.

Runs `ig_sweep_kernel!` on the shared chunked driver (`_chunked_argmax_eval`):
pair-chunks of `pair_chunk_size` bound the score block, and the model argmax is
the shared `argmax_models_kernel!`, which takes the FIRST strictly best model
and SKIPS a NaN score. The device sweep scores the model-certain marginal with
the flat label prior only: a `StudentTUniformScoring{Predictive}` or a scoring with
`profile_prior = true` is an `ArgumentError`, never scored as something else.
"""
function grid_evaluate(scoring::StudentTUniformScoring{P}, staged::IGSharedData,
                       outlier_halfwidths::AbstractVector{<:Real};
                       pair_chunk_size::Int = 256, return_idxs::Bool = false) where {P}
    P === Nothing || throw(ArgumentError(
        "grid_evaluate: the device marginal sweep is model-certain; got StudentTUniformScoring{$P}"))
    scoring.profile_prior && throw(ArgumentError(
        "grid_evaluate: the device marginal sweep has no profiled label prior; got profile_prior = true"))
    backend = staged.backend
    Pn = length(staged.K_per_pair)
    T  = length(outlier_halfwidths)
    Mm = Int(staged.M_max)
    cs = max(1, min(pair_chunk_size, Pn))
    prior = scale_prior(scoring)
    model_dof = minimal_sample_size(scoring); dg = codimension(scoring)
    # The log Γ table up to the largest staged pair (its entries do not depend on
    # the table's length, so any length covering every pair gives the same sweep).
    N_max = Pn == 0 ? 0 : Int(maximum(Array(staged.K_per_pair)))

    lg_table_d = _ka_upload(backend, Float64, _build_lg_table(N_max, dg; alpha0 = prior.alpha0))
    log_two_outlier_halfwidth_d = _ka_upload(backend, Float64,
                                             [log(2.0 * Float64(h)) for h in outlier_halfwidths])

    sweep = _ka_kernel(ig_sweep_kernel!, backend, 3)
    out = _chunked_argmax_eval(staged, T; cs, return_idxs) do scores_pmt, s, n
        sweep(scores_pmt, staged.sorted_sq_flat, staged.pair_offset,
              staged.K_per_pair, staged.M_per_pair, lg_table_d, log_two_outlier_halfwidth_d,
              Int32(model_dof), Int32(dg), prior, Int32(s); ndrange = (n, Mm, T))
        KernelAbstractions.synchronize(backend)
    end
    _free_local!(lg_table_d, log_two_outlier_halfwidth_d)
    return out
end

"""
    stage_ig_shared_from_pairs(pairs, residuals; backend) -> IGSharedData

Stage a pool once for every grid evaluator. The inputs are the light records `pairs`
(`AbstractVector{<:PoolPair}`) with their residual matrices `residuals`
(`residuals[p]` is pair `p`'s `num_points × num_models` Float32 matrix, in
`NormalizedImageFrame` units — the caller loads them; staging opens no file),
each pair precomputed by [`precompute_pair`](@ref) into the flat
[`IGSharedData`](@ref) layout on `backend`. No prior fit, no width grid baked in.
"""
function stage_ig_shared_from_pairs(pairs::AbstractVector{<:PoolPair},
                                    residuals::AbstractVector{<:AbstractMatrix{Float32}};
                                    backend = KernelAbstractions.CPU())
    length(pairs) == length(residuals) || throw(DimensionMismatch(
        "stage_ig_shared_from_pairs: $(length(pairs)) pairs and $(length(residuals)) residual matrices"))
    foreach(_check_residuals, pairs, residuals)
    P = length(pairs)
    # Lay out the flat buffer from per-pair SIZES (cheap, no sort) so the sorted
    # matrices never accumulate. Staging then builds the flat one pair at a time
    # and frees each `pp` immediately — peak host memory is flat_h + the raw
    # pairs, NOT flat_h + all-P sorted copies + raw (≈1.5× bigger, which OOMs on
    # the 4K val pool). Output is byte-identical to the all-at-once version.
    K_arr = Vector{Int32}(undef, P)
    M_arr = Vector{Int32}(undef, P)
    pair_offset   = Vector{Int64}(undef, P + 1)
    errors_offset = Vector{Int64}(undef, P + 1)
    pair_offset[1] = 0
    errors_offset[1] = 0
    for p in 1:P
        n, M = num_points(pairs[p]), num_models(pairs[p])
        K_arr[p] = n
        M_arr[p] = M
        pair_offset[p + 1]   = pair_offset[p]   + Int64(n) * M
        errors_offset[p + 1] = errors_offset[p] + Int64(M)
    end
    total_residuals = pair_offset[end]
    total_errors    = errors_offset[end]

    Tsq = Float32                  # the residual matrices' precision
    sorted_sq_flat_h = Vector{Tsq}(undef, total_residuals)
    errors_imc_flat_h  = Vector{Float64}(undef, total_errors)
    # Disjoint output slices per pair ⇒ thread-safe; each `pp` (one pair's sorted
    # matrix) is freed at the end of its iteration, so they never pile up.
    Threads.@threads for p in 1:P
        pp     = precompute_pair(pairs[p], residuals[p])
        base   = pair_offset[p]
        ebase  = errors_offset[p]
        n_res  = pp.n
        n_mod  = length(pp.errors_imc)
        for jm in 1:n_mod
            col = view(pp.sorted_sq, 1:n_res, jm)
            @inbounds for k in 1:n_res
                sorted_sq_flat_h[base + (jm - 1) * n_res + k] = col[k]
            end
            @inbounds errors_imc_flat_h[ebase + jm] = pp.errors_imc[jm]
        end
    end

    M_max = maximum(M_arr)

    sorted_sq_flat  = _ka_upload(backend, Tsq, sorted_sq_flat_h)
    errors_imc_flat = _ka_upload(backend, Float64, errors_imc_flat_h)
    pair_offset_d   = _ka_upload(backend, Int64,   pair_offset)
    K_per_pair_d    = _ka_upload(backend, Int32,   K_arr)
    M_per_pair_d    = _ka_upload(backend, Int32,   M_arr)
    errors_offset_d = _ka_upload(backend, Int64,   errors_offset)

    IGSharedData(backend,
                   sorted_sq_flat, pair_offset_d, K_per_pair_d, M_per_pair_d,
                   Int(M_max), errors_imc_flat, errors_offset_d)
end
