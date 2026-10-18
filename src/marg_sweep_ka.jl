# =============================================================================
# marg_sweep_ka.jl — device scale-marginal scoring.  Two evaluators share this
# file and ONE quadrature: the FULL-DATA scale marginals (Welsch / smooth-TLS,
# `grid_evaluate(::ScaleMarginal, …)`; this header) and the Track-B (Stage 6) MIXTURE marginal
# (`grid_evaluate(::LossSpec, …)`; second header below).
#
# The loss width is integrated out over t = log sq_loss_width by the fixed-node
# TRAPEZOID logsumexp of `_trapezoid_logevidence` on a `LossQuad` grid
# (inlier_losses.jl states the rule and why). Per node the data term is
# `_accumulate_data!`: Σᵢ Δ·ρ(shape, xᵢ/sq_loss_width_nodes[j], 1), VGC's shape at depth Δ. The full-data marginal is
# the mixture's evidence at k = K with every point counted (K·logZ̃): one
# spec per grid point (`loss_spec(::ScaleMarginal)`, t0 ± L around its prior
# centre), all on one absolute log-scale lattice, so the data term is summed
# ONCE per (pair, model) and each grid point reads its window of it.
#
# The full-data marginal used a 13-node cubic-Hermite spline over t0 ± L until
# 2026-09-30. Against a 40001-node trapezoid on the same domain (synthetic
# pools, 20 draws per K) its log-evidence error was, median / max:
#   Welsch      K = 100: 3e-3 / 0.07   K = 500: 7e-3 / 0.09   K = 2000: 1.5e-2 / 0.94
#   smooth-TLS  K = 100: 4e-3 / 0.31   K = 500: 3e-2 / 0.37   K = 2000: 0.18 / Inf
# and it needed dlogZ̃/dt, taken by finite difference. The trapezoid at
# `_MARG_NG` = 129 nodes, against the same reference on its own (snapped) domain:
#   Welsch      K = 100: 4e-7 / 4e-4   K = 500: 3e-14 / 7e-4  K = 2000: 0 / 0.11
#   smooth-TLS  K = 100: 4e-5 / 8e-3   K = 500: 2e-4 / 1.3e-2 K = 2000: 3e-4 / 1.2
# (smooth-TLS converges as h², its potential has a kink). Model selection on a
# 64-pair × 20-model × 9-grid-point pool agreed with a 2049-node reference at
# all 576 grid cells for both rules. Speed on that pool (CPU(), 32 threads,
# K ≈ 850): Welsch 36 → 71 ms (Float32), smooth-TLS 10.6 → 6.3 ms; summing the data
# term per grid point instead of once on the lattice took 272 ms for Welsch.
#
# Precision-generic (`S`): default Float32 (like the loss-width grid).  Model
# selection reuses `argmax_models_kernel!` (higher evidence ⇒ better); the score
# accumulator stays Float64 (drift-safe).
# =============================================================================

# allocate + copyto! in one step, converting to the target element type when it
# differs.  Every device-table upload in this file goes through here.
@inline function _ka_upload(backend, ::Type{T}, host::AbstractArray) where {T}
    d = KernelAbstractions.allocate(backend, T, size(host)...)
    KernelAbstractions.copyto!(backend, d, eltype(host) === T ? host : T.(host))
    return d
end

# Every kernel here is instantiated through `_ka_kernel(kernel, backend, nd)`
# (`nd` = the ndrange's dimensionality). On the CPU backend KernelAbstractions'
# default workgroup is large, and for these (pairs × models [× grid]) ndranges
# that leaves most threads idle: the full-data marginal kernel took 20.5 ms at the default
# and 1.9 ms at one work-item per workgroup (64 pairs × 20 models, 32 threads).
# One item per workgroup lets the CPU backend spread the items over every
# thread. On a GPU the backend's own choice stands.
_ka_kernel(kernel, backend::KernelAbstractions.CPU, nd::Int) = kernel(backend, ntuple(_ -> 1, nd))
_ka_kernel(kernel, backend, nd::Int) = kernel(backend)

# Eager release of FUNCTION-LOCAL device buffers: GPUArrays' `unsafe_free!` hands
# a buffer's memory back to the device pool now, not whenever the host GC next
# runs — and host memory pressure says nothing about the device, where a grid
# evaluation allocates GiB-scale scratch per pair-chunk. Host `Array`s (the CPU
# backend) have nothing to free.
#
# Only for buffers the calling function allocated, never returns and never
# hands to anything that outlives the call — a freed buffer must not be read
# again. Every call site below is such a buffer: device tables uploaded for one
# evaluation, per-chunk scratch, and the chunk loop's own outputs after their
# host copy-back. The staged `IGSharedData` is the CALLER's and is never freed
# here. (The `_try_unsafe_free!` these call sites used to reach freed nothing:
# its `::AbstractArray` no-op method caught `CuArray` too.)
_free_local!(xs::AbstractArray...) = (foreach(_unsafe_free_one!, xs); nothing)
@inline _unsafe_free_one!(x::GPUArrays.AbstractGPUArray) = GPUArrays.unsafe_free!(x)
@inline _unsafe_free_one!(::AbstractArray) = nothing

# The two device steps of the ONE scale quadrature, shared by both evaluators
# below. The mixture reads one spec's tables (every offset 0); the full-data
# evaluator reads a window of NG nodes of a shared lattice (`oD` into the data
# term and the lattice's logZ̃, `oB` into the grid point's column of the prior
# base and trapezoid weights).
#
# Add one squared residual `x` to the per-node data term of work item `wid`:
# Dⱼ += Δ·ρ(shape, x/sq_loss_width_nodes[j], 1), VGC's unit-width shape at depth
# Δ. `shape` is the loss-shape tag, a singleton the device kernel specialises on.
@inline function _accumulate_data!(Dscratch, wid, x::T, Δ::T, shape, sq_loss_width_nodes,
                                   ng::Int) where {T}
    @inbounds for j in 1:ng
        Dscratch[j, wid] += Δ * VgtkCore.ρ(shape, x / sq_loss_width_nodes[j], one(T))
    end
    return nothing
end

# log ∫ exp(−ψ(t)) dt by the trapezoid logsumexp over the nodes, where
# −ψⱼ = −(Dⱼ + ke·logZ̃ⱼ + baseⱼ) and ltwⱼ = log(trapezoid weight · Δt): two
# scans, max then sum.
@inline function _trapezoid_logevidence(Dscratch, wid, ke::T, logZnodes, base_nodes, ltw,
                                        oD::Int, oB::Int, ng::Int) where {T}
    mx = T(-Inf)
    @inbounds for j in 1:ng
        ll = -(Dscratch[oD + j, wid] + ke * logZnodes[oD + j] + base_nodes[oB + j]) + ltw[oB + j]
        ll > mx && (mx = ll)
    end
    se = zero(T)
    @inbounds for j in 1:ng
        ll = -(Dscratch[oD + j, wid] + ke * logZnodes[oD + j] + base_nodes[oB + j]) + ltw[oB + j]
        se += exp(ll - mx)
    end
    return mx + log(se)
end

# Per (pair p_local, model m): the full-data marginal log-evidence at every grid
# point t. The data term is summed ONCE over the NU-node lattice every grid
# point's nodes lie on (`loss_spec(::ScaleMarginal)`); grid point t reads its
# NG-node window at `win_off[t]`, with its own prior base and weights (column t
# of the (NG × T) tables). Every one of the K residuals enters the data term and
# the normalizer count.
@kernel function marg_evidence_kernel!(
        scores_out,                  # (chunk, M_max, T) Float64
        @Const(sorted_sq_flat),
        @Const(pair_offset), @Const(K_per_pair), @Const(M_per_pair),
        @Const(sq_loss_width_nodes), @Const(logZnodes),       # (NU,) T — the lattice
        @Const(base_nodes), @Const(ltw),          # (NG·T,) T — per grid point
        @Const(win_off),                          # (T,) Int32 — window start − 1
        Dscratch,                    # (NU, W) T — per-work-item accumulator
        Δ::T, NU::Int32, NG::Int32, T_val::Int32, p_offset::Int32, shape) where {T}
    p_local, m = @index(Global, NTuple)
    wid = @index(Global, Linear)
    p = Int(p_local) + Int(p_offset)
    inbounds_thread = p_local <= size(scores_out, 1) && p <= length(M_per_pair)
    M = inbounds_thread ? Int(M_per_pair[p]) : 0
    if m <= M
        nu = Int(NU); ng = Int(NG); K = Int(K_per_pair[p])
        boff = pair_offset[p] + (Int64(m) - Int64(1)) * Int64(K)
        @inbounds for j in 1:nu
            Dscratch[j, wid] = zero(T)
        end
        @inbounds for k in 1:K
            _accumulate_data!(Dscratch, wid, T(sorted_sq_flat[boff + k]), Δ, shape, sq_loss_width_nodes, nu)
        end
        @inbounds for t in 1:Int(T_val)
            scores_out[p_local, m, t] = Float64(_trapezoid_logevidence(
                Dscratch, wid, T(K), logZnodes, base_nodes, ltw,
                Int(win_off[t]), (t - 1) * ng, ng))
        end
    end
end

# Host: the grid's specs on their shared lattice — the lattice's squared loss
# widths and logZ̃ (NU nodes; logZ̃ depends on the width, the shape and the Z̃
# domain, not the prior),
# each grid point's window offset, and its prior base and trapezoid weights as
# column t of (NG × T) tables. Grid point t is `m`'s family and prior shape
# `alpha0`, centred at `prior_loss_width_grid[t]` (`_centred_marginal`, the
# keyword constructors' one form), with `m`'s Z̃ domain.
function _marg_quad_tables(m::ScaleMarginal{K}, prior_loss_width_grid, NG::Int) where {K}
    T = length(prior_loss_width_grid)
    margs = [_centred_marginal(K, m.alpha0, Float64(w), m.outlier_halfwidth)
             for w in prior_loss_width_grid]
    specs = [loss_spec(m; NG) for m in margs]
    i0 = [_marg_lattice_start(m.t0, NG) for m in margs]
    imin = minimum(i0); NU = maximum(i0) - imin + NG
    sq_loss_width = Vector{Float64}(undef, NU); logZ = similar(sq_loss_width)
    base = Matrix{Float64}(undef, NG, T); ltw = similar(base)
    for t in 1:T
        q = specs[t].quad; o = i0[t] - imin
        sq_loss_width[o .+ (1:NG)] .= q.sq_loss_width_nodes; logZ[o .+ (1:NG)] .= q.logZnodes
        base[:, t] .= q.base; ltw[:, t] .= q.ltw
    end
    # Every lattice node lies in some window when the windows overlap; a node
    # in a gap no window reads is given a harmless finite value.
    covered = falses(NU)
    for t in 1:T
        covered[(i0[t] - imin) .+ (1:NG)] .= true
    end
    for j in findall(!, covered)
        sq_loss_width[j] = exp((imin + j - 1) * _marg_lattice_spacing(NG)); logZ[j] = 0.0
    end
    spec = first(specs)
    return spec.shape, spec.Δ, sq_loss_width, logZ, base, ltw, Int32.(i0 .- imin), NU
end

# Shared chunked (scores → model-argmax → error-gather) driver loop.  The
# caller's `fill_scores!(scores_pmt, s, n)` launches whatever kernels fill the
# `(chunk, M_max, T)` Float64 score block for pairs `s+1:s+n` (and must
# synchronize); this owns the chunking, the shared `argmax_models_kernel!`,
# and the host copyback.  Function-typed `F` keeps the closure call static.
function _chunked_argmax_eval(fill_scores!::F, shared_ka, T::Int;
                              cs::Int, return_idxs::Bool) where {F}
    backend = shared_ka.backend
    P = length(shared_ka.K_per_pair); M_max = Int(shared_ka.M_max)
    scores_pmt = KernelAbstractions.allocate(backend, Float64, cs, M_max, T)
    errs_d     = KernelAbstractions.allocate(backend, Float64, cs, T)
    idxs_d     = return_idxs ? KernelAbstractions.allocate(backend, Int32, cs, T) :
                               KernelAbstractions.allocate(backend, Int32, 0)
    errs_h = Array{Float64}(undef, P, T)
    idxs_h = return_idxs ? Array{Int32}(undef, P, T) : Array{Int32}(undef, 0, 0)
    k_amax = _ka_kernel(argmax_models_kernel!, backend, 2)
    s = 0
    while s < P
        e = min(s + cs, P); n = e - s
        fill!(scores_pmt, -Inf)   # clear chunk slots: fill_scores! guards m ≤ M(p), so
                                  # slots beyond a pair's model count keep the PREVIOUS
                                  # chunk's scores and the argmax can select a phantom index
        fill_scores!(scores_pmt, s, n)
        k_amax(errs_d, idxs_d, scores_pmt, shared_ka.M_per_pair,
               shared_ka.errors_imc_flat, shared_ka.errors_offset, Int32(s);
               ndrange = (n, T))
        KernelAbstractions.synchronize(backend)
        # Contiguous full-buffer copyback: KA's copyto! from a STRIDED device
        # view (rows 1:n of cs, n < cs) copies the wrong elements on CUDA,
        # leaking the previous chunk's stale rows into this chunk's block
        # (phantom model indices > M(p) downstream).  Copy all cs rows
        # contiguously and slice on the host.
        errs_chunk = Array(errs_d)
        errs_h[s+1:e, :] .= @view errs_chunk[1:n, :]
        if return_idxs
            idxs_chunk = Array(idxs_d)
            idxs_h[s+1:e, :] .= @view idxs_chunk[1:n, :]
        end
        s = e
    end
    _free_local!(scores_pmt, errs_d, idxs_d)
    return return_idxs ? (errs_h, idxs_h) : errs_h
end

"""
    grid_evaluate([S = Float32,] m::ScaleMarginal, staged::IGSharedData, prior_loss_widths;
                  NG = _MARG_NG, return_idxs = false, mem_budget = 6_000_000_000)
        -> Matrix{Float64}

Full-data scale-marginal scoring (`WelschMarginal`, `SmoothTLSMarginal`) at
precision `S` (Float32 by default; Float64 for full precision), on the staged
data's backend — KernelAbstractions `CPU()` or a GPU. The scale is integrated
out by the package's one scale quadrature, the trapezoid logsumexp over `NG`
log-scale nodes spanning `t0 ± $(_MARG_L)` around each grid point's prior centre
(`loss_spec(::ScaleMarginal)`). Returns a `(P, T)` per-pair error matrix at
the model each grid-point's marginal selects.

`prior_loss_widths` is THIS FAMILY'S tuned parameter — the IG-prior centre on
the loss width (`prior_loss_width`, `NormalizedImageFrame` units), not a
decision threshold. Grid point `t` is `m`'s family with `m`'s prior shape
`alpha0`, Z̃ domain `m.outlier_halfwidth` (the declared outlier support,
`POOL_FRAME_R_MAX` on pool data), centred at
`prior_loss_widths[t]`: the grid REPLACES `m`'s own centre. `return_idxs=true`
also returns the selected pool indices. The pair-chunk is sized so the
NG-per-work-item scratch and the score block stay under `mem_budget` bytes.
"""
function grid_evaluate(::Type{S}, m::ScaleMarginal, staged::IGSharedData,
                       prior_loss_widths::AbstractVector{<:Real};
                       NG::Int = _MARG_NG, return_idxs::Bool = false,
                       mem_budget::Integer = 6_000_000_000) where {S<:AbstractFloat}
    backend = staged.backend
    P = length(staged.K_per_pair); T = length(prior_loss_widths)
    M_max = Int(staged.M_max)
    shape, Δ, sq_loss_width, logZ, base, ltw, win_off, NU =
        _marg_quad_tables(m, prior_loss_widths, NG)
    sq_loss_width_d = _ka_upload(backend, S, sq_loss_width)
    logZ_d = _ka_upload(backend, S, logZ)
    base_d = _ka_upload(backend, S, vec(base))
    ltw_d  = _ka_upload(backend, S, vec(ltw))
    off_d  = _ka_upload(backend, Int32, win_off)

    per_pair = M_max * NU * sizeof(S) + M_max * T * 8     # Dscratch + scores
    cs = clamp(Int(fld(mem_budget, max(per_pair, 1))), 1, min(256, P))
    Dscratch = KernelAbstractions.allocate(backend, S, NU, cs * M_max)
    k_ev = _ka_kernel(marg_evidence_kernel!, backend, 2)
    out = _chunked_argmax_eval(staged, T; cs, return_idxs) do scores_pmt, s, n
        k_ev(scores_pmt, staged.sorted_sq_flat, staged.pair_offset,
             staged.K_per_pair, staged.M_per_pair,
             sq_loss_width_d, logZ_d, base_d, ltw_d, off_d, Dscratch,
             S(Δ), Int32(NU), Int32(NG), Int32(T), Int32(s), shape; ndrange = (n, M_max))
        KernelAbstractions.synchronize(backend)
    end
    _free_local!(Dscratch, sq_loss_width_d, logZ_d, base_d, ltw_d, off_d)
    return out
end

# (The multi-GPU drivers `grid_evaluate_marg_ka_mgpu`/`delta_sweep_mix_ka_mgpu`
#  and their `_mgpu_staged_vcat`/`_marg_cuda_device!`/`_marg_ndev` helpers are
#  deleted: no caller in RSDR, RVG, VisualGeometryToolkitEval or LODR, and the device helpers
#  reached into `Main.CUDA` — CUDA code in the core package, behind a comment
#  claiming an extension that does not exist. A multi-GPU driver belongs in the
#  experiment that needs it, where `using CUDA` is real.)

# =============================================================================
# Track B (Stage 6) — the MIXTURE marginal: its one evaluation.
#
# Unlike the full-data kernel above (every point an inlier, K·logZ̃), this is the
# sort-and-sweep MIXTURE: the loss width is integrated out by the same
# trapezoid logsumexp over NG fixed log-scale nodes, with a tunable
# WELL DEPTH Δ and Z̃ normalised over the FIXED [0, r_max] domain (so the outlier
# half-width is DECOUPLED — profiled by the O(n) re-argmax over k,
# `max_k[E_in(k) − (K−k)·dg·log(2·outlier_halfwidth)]`).  Device-agnostic KA `@kernel`: identical
# code on CPU() and CUDABackend(); the gate (test/gate_marg_mix_ka.jl) checks it
# bit for bit against a plain host loop.
#
# Per work-item (pair, model): an NG-length data accumulator D_k(w²_j) is swept
# incrementally over the K sorted residuals (minimal-sample points x=0 at the
# front).  That NG buffer is per-work-item global scratch (`Dscratch`, column
# `wid`) — NG can be ~1500 for smooth-TLS, so Dscratch = NG × (chunk·M_max);
# size the pair-chunk accordingly on GPU (see `grid_evaluate(::LossSpec, …)`).
# =============================================================================

# Per (pair p_local, model mm): fill E_out[p_local, mm, k] with the inlier
# log-evidence of the top-k residuals (k = 1…K; −Inf for k ≤ model_dof and k>K),
# via the two-scan trapezoid logsumexp over the nodes (i = 1..k, node j inner).
@kernel function marg_mix_evidence_kernel!(
        E_out,                       # (chunk, M_max, Kmax) T
        @Const(sorted_sq_flat),      # F (Float64 bit-exact ref / Float32 fast) — sorted asc/model
        @Const(pair_offset), @Const(K_per_pair), @Const(M_per_pair),
        @Const(sq_loss_width_nodes), @Const(logZnodes), @Const(base_nodes), @Const(ltw),   # (NG,) T
        Dscratch,                    # (NG, W) T — per-work-item accumulator
        Δ::T, model_dof::Int32, NG::Int32, Kmax::Int32, p_offset::Int32,
        shape) where {T}
    p_local, mm = @index(Global, NTuple)
    wid = @index(Global, Linear)
    p = Int(p_local) + Int(p_offset)
    # ndrange padding guard: GPU backends round ndrange up to workgroup
    # multiples; padded threads must not read M_per_pair[p] (OOB → garbage M)
    # or write E_out/Dscratch rows beyond the chunk (corrupts neighbors).
    inbounds_thread = p_local <= size(E_out, 1) && p <= length(M_per_pair)
    M = inbounds_thread ? Int(M_per_pair[p]) : 0
    if mm <= M
        ng = Int(NG); Km = Int(Kmax); md = Int(model_dof)
        K = Int(K_per_pair[p])
        boff = pair_offset[p] + (Int64(mm) - Int64(1)) * Int64(K)
        @inbounds for j in 1:ng
            Dscratch[j, wid] = zero(T)
        end
        @inbounds for k in 1:K
            _accumulate_data!(Dscratch, wid, T(sorted_sq_flat[boff + k]), Δ, shape,
                              sq_loss_width_nodes, ng)
            if k <= md
                E_out[p_local, mm, k] = T(-Inf)
            else
                E_out[p_local, mm, k] = _trapezoid_logevidence(Dscratch, wid, T(k - md),
                    logZnodes, base_nodes, ltw, 0, 0, ng)
            end
        end
        @inbounds for k in (K + 1):Km
            E_out[p_local, mm, k] = T(-Inf)
        end
    end
end

# (`marg_mix_evidence_ka` — a one-chunk host launcher returning the raw evidence
#  tensor — and its Symbol-keyed family tag are deleted: nothing called it; the
#  gate test imported it without use. `grid_evaluate(::LossSpec, …)` below is the
#  launcher.)

# Per (pair p_local, model mm, grid-point t): profile the outlier half-width —
#   S = max_{k > model_dof} [ E[p_local,mm,k] − (K−k)·dg·log(2·outlier_halfwidth_t) ]
# (an O(n) re-argmax; E entries
# at k ≤ model_dof and k > K are −Inf by construction, so the plain max is safe).
# Scores accumulate in Float64 (argmax-safe), matching the other sweep kernels.
@kernel function marg_mix_profile_kernel!(
        scores_out,                  # (chunk, M_max, T) Float64
        @Const(E_in),                # (chunk, M_max, Kmax) T
        @Const(K_per_pair), @Const(M_per_pair),
        @Const(log_two_outlier_halfwidth_grid),   # (T,) Float64 — log(2·outlier_halfwidth_t)
        dg::Int32, p_offset::Int32)
    p_local, mm, t = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)
    inbounds_thread = p_local <= size(scores_out, 1) && t <= size(scores_out, 3) &&
                      p <= length(M_per_pair)
    M = inbounds_thread ? Int(M_per_pair[p]) : 0
    if mm <= M
        K = Int(K_per_pair[p])
        outlier_cost = Float64(dg) * log_two_outlier_halfwidth_grid[t]
        best = -Inf
        @inbounds for k in 1:K
            S = Float64(E_in[p_local, mm, k]) - (K - k) * outlier_cost
            S > best && (best = S)
        end
        @inbounds scores_out[p_local, mm, t] = best
    end
end

"""
    grid_evaluate([S = Float32,] spec::LossSpec, staged::IGSharedData, outlier_halfwidths;
                  model_dof, dg, return_idxs = false, mem_budget = 6_000_000_000)
        -> Matrix{Float64}

Track-B mixture-marginal grid evaluation, on the staged data's backend
(KernelAbstractions `CPU()` or a GPU), the ONLY evaluation of this evidence.
For the depth-`Δ` quadrature
selector `spec` (built by `loss_spec` — Z̃ over the FIXED `[0, r_max]` domain),
score every staged (pair, model) at every outlier half-width in
`outlier_halfwidths` (`NormalizedImageFrame` units) and
return the `(P, T)` per-pair error matrix at each grid-point's selected model
(`return_idxs=true` also returns the selected pool indices).  Three kernels per
pair-chunk: `marg_mix_evidence_kernel!` (E over k; the NG-node trapezoid
quadrature), `marg_mix_profile_kernel!` (the outlier-half-width profile
re-argmax over k), and
the shared `argmax_models_kernel!`.  Precision `S` drives the evidence
arithmetic — Float64 for full precision; Float32 is the fast production
setting.  The
pair-chunk is sized so the E block + the NG-per-work-item `Dscratch` stay under
`mem_budget` bytes. `model_dof` (the minimal sample size, whose points are the
exact inliers at the front of each column) and `dg` (the codimension) are the
problem's, and REQUIRED: a `LossSpec` carries neither.
"""
function grid_evaluate(::Type{S}, spec::LossSpec, shared_ka::IGSharedData,
                       outlier_halfwidth_grid::AbstractVector{<:Real};
                       model_dof::Int, dg::Int,
                       return_idxs::Bool = false,
                                   mem_budget::Integer = 6_000_000_000) where {S<:AbstractFloat}
    backend = shared_ka.backend
    P = length(shared_ka.K_per_pair); T = length(outlier_halfwidth_grid)
    M_max = Int(shared_ka.M_max)
    quad = spec.quad; NG = length(quad.sq_loss_width_nodes)

    # Host copies of the per-pair sizes (small) — Kmax sizes the E block.
    K_h = Array(shared_ka.K_per_pair)
    Kmax = Int(maximum(K_h))

    # Chunk sizing: per pair-slot bytes = M_max·(Kmax + NG)·sizeof(S)  (E + Dscratch)
    # + M_max·T·8 (scores).  The argmax outputs are negligible.
    per_pair = M_max * (Kmax + NG) * sizeof(S) + M_max * T * 8
    cs = clamp(Int(fld(mem_budget, max(per_pair, 1))), 1, min(256, P))

    # Device quadrature tables (depth Δ baked into quad.logZnodes by loss_spec).
    sq_loss_width_nodes_d = _ka_upload(backend, S, quad.sq_loss_width_nodes)
    logZ_d   = _ka_upload(backend, S, quad.logZnodes)
    base_d   = _ka_upload(backend, S, quad.base)
    ltw_d    = _ka_upload(backend, S, quad.ltw)
    log_two_outlier_halfwidth_d = _ka_upload(backend, Float64,
                                             [log(2 * Float64(h)) for h in outlier_halfwidth_grid])

    E_d      = KernelAbstractions.allocate(backend, S, cs, M_max, Kmax)
    Dscratch = KernelAbstractions.allocate(backend, S, NG, cs * M_max)

    k_ev   = _ka_kernel(marg_mix_evidence_kernel!, backend, 2)
    k_prof = _ka_kernel(marg_mix_profile_kernel!, backend, 3)
    out = _chunked_argmax_eval(shared_ka, T; cs, return_idxs) do scores_pmt, s, n
        fill!(E_d, S(-Inf))
        k_ev(E_d, shared_ka.sorted_sq_flat, shared_ka.pair_offset,
             shared_ka.K_per_pair, shared_ka.M_per_pair,
             sq_loss_width_nodes_d, logZ_d, base_d, ltw_d, Dscratch,
             S(spec.Δ), Int32(model_dof), Int32(NG), Int32(Kmax), Int32(s), spec.shape;
             ndrange = (n, M_max))
        KernelAbstractions.synchronize(backend)
        k_prof(scores_pmt, E_d, shared_ka.K_per_pair, shared_ka.M_per_pair,
               log_two_outlier_halfwidth_d, Int32(dg), Int32(s); ndrange = (n, M_max, T))
        KernelAbstractions.synchronize(backend)
    end
    _free_local!(E_d, Dscratch, sq_loss_width_nodes_d, logZ_d, base_d, ltw_d, log_two_outlier_halfwidth_d)
    return out
end

# Default precision Float32 for every quadrature marginal and the loss widths;
# callers that need the bit-exact reference supply Float64 explicitly.
grid_evaluate(spec::Union{LossSpec, ScaleMarginal}, staged::IGSharedData,
              widths::AbstractVector{<:Real}; kwargs...) =
    grid_evaluate(Float32, spec, staged, widths; kwargs...)

# (The Δ-sweep driver `delta_sweep_mix_ka` — stage once, loop `loss_spec` +
#  `grid_evaluate(::LossSpec, …)` over well depths — and its global scale-bounds
#  helper are deleted with the multi-GPU driver that was
#  their only caller. The loop is four lines over the two public calls above.)
