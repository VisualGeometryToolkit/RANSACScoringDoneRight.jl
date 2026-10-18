# =============================================================================
# Sort-and-Sweep — Scale-free marginal scoring (Algorithm 1)
# =============================================================================
#
# Standalone sweep infrastructure used by both StudentTUniformScoring (RANSAC)
# and the standalone sweep!(sws, ...) method (generic posterior IRLS).
#
# =============================================================================

# The scale prior `IGPrior` (and `JeffreysPrior()`) is VisualGeometryToolkitCore's.

# =============================================================================
# Loggamma Table
# =============================================================================

"""
    _build_lg_table(n, d_g=1; alpha0=0.0) -> Vector{Float64}

Precompute `log Γ(k·dg/2 + α₀)` for `k = 1..n` via SpecialFunctions.loggamma —
single uniform path for Jeffreys (α₀=0) and Inverse-Gamma (α₀>0) priors.

Cost: ~5000 host-side `loggamma` calls per probe (~0.5 ms total) — table is
read-only on the device for the entire sweep.  An earlier version used a
log-recurrence specialised on Jeffreys; we standardised on `loggamma` so
there's a single code path with uniform Float64-precision values, no
recurrence drift, and no separate code branch to maintain.
"""
function _build_lg_table(n::Int, d_g::Int=1; alpha0::Float64=0.0)
    lg = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        lg[k] = loggamma(k * d_g / 2 + alpha0)
    end
    return lg
end

# =============================================================================
# Radix perm-sort — allocation-free stable sortperm for the sweep fast path
# =============================================================================
#
# The single-sort fast path of `sweep!` spends nearly all of its time in
# `sortperm!` over the per-point scores (~72 of ~87 µs per scored candidate at
# n = 2500 in the RVG e2e RANSAC loop — the dominant cost of the whole loop).
# `Base.sortperm!` documents a STABILITY GUARANTEE ("the permutation is
# guaranteed to be stable ... indices of equal elements appear in ascending
# order"), so ANY stable ascending-by-`isless` permutation sort produces the
# identical `perm` — ties included. LSD radix sort over a monotone unsigned key
# is such a sort, runs in O(n) with small constants, and needs no comparisons.
#
# WHY A HAND-ROLLED RADIX AND NOT A LIBRARY (measured 2026-07-26, and please
# do not re-litigate without re-measuring). Sorting 715-residual columns over
# 4000 hypotheses of the PT-E british_museum pool, µs per hypothesis:
#
#     Base sortperm!(F64, reusable scratch)            19.70
#     Base sortperm!(F32 input)                        19.69   ← no F32 benefit
#     Base sort! on packed (key<<32 | index) UInt64    17.39
#     SortingAlgorithms.jl RadixSort, same packing     17.31
#     SortingAlgorithms.jl RadixSort via sortperm!     24.18
#     this implementation, UInt64 keys                 10.12
#     this implementation, UInt32 keys (default)        6.15
#
# All variants emit identical permutations; the gap is structural, not
# algorithmic cleverness: this version OWNS its scratch (the scoring loop
# calls it ~3·10⁵ times per scene — per-call allocation dominates the
# library paths), fuses every digit histogram into ONE scan over the keys,
# and skips digits on which all keys agree, which is most digits when
# residuals span a narrow range. A library sort cannot do the first or the
# third through its public interface. If a future Base/SortingAlgorithms
# release closes that gap, delete this and use it — after re-running
# neurips2026/f32_radix_probe.jl.
#
# Key transform (`_radix_key`): the standard IEEE-754 total-order flip
# (positive: set sign bit; negative: bitwise NOT), with every NaN collapsed to
# the maximum key. This reproduces `isless` exactly: -Inf < ... < -0.0 < +0.0
# < ... < +Inf < NaN, all NaNs mutually equal (payload/sign ignored — the
# collapse plus LSD stability preserves their original index order, exactly
# like a stable comparison sort under `isless`, which treats NaNs as
# incomparable-equal).
# =============================================================================

"""
    RadixPermBuffers{K}(n)
    RadixPermBuffers(n)          # K = UInt32 (Float32 key precision) — THE default

Scratch buffers for [`_radix_sortperm!`](@ref) — the allocation-free stable
LSD radix `sortperm` used by the `sweep!` single-sort fast path. Owned by
[`ScoringWorkspace`](@ref) (`sws.radix`); sized to the data count and trimmed
alongside the other sweep buffers (RVG `set_n!`).

### Key width `K` — the sort-precision knob (2026-07-26)

The radix does `sizeof(K)` byte iterations, each scattering a `K` key plus an
`Int` perm entry, so the key width sets both the iteration count and the scatter
bandwidth. `K = UInt32` (sort keys at Float32 precision) is the **default**:
measured 1.65× faster than `UInt64` on real PhotoTourism residual columns
(10.12 → 6.15 µs/hypothesis; ≈1.16× on the whole marginal per-pair scorer,
then `score_marginal`), and it
loses nothing, because the scored residuals are Float32 to begin with —
a pool's residual matrix is a `Matrix{Float32}`, so the Float64 sweep
buffer holds an exactly-representable widening and the round trip back to
Float32 is lossless. Verified on that pool: the emitted permutation is
BYTE-IDENTICAL to the `UInt64` path in 150/150 hypotheses (worst residual
delta exactly 0.0; `neurips2026/f32_radix_probe.jl`).

`K = UInt64` remains available and is the right choice for a caller whose
residuals carry genuine Float64 information (they are then ordered at full
precision). Both widths run the SAME generic code below — the width is a
type parameter, not a second implementation.

Ordering semantics at `K = UInt32`: the key is monotone but not injective on
Float64 (values within one Float32 ulp collapse to equal keys). Stability
then orders such a tie group by index, exactly as a stable comparison sort
under an `isless` that cannot separate them. Every sweep prefix boundary
inside such a group is between residuals equal to Float32 precision.
"""
struct RadixPermBuffers{K<:Unsigned}
    keys::Vector{K}          # monotone-transformed sort keys
    keys2::Vector{K}         # ping-pong partner of `keys`
    perm2::Vector{Int}       # ping-pong partner of the caller's perm
    counts::Matrix{Int}      # 256 × sizeof(K) per-digit histograms (one fused scan)
end
RadixPermBuffers{K}(n::Int) where {K<:Unsigned} = RadixPermBuffers{K}(
    Vector{K}(undef, n), Vector{K}(undef, n),
    Vector{Int}(undef, n), Matrix{Int}(undef, 256, sizeof(K)))
RadixPermBuffers(n::Int) = RadixPermBuffers{UInt32}(n)

"""
    _radix_key(x::Float64, ::Type{K}) -> K

Monotone key: `_radix_key(a, K) < _radix_key(b, K)` ⟹ `isless(a, b)`.
For `K = UInt64` the converse holds too — `_radix_key(a) == _radix_key(b)` ⟺
neither `isless(a, b)` nor `isless(b, a)` (every NaN maps to the single
maximum key). Total-order equivalent of `isless` on all Float64 values,
including ±0.0 (distinct, -0.0 < +0.0 — exactly `isless`'s ordering) and
±Inf.

For `K = UInt32` the value is first rounded to Float32 (the precision the
scored residuals actually carry); the map stays monotone non-decreasing, so
the emitted order is a valid ascending order, with ties broken by index
through LSD stability.
"""
@inline function _radix_key(x::Float64, ::Type{UInt64})
    isnan(x) && return 0xffffffffffffffff
    u = reinterpret(UInt64, x)
    return (u & 0x8000000000000000) == zero(UInt64) ? (u | 0x8000000000000000) : ~u
end
@inline function _radix_key(x::Float64, ::Type{UInt32})
    f = Float32(x)
    isnan(f) && return 0xffffffff
    u = reinterpret(UInt32, f)
    return (u & 0x80000000) == zero(UInt32) ? (u | 0x80000000) : ~u
end
# One-argument form at the UInt64 (full-Float64) key width — the form
# radix_sortperm_tests.jl's order-equivalence check uses. Not back-compat: it is
# live, and the only caller that wants a key without naming a width.
@inline _radix_key(x::Float64) = _radix_key(x, UInt64)

# Float32 targets (Float32-storage RansacWorkspace sorts its scores buffer
# directly): UInt32 keys are the identity precision — no rounding, exact
# `isless` order; UInt64 keys via lossless widening.
@inline function _radix_key(x::Float32, ::Type{UInt32})
    isnan(x) && return 0xffffffff
    u = reinterpret(UInt32, x)
    return (u & 0x80000000) == zero(UInt32) ? (u | 0x80000000) : ~u
end
@inline _radix_key(x::Float32, ::Type{UInt64}) = _radix_key(Float64(x), UInt64)

"""
    _radix_sortperm!(perm, target, rb::RadixPermBuffers, n) -> nothing

Fill `perm[1:n]` with the stable ascending-`isless` permutation of
`target[1:n]` — bitwise identical to `sortperm!(perm, target)` for every
input (see the stability argument in the header comment). O(n): one fused
histogram scan, then ≤ 8 stable counting iterations over byte digits, skipping
digits on which all keys agree (a stable no-op). Allocation-free; buffers
may have capacity ≥ n.
"""
function _radix_sortperm!(perm::Vector{Int}, target::AbstractVector{<:Union{Float32,Float64}},
                          rb::RadixPermBuffers{K}, n::Int) where {K<:Unsigned}
    keys = rb.keys; keys2 = rb.keys2; perm2 = rb.perm2; counts = rb.counts
    ndig = sizeof(K)                       # byte iterations: 4 for UInt32, 8 for UInt64
    @inbounds for i in 1:n
        keys[i] = _radix_key(target[i], K)
        perm[i] = i
    end
    fill!(counts, 0)
    # Fused histogram: one scan over the keys accumulating every byte digit.
    # `ndig` is a compile-time constant per K, so the loop fully unrolls.
    @inbounds for i in 1:n
        k = keys[i]
        for d in 1:ndig
            counts[Int((k >> (8 * (d - 1))) & 0xff) + 1, d] += 1
        end
    end
    src_k, dst_k = keys, keys2
    src_p, dst_p = perm, perm2
    @inbounds for d in 1:ndig
        # Trivial digit (all n keys in one bucket): the stable iteration would be
        # an identity copy — skip it.
        b1 = 1
        while counts[b1, d] == 0
            b1 += 1
        end
        counts[b1, d] == n && continue
        # Prefix sums: counts[b,d] becomes the 0-based start offset of bucket b.
        off = 0
        for b in 1:256
            c = counts[b, d]
            counts[b, d] = off
            off += c
        end
        shift = 8 * (d - 1)
        for i in 1:n
            k = src_k[i]
            b = Int((k >> shift) & 0xff) + 1
            j = counts[b, d] + 1
            counts[b, d] = j
            dst_k[j] = k
            dst_p[j] = src_p[i]
        end
        src_k, dst_k = dst_k, src_k
        src_p, dst_p = dst_p, src_p
    end
    if src_p !== perm
        copyto!(perm, 1, src_p, 1, n)
    end
    return nothing
end

# =============================================================================
# ScoringWorkspace — Pre-allocated buffers for Algorithm 1 sweep
# =============================================================================

"""
    ScoringWorkspace

Pre-allocated scratch buffers for scale-free marginal scoring (Algorithm 1).

Separated from [`StudentTUniformScoring`](@ref) so that scoring carries only the
immutable spec (model DOF, outlier half-width, codimension) while the mutable
workspace lives on `RansacWorkspace`. Mirrors the spec/workspace
split already in place for local-optimization via `lo_ws`.

### Prior consistency

The `lg_table` is built at allocation time with a specific `α₀` — once a
`ScoringWorkspace` is constructed it is **committed to that scale prior**.
The `scale_prior` field is the single source of truth for which prior the
buffers are configured for; `sweep!(sws, ...)` reads it from here rather
than accepting a runtime kwarg, which would risk pairing a Jeffreys
lg_table with an IG score formula.

# Fields
- `perm::Vector{Int}` — sortperm buffer, mutated by every `sweep!`
- `scratch::Vector{Int}` — sortperm scratch (comparison-sort fallback paths)
- `radix::RadixPermBuffers` — scratch for the allocation-free stable radix
  `sortperm` taken by the exact-size single-sort fast path (bitwise-identical
  permutation; see [`_radix_sortperm!`](@ref))
- `lg_table::Vector{Float64}` — log Γ(k·dg/2 + α₀); depends on n, dg, and the scale prior
- `penalties::Vector{Float64}` — per-point ℓᵢ = log|Ṽᵢ| (predictive scoring)
- `sort_key::Vector{Float64}` — the sweep ordering key: the per-point unit-scale
  Mahalanobis `q` in Predictive scoring (raw r² only in the model-certain
  fallback). The historical raw-r² key is retired — with heterogeneous penalties
  `sweep!` follows the generalized sweep lemma's parametric ordering.
- `prior_table::Vector{Float64}` — profiled-γ correction; empty = flat prior
- `scale_prior::IGPrior` — the prior the lg_table was built for (α₀=β₀=0 ⇒ Jeffreys)

Build with `alloc_scoring_workspace(problem, scoring)` (defined in
`estimators/ransac/scoring.jl`) or `default_scoring_workspace(n; dg, scale_prior)`.
"""
mutable struct ScoringWorkspace{K<:Unsigned}
    const perm::Vector{Int}
    const scratch::Vector{Int}
    const lg_table::Vector{Float64}
    const penalties::Vector{Float64}
    const sort_key::Vector{Float64}
    const prior_table::Vector{Float64}
    const scale_prior::IGPrior
    const radix::RadixPermBuffers{K}   # K = sort-key width; see RadixPermBuffers
    k::Int                             # k* of the last `score!` (the scoring contract's `inlier_count`)
end

# Seven-argument form: derive the radix scratch from the perm buffer's length,
# at the default key width. NOT a back-compat shim, which is what this comment
# used to call it — it is the form BOTH of this package's workspace builders
# call (`default_scoring_workspace` here, `alloc_scoring_workspace` in
# scoring.jl). The eight-argument inner constructor is the one nothing calls
# directly.
ScoringWorkspace(perm::Vector{Int}, scratch::Vector{Int}, lg_table::Vector{Float64},
                 penalties::Vector{Float64}, sort_key::Vector{Float64},
                 prior_table::Vector{Float64}, scale_prior::IGPrior) =
    ScoringWorkspace(perm, scratch, lg_table, penalties, sort_key, prior_table,
                     scale_prior, RadixPermBuffers(length(perm)), 0)

# -----------------------------------------------------------------------------
# Accessors — downstream reaches the workspace through these, never by field
# name (VisualGeometryToolkitEstimation's RANSAC glue read seven of its fields directly).
# Each returns the buffer itself, so the glue's elementwise writes into
# `penalties`/`sort_key` go through them unchanged.
# -----------------------------------------------------------------------------

"""
    sweep_order(sws::ScoringWorkspace) -> Vector{Int}

The order the last `sweep!` visited the points in: `sweep_order(sws)[1:k]` is
the support of the partition it returned `k` for (the scoring contract's
`consensus_order`).
"""
sweep_order(sws::ScoringWorkspace) = sws.perm

"""
    penalties(sws::ScoringWorkspace) -> Vector{Float64}

Per-point log-determinant penalties `ℓᵢ = log|Vᵢ|` of predictive scoring
(`StudentTUniformScoring{Predictive}`), filled by the scoring glue before a sweep.
"""
penalties(sws::ScoringWorkspace) = sws.penalties

"""
    sort_key(sws::ScoringWorkspace) -> Vector{Float64}

The per-point sweep ordering key of predictive scoring (see
[`ScoringWorkspace`](@ref)), filled by the scoring glue before a sweep.
"""
sort_key(sws::ScoringWorkspace) = sws.sort_key

"""
    lg_table(sws::ScoringWorkspace) -> Vector{Float64}

The `log Γ(k·dg/2 + α₀)` table the workspace was built with; its length is the
largest `n` the workspace can sweep (see `resize!`).
"""
lg_table(sws::ScoringWorkspace) = sws.lg_table

"""
    scale_prior(sws::ScoringWorkspace) -> IGPrior

The scale prior the workspace is committed to (that `sweep!(sws, …)` scores
under); a `StudentTUniformScoring` carries its own (the scoring contract's
`scale_prior`).
"""
scale_prior(sws::ScoringWorkspace) = sws.scale_prior

"""
    resize!(sws::ScoringWorkspace, n) -> sws

Size every per-point buffer of `sws` for a problem of `n` points. They are the sort
permutation and its scratch, the predictive `penalties` (zeroed, as a fresh
workspace has them) and `sort_key`, and the radix perm-sort scratch — keeping
the exact-size radix fast path of `sweep!` reachable. A profiled-γ
`prior_table` is REBUILT for `n`, since `k·log(k/N) + (N−k)·log((N−k)/N)`
depends on `N`: truncating a table built for a larger `N` would keep that
`N`'s values. At `n = 0` the table is left as it is, because its
non-emptiness is the workspace's only record that the scoring is profiled.

`lg_table` is not resized: its entries depend on `k` alone, so a table of
capacity ≥ `n` serves any `n`. It cannot grow — the workspace does not record
the codimension it was built for — so `n > length(lg_table(sws))` throws;
build a larger workspace instead. (Growing the other buffers past it used to
leave `sweep!` reading beyond the table under `@inbounds`.)

Replaces VisualGeometryToolkitEstimation's `_set_n!(::ScoringWorkspace, n)`, which reached
into seven of these fields by name and truncated the `prior_table`.
"""
function Base.resize!(sws::ScoringWorkspace, n::Integer)
    n >= 0 || throw(ArgumentError("resize!: n must be non-negative, got $n"))
    n <= length(sws.lg_table) || throw(ArgumentError(
        "resize!: this ScoringWorkspace's lg_table covers $(length(sws.lg_table)) " *
        "points, not $n; build a workspace for the larger problem"))
    resize!(sws.perm, n)
    resize!(sws.scratch, n)
    fill!(resize!(sws.penalties, n), 0.0)
    resize!(sws.sort_key, n)
    if n > 0 && !isempty(sws.prior_table)
        copyto!(resize!(sws.prior_table, n), _build_prior_table(n))
    end
    resize!(sws.radix.keys, n)
    resize!(sws.radix.keys2, n)
    resize!(sws.radix.perm2, n)
    return sws
end

"""
    default_scoring_workspace(n; dg=1, scale_prior=JeffreysPrior()) -> ScoringWorkspace

Vanilla scoring workspace, flat γ prior. Length-`n` buffers for a problem
of `n` points with codimension `dg`. The `scale_prior` kwarg determines the
lg_table's `α₀`; the returned sws carries that prior and cannot be
repurposed for a different one.

Used by standalone callers (BlobBoards blob adaptation, the CUDA extension's
host-side seed loop, raw low-level `RansacWorkspace` construction in tests)
where no `StudentTUniformScoring` is in hand.
"""
function default_scoring_workspace(n::Int; dg::Int=1,
                                   scale_prior::IGPrior=IGPrior())
    ScoringWorkspace(
        Vector{Int}(undef, n),
        Vector{Int}(undef, n),
        _build_lg_table(n, dg; alpha0=_alpha0(scale_prior)),
        zeros(Float64, n),
        Vector{Float64}(undef, n),
        Float64[],
        scale_prior,
    )
end

# =============================================================================
# Per-partition score (single formula; subsumes Jeffreys at α₀=β₀=0)
# =============================================================================

@inline function _sweep_score(p::IGPrior, lg_k, ν_half, RSS::T, outlier_term) where T
    lg_k - ν_half * log(2 * T(π)) - (ν_half + T(p.alpha0)) * log(RSS / 2 + T(p.beta0)) - outlier_term
end

# The score S(k) of the top-k partition of n points, from its running RSS: the
# ONE statement of the per-partition arithmetic every sweep shares — the host
# `sweep!`, the parametric sweep's `_prefix_score` and the device grid kernel
# (`ig_sweep_kernel!`). The RSS floor clamps a COPY, never the running sum;
# ν = dg·(k − m) residual dof, and the n − k outliers pay (2·outlier_halfwidth)^{−dg} each.
# Plain Float64 arithmetic, callable inside a KernelAbstractions kernel.
@inline function _partition_score(prior::IGPrior, lg_table, RSS::Float64, k::Integer,
                                  model_dof::Integer, dg_f::Float64, n::Integer,
                                  log_two_outlier_halfwidth::Float64)
    return _sweep_score(prior, lg_table[k - model_dof], dg_f * (k - model_dof) / 2,
                        max(RSS, eps(Float64)), (n - k) * dg_f * log_two_outlier_halfwidth)
end

# =============================================================================
# Sort-and-Sweep
# =============================================================================

"""
    sweep!(perm, lg_table, model_dof, log_two_outlier_halfwidth, dg, scores, n; ...) -> (S*, k*)

Low-level sort-and-sweep kernel for the scale-free marginal score (Algorithm 1).

### The per-hypothesis path — deliberately not the KernelAbstractions kernel

The batched marginal GRID is one KernelAbstractions kernel (`ig_sweep_kernel!`
under `grid_evaluate(::StudentTUniformScoring, …)`, CPU and GPU alike). This function is kept beside
it on purpose, as the ONE per-hypothesis path, and is not a twin of it:

- It is latency-bound. A RANSAC loop scores one hypothesis at a time, and
  routing one through the kernel on `KernelAbstractions.CPU()` (squares sorted
  into a preallocated flat buffer, then a 1×1×1 launch) measured slower
  (2026-09-21, BenchmarkTools, Float64): n = 200: 1.88 → 2.57 µs (1.37×);
  n = 715: 6.16 → 10.61 µs (1.72×); n = 2500: 24.5 → 31.7 µs (1.29×).
- It does more. The kernel scores pre-sorted, penalty-free Jeffreys/IG columns;
  this also sorts (the radix below), carries the predictive per-point
  penalties, the profiled-γ `prior_table`, and the exact parametric sweep
  for heterogeneous gains.

Sorts points (ascending by `scores`, or `sort_key` if provided), then sweeps
prefixes k = m+1,...,n evaluating the partition score via
`_sweep_score(scale_prior, ...)`. Returns the best score S* and optimal
inlier count k*. Mutates `perm`.

### Complexity — depends on which sort path is taken (see Sizing)

- **Θ(n)** on the exact-size fast path: [`_radix_sortperm!`](@ref) is an LSD
  radix over monotone unsigned keys — one fused histogram scan plus at most
  `sizeof(K)` stable counting iterations (4 for the default `UInt32`), with trivial
  digits skipped. No comparisons, so no `log` factor. The sweep itself is a
  single Θ(n) scan, so the whole kernel is linear.
- **Θ(n log n)** on the partial-buffer path, which falls back to the
  comparison-based `Base.sortperm!`.

This is not a constant-factor distinction: an oversized workspace silently
changes the complexity class. Measured on PT-E-SIFT-4K (2026-07-26), the same
scorer cost 51.5 µs/hypothesis through the partial path and 21.7 through the
radix path, because a caller had sized one workspace to the pool-wide maximum
instead of per-problem. `resize!(sws, n)` resizes every sortable buffer to
`n` for exactly this reason.

### Sizing

Two invariants are handled:

- **Exact-size fast path** (`length(perm) == n` and `length(sort_target) == n`):
  sorts via [`_radix_sortperm!`](@ref) when radix buffers of capacity ≥ n are
  present (the emitted permutation is bitwise identical to `sortperm!`), else
  via `sortperm!` directly. Hot path for RANSAC scoring, where `ws.scores`,
  `sws.perm`, `sws.scratch` are all sized to `data_size(problem)`.
- **Partial-buffer path** (`length(perm) > n` or `length(sort_target) > n`):
  sorts views of the first-n prefix through `sortperm!`. The radix is NOT
  reachable here — it requires exact-size concrete `Vector`s, and a `@view` is
  neither. Used by BlobBoards blob adaptation (one `sws` sized to `max_n`,
  per-blob `n` varies) and the CUDA-ext host-side seed loop.

`scratch` (if provided) is forwarded to `sortperm!` only when `length(scratch) == n`,
because `Base.sortperm!` rejects `SubArray` scratch.
"""
function sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                model_dof::Int, log_two_outlier_halfwidth::Float64, dg::Int,
                scores::AbstractVector{T}, n::Int;
                scratch::Union{Nothing,Vector{Int}}=nothing,
                penalties::Union{Nothing,AbstractVector{Float64}}=nothing,
                sort_key::Union{Nothing,AbstractVector{Float64}}=nothing,
                prior_table::Union{Nothing,AbstractVector{Float64}}=nothing,
                scale_prior::IGPrior=IGPrior(),
                radix::Union{Nothing,RadixPermBuffers}=nothing) where T
    # LEMMA-CONFORMANT PATH SELECTION (uncertain_evidence 35ae25c, Lemma
    # "Generalized sweep"): when per-point gains g_i = -penalties_i/2 VARY,
    # the optimal support is a top-k set of t_i(mu) = g_i - mu*q_i at the
    # support's own profiled mu_I = nu_k/(2 Q_I) — enumerated EXACTLY by the
    # parametric sweep below. When all gains are equal (homoscedastic scorer,
    # model-certain fallback: penalties nothing or constant), there are no
    # crossings, every mu shares one ordering, and the single-sort fast path
    # below is PROVEN exact (lemma consequence (i)) — it is taken untouched,
    # bit-identically.
    if penalties !== nothing && n > 1
        pmin = penalties[1]; pmax = penalties[1]
        @inbounds for i in 2:n
            pmin = min(pmin, penalties[i]); pmax = max(pmax, penalties[i])
        end
        if pmax > pmin   # heterogeneous gains ⇒ exact parametric enumeration
            return _parametric_sweep!(perm, lg_table, model_dof, log_two_outlier_halfwidth, dg,
                                      scores, penalties, prior_table,
                                      scale_prior, n)
        end
    end
    sort_target_full = isnothing(sort_key) ? scores : sort_key

    if length(perm) == n && length(sort_target_full) == n
        # Exact-size fast path (RANSAC scoring). With radix buffers in hand
        # the stable O(n) radix perm-sort replaces the comparison sortperm! —
        # the output permutation is bitwise identical (sortperm!'s documented
        # stability guarantee + the isless-exact key transform; pinned by
        # radix_sortperm_tests.jl incl. tie-heavy and NaN/±0.0 cases).
        if radix !== nothing && sort_target_full isa Union{Vector{Float64},Vector{Float32}} &&
           length(radix.keys) >= n && length(radix.keys2) >= n && length(radix.perm2) >= n
            _radix_sortperm!(perm, sort_target_full, radix, n)
        elseif !isnothing(scratch) && length(scratch) >= n
            sortperm!(perm, sort_target_full; scratch)
        else
            sortperm!(perm, sort_target_full)
        end
    else
        # Partial path (BB blob adaptation, CUDA ext host seed): sort views
        # of the first-n prefix of each oversized buffer.
        perm_v = @view perm[1:n]
        tgt_v = length(sort_target_full) == n ? sort_target_full : @view(sort_target_full[1:n])
        if !isnothing(scratch) && length(scratch) >= n
            sortperm!(perm_v, tgt_v; scratch)
        else
            sortperm!(perm_v, tgt_v)
        end
    end

    m = model_dof
    # Accumulate in Float64 REGARDLESS of the scores eltype: the sweep picks
    # k* from score differences between adjacent partitions, and a Float32
    # running RSS (cumsum error ~ n·eps·RSS) shifts the partition boundary.
    # Float32 storage upstream (RansacWorkspace{_, _, Float32}) is fine —
    # per-point precision is not the issue; the running sums are.
    dg_f = Float64(dg)
    RSS = 0.0
    pen_sum = 0.0
    best_S = -Inf
    best_k = 0
    @inbounds for k in 1:n
        idx = perm[k]
        RSS += Float64(scores[idx])
        if penalties !== nothing
            pen_sum += Float64(penalties[idx])
        end
        k <= m && continue
        # Clamp a COPY: the floor guards `log(RSS)` in this partition's score,
        # and must not leak into the running sum. It used to clamp `RSS` in
        # place, so on a (near-)noise-free prefix — RSS < eps after the sample
        # points — every later partition's RSS carried an extra ≈eps, and this
        # sweep scored a different S(k) sequence than the (since deleted)
        # soft sweep, the parametric sweep and the device kernel, which all
        # clamp a copy.
        S = _partition_score(scale_prior, lg_table, RSS, k, m, dg_f, n, log_two_outlier_halfwidth)
        if penalties !== nothing
            S -= pen_sum / 2
        end
        if prior_table !== nothing
            S += prior_table[k]
        end
        if S > best_S
            best_S = S
            best_k = k
        end
    end
    return best_S, best_k
end

"""
    sweep!(sws::ScoringWorkspace, sq_residuals, n;
           model_dof, log_two_outlier_halfwidth, codimension=1) -> (S*, k*)

The sort-and-sweep on a bare `ScoringWorkspace`, for callers without a `StudentTUniformScoring`.
Student-t / posterior IRLS callers have a `ScoringWorkspace` and raw
spec arguments (`model_dof`, `log_two_outlier_halfwidth`, codimension).

The scale prior is **not a parameter** here: it is a property of `sws`,
committed at allocation time (see [`ScoringWorkspace`](@ref)). Callers
that want IG choose the prior when building the sws via
`default_scoring_workspace(n; dg, scale_prior=IGPrior(...))`.

Used by BlobBoards blob adaptation (`posterior_adapt.jl`) and the CUDA
extension's host-side seeding loop. Both of these share one `sws` across many calls with varying
`n`, so the partial-buffer path in the low-level kernel is the common case.

Forwarder — no sweep loop of its own.
"""
sweep!(sws::ScoringWorkspace, sq_residuals::AbstractVector{T}, n::Int;
       model_dof::Int, log_two_outlier_halfwidth::Float64,
       codimension::Int=1) where T =
    sweep!(sws.perm, sws.lg_table, model_dof, log_two_outlier_halfwidth, codimension,
           sq_residuals, n; scratch=sws.scratch, scale_prior=sws.scale_prior,
           radix=sws.radix)


# =============================================================================
# Parametric sweep — the exact enumeration of the generalized sweep lemma
# (uncertain_evidence 35ae25c, §"The generalized sweep"). Per-point gains
# g_i = -penalties_i/2 (the covariance penalty -1/2·log|V_i|; per-point
# prior/floor terms drop into the SAME vector when they exist), unit-scale
# residuals q_i = scores_i. The key t_i(μ) = g_i - μ·q_i is linear in μ, so
# the descending-t ordering is CONSTANT on each open interval between
# pairwise crossings μ_ij = (g_i - g_j)/(q_i - q_j) (positive only for
# CONFLICTED pairs: larger gain together with larger residual). Enumerating
# one representative μ per interval — below the first crossing, between
# consecutive distinct crossings, above the last — visits every top-k set
# attainable at any μ > 0; the optimal support is one of them (Lemma
# "Generalized sweep": it is a top-k set at its own self-consistent
# μ_I = ν_k/(2Q_I); if μ_I lands exactly ON a crossing, the gradient
# inequality makes the tie-partner support optimal too, and that one is
# realized on the adjacent open interval — interval interiors suffice).
#
# KINETIC enumeration (2026-09-22; it replaced a re-sort per interval). The
# ordering changes only AT a crossing, and a lone crossing (i, j) swaps one
# ADJACENT pair: the prefix sets below and above that position are unchanged,
# so only the one prefix at k = min(pos) is a new top-k set and is scored. The
# walk visits the crossings in ascending μ:
#   * the first interval is sorted and every prefix scored (Θ(n log n));
#   * a group of crossings that coincide (within 1e-13 relative of the group's
#     first), or a lone crossing whose pair is not adjacent (floating-point
#     disagreement between the crossing and the ordering), re-sorts at the
#     next interval's representative μ and re-scores every prefix — what the
#     retired version did at EVERY interval;
#   * every other crossing is one swap and one scored prefix, O(1).
# Total O(n² log n) for the crossings plus O(n log n) per re-sort, against the
# retired O(C·n log n). Measured (BenchmarkTools minimum, predictive-like
# instances): n = 60, 2681 → 36.2 µs (74×); n = 250, 139.0 → 0.716 ms (194×).
# On RVG's predictive problems 1.1–2.2 ms → 0.04–0.08 ms (n = 60) and
# 127–153 ms → 0.9–1.1 ms (n = 250) per hypothesis, with 56/56 full predictive
# RANSAC runs identical (error, trials, hypothesis count, inlier mask).
# Prefix sums are carried forward across a
# swap rather than re-summed, so S* agrees with a re-summing enumeration to
# summation order (≲ 1e-15 relative).
#
# A plain generic function, not a KernelAbstractions kernel: it is an ordered
# event walk over one hypothesis — sequential by construction — and it runs on
# the latency-bound per-hypothesis path (`sweep!`'s docstring).
#
# The historical "raw r² sort to block high-leverage infiltration" is
# RETIRED: the covariance penalty sits inside the key, so uncertain points
# are down-ranked by the ordering itself (lemma consequence (iv)).
# =============================================================================
function _parametric_sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                            model_dof::Int, log_two_outlier_halfwidth::Float64, dg::Int,
                            scores::AbstractVector{T}, penalties::AbstractVector{Float64},
                            prior_table::Union{Nothing,AbstractVector{Float64}},
                            scale_prior::IGPrior, n::Int) where T
    # all positive pairwise crossings (conflicted pairs only), with their pair
    cross_μ = Float64[]; cross_i = Int[]; cross_j = Int[]
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        dq = Float64(scores[i]) - Float64(scores[j])
        dq == 0.0 && continue
        μ = ((penalties[j] - penalties[i]) / 2) / dq     # (g_i - g_j)/(q_i - q_j)
        if μ > 0.0
            push!(cross_μ, μ); push!(cross_i, i); push!(cross_j, j)
        end
    end
    order = sortperm(cross_μ)
    nc = length(order)
    # groups of coinciding crossings: `group_start[g]` indexes `order`
    group_start = Int[]
    if nc > 0
        push!(group_start, 1)
        prev = cross_μ[order[1]]
        @inbounds for r in 2:nc
            c = cross_μ[order[r]]
            if c > prev * (1 + 1e-13) + 1e-300
                push!(group_start, r); prev = c
            end
        end
    end
    ng = length(group_start)

    ord = Vector{Int}(undef, n)          # the current descending-t(μ) order
    pos = Vector{Int}(undef, n)          # its inverse
    rss = Vector{Float64}(undef, n)      # prefix sums of q (Float64 regardless of T)
    pen = Vector{Float64}(undef, n)      # prefix sums of the penalties
    bestord = Vector{Int}(undef, n)
    best = _sweep_all_prefixes!(ord, pos, rss, pen, bestord, (-Inf, 0),
                                _interval_mu(cross_μ, order, group_start, 1),
                                lg_table, model_dof, log_two_outlier_halfwidth, dg, scores, penalties,
                                prior_table, scale_prior, n)
    @inbounds for g in 1:ng
        r0 = group_start[g]
        r1 = g == ng ? nc : group_start[g + 1] - 1
        if r0 == r1                                       # a lone crossing
            a = cross_i[order[r0]]; b = cross_j[order[r0]]
            pa = pos[a]; pb = pos[b]
            if abs(pa - pb) == 1                          # ... of an adjacent pair: swap
                p = min(pa, pb)
                x = ord[p]; y = ord[p + 1]
                ord[p] = y; ord[p + 1] = x; pos[y] = p; pos[x] = p + 1
                rss[p] = (p == 1 ? 0.0 : rss[p - 1]) + Float64(scores[y])
                pen[p] = (p == 1 ? 0.0 : pen[p - 1]) + Float64(penalties[y])
                if p > model_dof                          # the one new prefix set
                    S = _prefix_score(rss, pen, p, lg_table, model_dof, log_two_outlier_halfwidth, dg,
                                      prior_table, scale_prior, n)
                    if S > best[1]
                        best = (S, p)
                        copyto!(bestord, ord)
                    end
                end
                continue
            end
        end
        # coinciding crossings, or a non-adjacent pair: re-sort the next interval
        best = _sweep_all_prefixes!(ord, pos, rss, pen, bestord, best,
                                    _interval_mu(cross_μ, order, group_start, g + 1),
                                    lg_table, model_dof, log_two_outlier_halfwidth, dg, scores, penalties,
                                    prior_table, scale_prior, n)
    end
    # perm[1:k*] IS the selected support (ConsensusSet/mask contract)
    if length(perm) == n
        copyto!(perm, bestord)
    else
        @inbounds for k in 1:n; perm[k] = bestord[k]; end
    end
    return best
end

# The representative μ of ordering interval g: below the first crossing (g = 1),
# the midpoint of consecutive groups' first crossings, above the last (g = ng+1).
function _interval_mu(cross_μ, order, group_start, g)
    nc = length(order); ng = length(group_start)
    nc == 0 && return 1.0
    g == 1 && return cross_μ[order[1]] / 2
    g == ng + 1 && return cross_μ[order[nc]] * 2 + 1.0
    return (cross_μ[order[group_start[g - 1]]] + cross_μ[order[group_start[g]]]) / 2
end

# The partition score of the prefix k of the current order (`sweep!`'s formula,
# with the per-point penalties and the profiled-γ table).
@inline function _prefix_score(rss, pen, k, lg_table, model_dof, log_two_outlier_halfwidth, dg,
                               prior_table, scale_prior, n)
    S = _partition_score(scale_prior, lg_table, Float64(rss[k]), k, model_dof, Float64(dg),
                         n, log_two_outlier_halfwidth) - pen[k] / 2
    prior_table !== nothing && (S += Float64(prior_table[k]))
    return S
end

# Sort by descending t(μ) — ties by (ascending q, index), deterministic — and
# score every prefix; returns the running best (S, k), `bestord` its order.
function _sweep_all_prefixes!(ord, pos, rss, pen, bestord, best, μ, lg_table, model_dof,
                              log_two_outlier_halfwidth, dg, scores, penalties, prior_table, scale_prior, n)
    sortperm!(ord, 1:n;
              by = i -> (-(-penalties[i] / 2 - μ * Float64(scores[i])), Float64(scores[i]), i))
    r = 0.0; pl = 0.0
    @inbounds for k in 1:n
        idx = ord[k]; pos[idx] = k
        r += Float64(scores[idx]); pl += Float64(penalties[idx])
        rss[k] = r; pen[k] = pl
        k <= model_dof && continue
        S = _prefix_score(rss, pen, k, lg_table, model_dof, log_two_outlier_halfwidth, dg, prior_table, scale_prior, n)
        if S > best[1]
            best = (S, k)
            copyto!(bestord, ord)
        end
    end
    return best
end
