#!/usr/bin/env julia
# =============================================================================
# neurips2026/scripts/studies/f32_radix_probe.jl — DECISION PROBE for the Float32 scoring hypothesis.
#
# The scoring hot loop's dominant structure is an LSD radix `sortperm` over
# monotone UInt64 keys (sweep.jl `_radix_sortperm!`): 8 byte iterations, each
# scattering an 8-byte key + an 8-byte Int perm entry. A Float32 sort key is
# UInt32 — 4 iterations at half the key width — and an Int32 perm halves the
# scatter payload again. This probe measures whether that structural
# reduction actually buys wall-clock at OUR problem sizes (n ≈ 700–1600
# points, which fit in L1/L2, so the bandwidth argument alone is not
# decisive), on the REAL residual distributions of a PhotoTourism pool.
#
# It measures the SORT only — no package changes, no refactor. If the sort
# shows no win here, the Tq parameterization of the scoring path is not
# worth its complexity and the investigation stops.
#
# Usage: julia -t 1 --project=neurips2026 neurips2026/scripts/studies/f32_radix_probe.jl
# =============================================================================
using RANSACScoringDoneRight
const RSDR = RANSACScoringDoneRight
import JLD2
using Printf
using VisualGeometryToolkitCore.Pools: num_points
using VisualGeometryToolkitEval.Pools: load_pool, load_pool_residuals, residuals_path, residual_matrix

# The data root: ENV["CAU_DATA_ROOT"] (VisualGeometryToolkitEval's data-root variable).
const DATA_ROOT = get(ENV, "CAU_DATA_ROOT") do
    error("set CAU_DATA_ROOT to the data root holding accv2026/data/PT-E-SIFT-4K")
end
const POOL  = joinpath(DATA_ROOT, "accv2026", "data", "PT-E-SIFT-4K", "british_museum.jld2")
const REPS  = 5

# --- the three sort variants ------------------------------------------------
# V1 = the production path, verbatim (F64 key → UInt64, 8 iterations, Int perm).
# V2 = F32 key → UInt32, 4 iterations, Int perm.
# V3 = F32 key → UInt32, 4 iterations, Int32 perm (half the scatter payload).

@inline function _key32(x::Float32)
    isnan(x) && return 0xffffffff
    u = reinterpret(UInt32, x)
    return (u & 0x80000000) == zero(UInt32) ? (u | 0x80000000) : ~u
end

struct Radix32{P<:Integer}
    keys::Vector{UInt32}
    keys2::Vector{UInt32}
    perm2::Vector{P}
    counts::Matrix{Int}
end
Radix32{P}(n::Int) where {P} = Radix32{P}(Vector{UInt32}(undef, n), Vector{UInt32}(undef, n),
                                          Vector{P}(undef, n), Matrix{Int}(undef, 256, 4))

function radix32_sortperm!(perm::Vector{P}, target::AbstractVector{Float32},
                           rb::Radix32{P}, n::Int) where {P<:Integer}
    keys = rb.keys; keys2 = rb.keys2; perm2 = rb.perm2; counts = rb.counts
    @inbounds for i in 1:n
        keys[i] = _key32(target[i])
        perm[i] = P(i)
    end
    fill!(counts, 0)
    @inbounds for i in 1:n
        k = keys[i]
        counts[Int(k & 0xff) + 1, 1] += 1
        counts[Int((k >> 8) & 0xff) + 1, 2] += 1
        counts[Int((k >> 16) & 0xff) + 1, 3] += 1
        counts[Int((k >> 24) & 0xff) + 1, 4] += 1
    end
    src_k, dst_k = keys, keys2
    src_p, dst_p = perm, perm2
    @inbounds for d in 1:4
        b1 = 1
        while counts[b1, d] == 0
            b1 += 1
        end
        counts[b1, d] == n && continue
        off = 0
        for b in 1:256
            c = counts[b, d]; counts[b, d] = off; off += c
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
    src_p !== perm && copyto!(perm, 1, src_p, 1, n)
    return nothing
end

# --- harvest REAL residual vectors from the pool ----------------------------
pool  = load_pool(POOL)
pairs = pool.pairs
byn   = sortperm(pairs; by = num_points, rev = true)
groups = (("typical", collect(1:20)), ("dense", byn[1:20]))
res   = load_pool_residuals(residuals_path(POOL), pool;
                            pairs = unique(vcat(last.(groups)...)))

function residual_sets(ix)
    out = Vector{Float64}[]
    for i in ix
        D = residual_matrix(res, i)         # npts × nmodels, Float32 on disk
        for j in 1:min(size(D, 2), 50)      # 50 hypotheses per pair is plenty
            push!(out, Float64.(@view D[:, j]))
        end
    end
    return out
end

function bench(sets, label)
    n = length(first(sets))
    perm64 = Vector{Int}(undef, n); rb64 = RSDR.RadixPermBuffers{UInt64}(n)   # V1 is the UInt64 path; the bare default is UInt32 since this probe
    perm32 = Vector{Int}(undef, n); rb32 = Radix32{Int}(n)
    permI32 = Vector{Int32}(undef, n); rbI32 = Radix32{Int32}(n)
    sets32 = [Float32.(s) for s in sets]

    # warmup
    RSDR._radix_sortperm!(perm64, sets[1], rb64, n)
    radix32_sortperm!(perm32, sets32[1], rb32, n)
    radix32_sortperm!(permI32, sets32[1], rbI32, n)

    t64 = Inf; t32 = Inf; tI32 = Inf
    for _ in 1:REPS
        t0 = time_ns(); for s in sets;   RSDR._radix_sortperm!(perm64, s, rb64, n) end
        t64 = min(t64, (time_ns() - t0) / length(sets))
        t0 = time_ns(); for s in sets32; radix32_sortperm!(perm32, s, rb32, n) end
        t32 = min(t32, (time_ns() - t0) / length(sets))
        t0 = time_ns(); for s in sets32; radix32_sortperm!(permI32, s, rbI32, n) end
        tI32 = min(tI32, (time_ns() - t0) / length(sets))
    end

    # ORDER AGREEMENT: does the F32 permutation rank identically to F64?
    # (ties may legitimately reorder; count strict positional agreement and
    # the worst |Δ| between the F64-sorted and F32-sorted residual sequences)
    agree = 0; worst = 0.0
    for (s, s32) in zip(sets, sets32)
        RSDR._radix_sortperm!(perm64, s, rb64, n)
        radix32_sortperm!(perm32, s32, rb32, n)
        agree += (perm64 == perm32)
        for i in 1:n
            worst = max(worst, abs(s[perm64[i]] - s[perm32[i]]))
        end
    end

    @printf("%-9s n=%-5d  F64 %7.2f µs | F32 %7.2f µs (%.2fx) | F32+I32 %7.2f µs (%.2fx) | perm-identical %d/%d, worst Δresid %.3e\n",
            label, n, t64/1e3, t32/1e3, t64/t32, tI32/1e3, t64/tI32,
            agree, length(sets), worst)
    return (; label, n, t64, t32, tI32, agree, nsets = length(sets), worst)
end

results = NamedTuple[]
# group by exact point count so each bench uses one n (buffers are sized once)
for (label, ix) in groups
    sets = residual_sets(ix)
    bylen = Dict{Int,Vector{Vector{Float64}}}()
    for s in sets; push!(get!(bylen, length(s), Vector{Float64}[]), s); end
    n_best = argmax(k -> length(bylen[k]), collect(keys(bylen)))
    push!(results, bench(bylen[n_best], label))
end

out = joinpath(DATA_ROOT, "accv2026", "results", "score_timing", "f32_radix_probe.jld2")
mkpath(dirname(out))
JLD2.jldsave(out; results, reps = REPS, pool = POOL)
println("\nsaved -> ", out)
