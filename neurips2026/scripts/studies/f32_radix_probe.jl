#!/usr/bin/env julia
# =============================================================================
# neurips2026/scripts/studies/f32_radix_probe.jl — DECISION PROBE for the Float32 scoring hypothesis.
#
# The scoring hot loop's dominant structure is an LSD radix `sortperm` over
# monotone UInt64 keys (VisualGeometryToolkitCore's `radix_sortperm!`, robust/partition_sweep.jl; RSDR's sweep.jl until 2026-11-11): 8 byte iterations, each
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
# reproduce: reads configs/f32_radix_probe.toml
#
# STAGES
#   sort  the three sort variants on the typical and dense pairs of [inputs.pool]
#         → <results>/phototourism/sift/E/f32_radix_probe/sort-<run key>/probe.jld2
#
#   julia -t 1 --project=. scripts/studies/f32_radix_probe.jl [--plan] [--stage=NAME] [--force]
# =============================================================================
using DrWatson
@quickactivate "NeurIPS2026"
include(srcdir("NeurIPS2026.jl"))
using RANSACScoringDoneRight
import VisualGeometryToolkitCore as VgtkCore   # the sweep's radix sort (Core's)
using Printf
using VisualGeometryToolkitCore.Pools: num_points
using VisualGeometryToolkitEval.Pools: load_pool, load_pool_residuals, residual_matrix

const CFG_PATH = projectdir("configs", "f32_radix_probe.toml")
const CFG = TOML.parsefile(CFG_PATH)
const KEY = run_key(CFG)
const OUTDIR = keyed_cell_dir(CFG, "f32_radix_probe", "sort")
const RESULT = "probe.jld2"

# --- the three sort variants ------------------------------------------------
# V1 = F64 key → UInt64, 8 iterations, Int perm (VisualGeometryToolkitCore's radix at UInt64 keys).
# V2 = F32 key → UInt32, 4 iterations, Int perm (VisualGeometryToolkitCore's radix at its default
#      UInt32 keys: the production path since this probe).
# V3 = F32 key → UInt32, 4 iterations, Int32 perm (half the scatter payload): the one
#      variant Core's radix does not take (its perm is a Vector{Int}), so its sort is
#      below, on Core's key (`radix_key`).

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
        keys[i] = VgtkCore.radix_key(UInt32, target[i])
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
        # paper-project: allow C2 the 256 buckets of one byte digit: the radix sort's algorithm, not data
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

# --- REAL residual vectors from the pool ------------------------------------
# `nhyp` hypotheses of each pair `ix` of the residuals `res`, as Float64 vectors.
function residual_sets(res, ix, nhyp)
    out = Vector{Float64}[]
    for i in ix
        D = residual_matrix(res, i)         # npts × nmodels, Float32 on disk
        for j in 1:min(size(D, 2), nhyp)
            push!(out, Float64.(@view D[:, j]))
        end
    end
    return out
end

function bench(sets, label, reps)
    n = length(first(sets))
    perm64 = Vector{Int}(undef, n); rb64 = VgtkCore.RadixPermBuffers{UInt64}(n)   # V1 is the UInt64 path; the bare default is UInt32 since this probe
    perm32 = Vector{Int}(undef, n); rb32 = VgtkCore.RadixPermBuffers{UInt32}(n)
    permI32 = Vector{Int32}(undef, n); rbI32 = Radix32{Int32}(n)
    sets32 = [Float32.(s) for s in sets]

    # warmup
    VgtkCore.radix_sortperm!(perm64, sets[1], rb64, n)
    VgtkCore.radix_sortperm!(perm32, sets32[1], rb32, n)
    radix32_sortperm!(permI32, sets32[1], rbI32, n)

    t64 = Inf; t32 = Inf; tI32 = Inf
    for _ in 1:reps
        t0 = time_ns(); for s in sets;   VgtkCore.radix_sortperm!(perm64, s, rb64, n) end
        t64 = min(t64, (time_ns() - t0) / length(sets))
        t0 = time_ns(); for s in sets32; VgtkCore.radix_sortperm!(perm32, s, rb32, n) end
        t32 = min(t32, (time_ns() - t0) / length(sets))
        t0 = time_ns(); for s in sets32; radix32_sortperm!(permI32, s, rbI32, n) end
        tI32 = min(tI32, (time_ns() - t0) / length(sets))
    end

    # ORDER AGREEMENT: does the F32 permutation rank identically to F64?
    # (ties may legitimately reorder; count strict positional agreement and
    # the worst |Δ| between the F64-sorted and F32-sorted residual sequences)
    agree = 0; worst = 0.0
    for (s, s32) in zip(sets, sets32)
        VgtkCore.radix_sortperm!(perm64, s, rb64, n)
        VgtkCore.radix_sortperm!(perm32, s32, rb32, n)
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

function probe()
    p = CFG["probe"]
    Threads.nthreads() == p["threads"] || error("$(p["threads"]) thread(s) by protocol — use `julia -t $(p["threads"])`")
    path, rpath = input_path(CFG, "pool"), input_path(CFG, "residuals")
    pool  = load_pool(path)
    pairs = pool.pairs
    k     = min(p["npairs"], length(pairs))
    byn   = sortperm(pairs; by = num_points, rev = true)
    groups = (("typical", collect(1:k)), ("dense", byn[1:k]))
    res   = load_pool_residuals(rpath, pool; pairs = unique(vcat(last.(groups)...)))
    results = NamedTuple[]
    # group by exact point count so each bench uses one n (buffers are sized once)
    for (label, ix) in groups
        sets = residual_sets(res, ix, p["hypotheses_per_pair"])
        bylen = Dict{Int,Vector{Vector{Float64}}}()
        for s in sets; push!(get!(bylen, length(s), Vector{Float64}[]), s); end
        n_best = argmax(k -> length(bylen[k]), collect(keys(bylen)))
        push!(results, bench(bylen[n_best], label, p["reps"]))
    end
    mkpath(OUTDIR)
    JLD2.jldsave(joinpath(OUTDIR, RESULT); results)
    write_run_records(OUTDIR, CFG; driver = "studies/f32_radix_probe.jl", config_source = CFG_PATH, key = KEY,
                      inputs = [path, rpath])
    return nothing
end

const STAGES = [
    Stage(:sort, "the F64 / F32 / F32+Int32 radix sorts on real residuals, keyed by the run key",
          () -> run_complete(OUTDIR; sidecar = RESULT), probe),
]

run_stages("f32_radix_probe", STAGES)
