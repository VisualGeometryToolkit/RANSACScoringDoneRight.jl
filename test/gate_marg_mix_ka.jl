#!/usr/bin/env julia
# =============================================================================
# gate_marg_mix_ka.jl — Stage-6 acceptance gate for the Track-B GPU port.
#
# Proves the device path (marg_mix_evidence_kernel! → marg_mix_profile_kernel!
# → argmax_models_kernel!) is BIT-IDENTICAL to a plain host loop (the
# incremental data term, the trapezoid logsumexp, the O(n) outlier-half-width
# profile re-argmax)
# on a CPU() backend at
# Float64, for both the Jeffreys and a proper IG prior, across all three
# redescending families.  Also reports the Float32 selection drift
# (informational — near-ties may flip; the production setting).
#
#   julia --project=. test/gate_marg_mix_ka.jl                       # CPU-only
#   julia --startup-file=no --project=<env-with-CUDA> \
#         test/gate_marg_mix_ka.jl --cuda                            # + GPU run
#
# NOTE: `--startup-file=no` matters for the --cuda form — the local startup.jl
# auto-activates the cwd's project, which silently overrides `--project` and
# drops CUDA off the load path (RSDR itself does not depend on CUDA; run the
# GPU run from an environment that does, e.g. VisualGeometryToolkitEstimation's).
#
# Exit 0 when every check succeeds, 1 on FAIL.  Synthetic data only — no dataset needed, so this
# runs anywhere (CI, laptop, cluster).  On a CUDA machine, `--cuda` adds a
# CUDABackend Float64 run gated against the same reference (tolerance 1e-12,
# not bit-exact: GPU exp/log are correctly-rounded but not guaranteed
# identical to the host libm).
# =============================================================================

using RANSACScoringDoneRight
const RSDR = RANSACScoringDoneRight
using RANSACScoringDoneRight: IGSharedData, LossSpec,
    loss_spec, loss_scale_bounds,
    grid_evaluate,
    WelschMarginal, SmoothTLSMarginal
using KernelAbstractions
import VisualGeometryToolkitCore as VGC
using VisualGeometryToolkitCore: WelschShape, GemanMcclureShape, SmoothTLSShape, IGPrior
using Random, Printf
using LogExpFunctions: logsumexp

# Conditional CUDA load must happen at top level (run with an environment that
# provides CUDA, e.g. VisualGeometryToolkitEstimation's).
const USE_CUDA = "--cuda" in ARGS
USE_CUDA && @eval using CUDA

const M_DOF = 5
const DG    = 1

# -----------------------------------------------------------------------------
# Synthetic staged data — hand-built IGSharedData on any backend.
# -----------------------------------------------------------------------------
function synth_shared(rng::AbstractRNG; P::Int = 6, backend = CPU())
    K_arr = Int32[rand(rng, 60:140) for _ in 1:P]
    M_arr = Int32[rand(rng, 3:8) for _ in 1:P]
    pair_offset   = zeros(Int64, P + 1)
    errors_offset = zeros(Int64, P + 1)
    for p in 1:P
        pair_offset[p + 1]   = pair_offset[p]   + Int64(K_arr[p]) * M_arr[p]
        errors_offset[p + 1] = errors_offset[p] + Int64(M_arr[p])
    end
    flat = Vector{Float32}(undef, pair_offset[end])
    errs = Vector{Float64}(undef, errors_offset[end])
    for p in 1:P
        K = Int(K_arr[p]); σ = 10.0^rand(rng, -3.0:0.25:-2.0)
        for m in 1:Int(M_arr[p])
            col = Vector{Float64}(undef, K)
            col[1:M_DOF] .= 0.0                              # minimal sample: exact fit
            for i in (M_DOF + 1):K
                d = rand(rng) < 0.7 ? abs(σ * randn(rng)) : 0.5 * rand(rng)
                col[i] = d * d
            end
            sort!(col)
            base = pair_offset[p] + (m - 1) * K
            @inbounds for i in 1:K
                flat[base + i] = Float32(col[i])
            end
            errs[errors_offset[p] + m] = 10.0 * rand(rng)
        end
    end
    flat_d = KernelAbstractions.allocate(backend, Float32, length(flat))
    KernelAbstractions.copyto!(backend, flat_d, flat)
    errs_d = KernelAbstractions.allocate(backend, Float64, length(errs))
    KernelAbstractions.copyto!(backend, errs_d, errs)
    po_d  = KernelAbstractions.allocate(backend, Int64, P + 1)
    KernelAbstractions.copyto!(backend, po_d, pair_offset)
    eo_d  = KernelAbstractions.allocate(backend, Int64, P + 1)
    KernelAbstractions.copyto!(backend, eo_d, errors_offset)
    kpp_d = KernelAbstractions.allocate(backend, Int32, P)
    KernelAbstractions.copyto!(backend, kpp_d, K_arr)
    mpp_d = KernelAbstractions.allocate(backend, Int32, P)
    KernelAbstractions.copyto!(backend, mpp_d, M_arr)
    shared = IGSharedData(backend, flat_d, po_d, kpp_d, mpp_d, Int(maximum(M_arr)),
                            errs_d, eo_d)
    return shared, flat, pair_offset, K_arr, M_arr, errs, errors_offset
end

# -----------------------------------------------------------------------------
# Host oracle for the Track-B mixture (grid_evaluate on a LossSpec). Its host
# evaluation (`inlier_evidence!` + the outlier-half-width profile) is deleted — the KA kernels
# are the one evaluation, on CPU() as on a GPU — so the oracle is a PLAIN host
# loop, written from VisualGeometryToolkitCore's ρ and the spec's quadrature tables:
# per (pair, model) the incremental data term D_j = Σ_{i≤k} Δ·ρ(shape, x_i/w²_j, 1),
# the trapezoid logsumexp (`LogExpFunctions.logsumexp`) over nodes j, the
# outlier-half-width profile max over k and the model argmax — strict `>` and ascending order, as the device kernels.
# -----------------------------------------------------------------------------
function reference_errs(spec::LossSpec, flat, pair_offset, K_arr, M_arr,
                        errs, errors_offset, outlier_halfwidth_grid)
    P = length(K_arr); T = length(outlier_halfwidth_grid)
    q = spec.quad; NG = length(q.sq_loss_width_nodes); Δ = spec.Δ; sh = spec.shape
    out  = Matrix{Float64}(undef, P, T)
    for p in 1:P
        K = Int(K_arr[p]); M = Int(M_arr[p])
        E = fill(-Inf, K, M)
        for m in 1:M
            base = pair_offset[p] + (m - 1) * K
            D = zeros(NG)
            for k in 1:K
                x = Float64(flat[base + k])               # same rounded inputs
                for j in 1:NG
                    D[j] += Δ * VGC.ρ(sh, x / q.sq_loss_width_nodes[j], 1.0)
                end
                k <= M_DOF && continue
                ke = Float64(k - M_DOF)
                E[k, m] = logsumexp(-(D[j] + ke * q.logZnodes[j] + q.base[j]) + q.ltw[j]
                                    for j in 1:NG)
            end
        end
        for (t, halfwidth) in enumerate(outlier_halfwidth_grid)
            outlier_cost = Float64(DG) * log(2 * Float64(halfwidth))
            best_S = -Inf; best_m = 1
            for m in 1:M
                Sm = -Inf
                for k in 1:K
                    S = E[k, m] - (K - k) * outlier_cost
                    S > Sm && (Sm = S)
                end
                if Sm > best_S
                    best_S = Sm; best_m = m
                end
            end
            out[p, t] = errs[errors_offset[p] + best_m]
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Host oracle for the FULL-DATA scale marginal (grid_evaluate on a ScaleMarginal). The KA
# kernel is the one evaluation, on CPU() as on a GPU, so the oracle is a PLAIN
# host loop, written from VisualGeometryToolkitCore's ρ and the spec's quadrature
# tables (`loss_spec(::ScaleMarginal)`): per (pair, model, grid point) the
# data term D_j = Σ_i Δ·ρ(shape, x_i/w²_j, 1) over ALL K residuals, the
# trapezoid logsumexp (`LogExpFunctions.logsumexp`) with K·logZ̃_j, and the
# model argmax — strict `>`, ascending, as the device kernels.
# -----------------------------------------------------------------------------
function reference_marg_errs(family_sym::Symbol, flat, pair_offset, K_arr, M_arr,
                             errs, errors_offset, prior_loss_width_grid, outlier_halfwidth)
    P = length(K_arr); T = length(prior_loss_width_grid)
    out = Matrix{Float64}(undef, P, T)
    for (t, prior_loss_width) in enumerate(prior_loss_width_grid)
        m = family_sym === :welsch ?
            WelschMarginal(; prior_loss_width = Float64(prior_loss_width), outlier_halfwidth) :
            SmoothTLSMarginal(; prior_loss_width = Float64(prior_loss_width), outlier_halfwidth)
        spec = loss_spec(m); q = spec.quad; NG = length(q.sq_loss_width_nodes)
        for p in 1:P
            K = Int(K_arr[p]); M = Int(M_arr[p])
            best_s = -Inf; best_m = 1
            for mm in 1:M
                base = pair_offset[p] + (mm - 1) * Int64(K)
                D = zeros(NG)
                for k in 1:K, j in 1:NG
                    D[j] += spec.Δ * VGC.ρ(spec.shape, Float64(flat[base + k]) / q.sq_loss_width_nodes[j], 1.0)
                end
                s = logsumexp(-(D[j] + K * q.logZnodes[j] + q.base[j]) + q.ltw[j] for j in 1:NG)
                if s > best_s
                    best_s = s; best_m = mm
                end
            end
            out[p, t] = errs[errors_offset[p] + best_m]
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Run the gate
# -----------------------------------------------------------------------------
function run_gate(; cuda::Bool = false)
    rng = MersenneTwister(20260702)
    shared, flat, pair_offset, K_arr, M_arr, errs, errors_offset = synth_shared(rng)
    outlier_halfwidth_grid = collect(10.0 .^ range(-2.5, 0.0; length = 9))

    # Global scale bounds over the pooled positive squared residuals.
    pooled = Float64.(filter(x -> x > 0, flat))
    t_lo, t_hi = loss_scale_bounds(pooled, length(pooled))

    fails = 0
    for (shape, name) in ((WelschShape(), "welsch"),
                           (GemanMcclureShape(), "geman"),
                           (SmoothTLSShape(), "smoothtls")),
        (prior, pname) in ((IGPrior(), "jeffreys"),
                           (IGPrior(2.0, 2.0e-5), "IG(2,2e-5)")),
        Δ in (0.5, 2.0)

        spec = loss_spec(shape, prior; Δ = Δ, t_lo = t_lo, t_hi = t_hi, NG = 257)
        ref = reference_errs(spec, flat, pair_offset, K_arr, M_arr, errs, errors_offset, outlier_halfwidth_grid)

        dev64 = grid_evaluate(Float64, spec, shared, outlier_halfwidth_grid;
                                          model_dof = M_DOF, dg = DG)
        exact = dev64 == ref
        exact || (fails += 1)
        dev32 = grid_evaluate(Float32, spec, shared, outlier_halfwidth_grid;
                                          model_dof = M_DOF, dg = DG)
        drift = count(dev32 .!= ref) / length(ref)
        @printf("%-10s %-11s Δ=%-4.1f  F64 bit-exact: %-5s   F32 selection drift: %5.1f%%\n",
                name, pname, Δ, exact ? "OK" : "FAIL", 100 * drift)
    end

    # ---- Multi-chunk regression (same backend, cs < P vs cs ≥ P) ----
    # The 2026-07 production crash: KA copyto! from a STRIDED device view in
    # _chunked_argmax_eval leaked the previous chunk's rows into later chunks
    # on CUDA (phantom model indices > M(p)).  Same-backend comparison across
    # chunk sizes is order-identical, so it must be BIT-exact — no tie caveat.
    let spec = loss_spec(WelschShape(), IGPrior(); Δ = 0.5,
                           t_lo = t_lo, t_hi = t_hi, NG = 65)
        e1, i1 = grid_evaluate(Float64, spec, shared, outlier_halfwidth_grid;
                                           model_dof = M_DOF, dg = DG, return_idxs = true)
        e2, i2 = grid_evaluate(Float64, spec, shared, outlier_halfwidth_grid;
                                           model_dof = M_DOF, dg = DG, return_idxs = true,
                                           mem_budget = 1)   # cs = 1 → P chunks
        ok = (e1 == e2) && (i1 == i2) &&
             all(Int(i2[p, t]) <= M_arr[p] for p in eachindex(M_arr), t in eachindex(outlier_halfwidth_grid))
        ok || (fails += 1)
        @printf("multi-chunk (cs=1) CPU        F64 bit-exact: %s\n", ok ? "OK" : "FAIL")
    end

    # ---- Full-data scale marginal (grid_evaluate on a ScaleMarginal) ----
    outlier_halfwidth = 1.0
    prior_loss_width_grid = collect(10.0 .^ range(-3.0, -1.0; length = 5))
    for fam in (:welsch, :smoothtls_marg)
        ref = reference_marg_errs(fam, flat, pair_offset, K_arr, M_arr,
                                  errs, errors_offset, prior_loss_width_grid, outlier_halfwidth)
        m = fam === :welsch ? WelschMarginal(; prior_loss_width = 1.0, outlier_halfwidth) :
                              SmoothTLSMarginal(; prior_loss_width = 1.0, outlier_halfwidth)
        dev64 = grid_evaluate(Float64, m, shared, prior_loss_width_grid)
        exact = dev64 == ref
        exact || (fails += 1)
        @printf("full-data %-14s        F64 bit-exact: %s\n", fam,
                exact ? "OK" : "FAIL")
    end

    if cuda
        @info "CUDA run requested — staging on CUDABackend"
        backend = CUDA.CUDABackend()
        sh_gpu, = synth_shared(MersenneTwister(20260702); backend)
        spec = loss_spec(WelschShape(), IGPrior(); Δ = 0.5,
                           t_lo = t_lo, t_hi = t_hi, NG = 257)
        ref = reference_errs(spec, flat, pair_offset, K_arr, M_arr, errs, errors_offset, outlier_halfwidth_grid)
        devg = grid_evaluate(Float64, spec, sh_gpu, outlier_halfwidth_grid;
                                         model_dof = M_DOF, dg = DG)
        ok = maximum(abs.(devg .- ref)) <= 1e-12
        ok || (fails += 1)
        @printf("CUDA F64 (welsch, jeffreys, Δ=0.5): %s\n", ok ? "OK" : "FAIL")

        # Multi-chunk regression on the DEVICE (the configuration that shipped
        # broken): single-chunk vs cs=1 on the same backend must be bit-exact.
        eg1, ig1 = grid_evaluate(Float64, spec, sh_gpu, outlier_halfwidth_grid;
                                             model_dof = M_DOF, dg = DG, return_idxs = true)
        eg2, ig2 = grid_evaluate(Float64, spec, sh_gpu, outlier_halfwidth_grid;
                                             model_dof = M_DOF, dg = DG, return_idxs = true,
                                             mem_budget = 1)
        okc = (eg1 == eg2) && (ig1 == ig2) &&
              all(Int(ig2[p, t]) <= M_arr[p] for p in eachindex(M_arr), t in eachindex(outlier_halfwidth_grid))
        okc || (fails += 1)
        @printf("multi-chunk (cs=1) CUDA       F64 bit-exact: %s\n", okc ? "OK" : "FAIL")
    end

    if fails == 0
        println("GATE OK — the KA kernels agree bit for bit with the host oracles.")
        return 0
    else
        println("GATE FAIL — $fails configuration(s) diverged from the host oracle.")
        return 1
    end
end

# Standalone: exit with the gate status.  Under `include` (the "marg_mix_gate"
# test item, marg_mix_gate_tests.jl) just define `run_gate` and let the caller
# assert on it.
if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_gate(; cuda = USE_CUDA))
end
