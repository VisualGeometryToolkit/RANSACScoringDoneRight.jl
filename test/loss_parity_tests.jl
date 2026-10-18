# The loss potentials ARE VisualGeometryToolkitCore's.
#
# VGC owns the shape math (`ρ(shape, s, sq_loss_width)`, `s` the squared
# residual; `weight = ∂ρ/∂s`). RSDR's Track-B mixture and full-data scale
# marginals work in the depth-parameterised potential
#
#     ρ̃(s; w², Δ) = Δ · ρ(shape, s/w², 1)          (= (Δ/w²)·ρ(shape, s, w²))
#
# (w the loss width) and CALL VGC for it, on every path: the scale quadrature's
# data term (shared by the Track-B evidence device kernel and the full-data
# marginal device kernel) and the Z̃ quadrature (VGC's `truncated_log_normaliser`,
# tested there) — each the ONE
# evaluation of its quantity (the host copies are deleted). The restated copies
# (`_rho`, `_mix_rho_ka`, and the Welsch / smooth-TLS sufficient-statistic ψ
# methods) are deleted (2026-09-21).
#
# Against those copies the VGC spelling is bit-exact for Welsch and smooth-TLS
# and moves Geman-McClure by the last ulp or so (the copy rounded Δ·s/(w² + s)
# where VGC rounds w²·s/(w² + s)). Accepted.
#
# What this file pins now is the STRUCTURE: that what both device kernels
# integrate is VGC's ρ at depth Δ, by the one trapezoid rule. The CPU-vs-device
# bit-exactness of the whole evidence pipeline is the gate's
# (test/gate_marg_mix_ka.jl).

@testitem "loss_parity" begin
    using Test
    import VisualGeometryToolkitCore as VGC
    using RansacScoringDoneRight
    using VisualGeometryToolkitCore: WelschShape, GemanMcclureShape, SmoothTLSShape
    using RansacScoringDoneRight: WelschMarginal, SmoothTLSMarginal, _default_depth
    using VisualGeometryToolkitCore: IGPrior
    using LogExpFunctions: logsumexp
    const _R = RansacScoringDoneRight

    @testset "loss potentials are VGC's ρ at depth Δ" begin
        # Track-B evidence (the evidence device kernel, on CPU()): with ONE residual past
        # the model dof, E is the trapezoid-logsumexp of
        # −(Δ·ρ(shape, x/w²ⱼ, 1) + logZ̃ⱼ + baseⱼ) + ltwⱼ.
        for shape in (WelschShape(), GemanMcclureShape(), SmoothTLSShape()), Δ in (0.5, 2.0)
            spec = _R.loss_spec(shape, IGPrior(); Δ, t_lo = -8.0, t_hi = 2.0, NG = 65)
            q = spec.quad; NG = length(q.sq_loss_width_nodes)
            E = fill(-Inf, 1, 1, 2)
            be = _R.KernelAbstractions.CPU()
            _R.marg_mix_evidence_kernel!(be)(E, [0.0, 0.03], Int64[0, 2], Int32[2], Int32[1],
                q.sq_loss_width_nodes, q.logZnodes, q.base, q.ltw, zeros(NG, 1), Δ, Int32(1), Int32(NG),
                Int32(2), Int32(0), shape; ndrange = (1, 1))
            _R.KernelAbstractions.synchronize(be)
            ll = [-(Δ * VGC.ρ(shape, 0.03 / q.sq_loss_width_nodes[j], 1.0) + q.logZnodes[j] + q.base[j]) + q.ltw[j]
                  for j in eachindex(q.sq_loss_width_nodes)]
            @test E[1, 1, 2] ≈ logsumexp(ll) rtol = 1e-13
        end
    end

    @testset "full-data marginal: the trapezoid of VGC's ρ over all K points" begin
        # `marg_evidence_kernel!` integrates, at every grid point's nodes,
        #   −ψⱼ = −(Δ·Σᵢ ρ(shape, sq_rᵢ/w²ⱼ, 1) + K·logZ̃ⱼ + α₀tⱼ + β₀/w²ⱼ)
        # at the shape's canonical depth, by the trapezoid over t0 ± _MARG_L
        # (snapped to the absolute lattice t = i·h) — the mixture's quadrature
        # (above) at k = K with every point counted.
        sq_r = [0.0, 0.0, 1e-4, 4e-4, 2.5e-3, 0.01, 0.04, 0.3, 0.9, 2.0]
        n = length(sq_r); be = _R.KernelAbstractions.CPU()
        for (m, shape) in ((WelschMarginal(; alpha0 = 1.5, prior_loss_width = 0.05, outlier_halfwidth = 1.0), WelschShape()),
                       (SmoothTLSMarginal(; alpha0 = 1.5, prior_loss_width = 0.05, outlier_halfwidth = 1.0), SmoothTLSShape()))
            spec = _R.loss_spec(m; NG = 129); q = spec.quad; NG = length(q.sq_loss_width_nodes)
            @test spec.Δ == _default_depth(shape)
            h = q.tnodes[2] - q.tnodes[1]                  # t0 ± L, snapped to the lattice
            @test abs(q.tnodes[1] - (m.t0 - _R._MARG_L)) <= h / 2 + 1e-12
            @test q.tnodes[end] - q.tnodes[1] ≈ 2 * _R._MARG_L
            @test round(q.tnodes[1] / h) * h ≈ q.tnodes[1] atol = 1e-12
            @test q.base ≈ m.alpha0 .* q.tnodes .+ m.beta0 ./ q.sq_loss_width_nodes rtol = 1e-14
            S = fill(-Inf, 1, 1, 1)
            _R.marg_evidence_kernel!(be)(S, sq_r, Int64[0, n], Int32[n], Int32[1],
                q.sq_loss_width_nodes, q.logZnodes, q.base, q.ltw, Int32[0], zeros(NG, 1), spec.Δ,
                Int32(NG), Int32(NG), Int32(1), Int32(0), shape; ndrange = (1, 1))
            _R.KernelAbstractions.synchronize(be)
            ψ = [spec.Δ * sum(VGC.ρ(shape, x / sq_loss_width, 1.0) for x in sq_r) for sq_loss_width in q.sq_loss_width_nodes] .+
                n .* q.logZnodes .+ q.base
            w = fill(q.tnodes[2] - q.tnodes[1], NG); w[1] /= 2; w[end] /= 2
            @test S[1, 1, 1] ≈ logsumexp(log.(w) .- ψ) rtol = 1e-13
        end
    end
end
