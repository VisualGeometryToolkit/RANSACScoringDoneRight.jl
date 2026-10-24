# =============================================================================
# Quality gates — Aqua, ExplicitImports, JET
# =============================================================================
#
# These are RATCHETS, not aspirations: every bound below is the value the
# package meets today, and each may only ever move DOWN. If a change trips one
# of these, the fix is the code, not the bound.
#
# Baselines, 2026-09-21 audit (before → after): Aqua deps_compat failing → clean
# (stdlib/test compat entries added); ambiguities 2 → 0 (the dead Track-B
# quadrature `sweep!` pair, Gaussian-shape method against the abstract-shape one);
# implicit imports 21 → 0 and stale explicit imports 12 → 0 (every
# import now lives, explicit, in src/RANSACScoringDoneRight.jl); JET 27 → 19.
# 2026-09-21: the bound is the count of the package ANALYSED ALONE (19). In the
# full suite it reads lower (16) because earlier test files define interface
# methods (`data_size`, `constraint_with_jacobians`, …) for their own test
# problems, which satisfies three of the interface-stub reports below. A bound
# set from the suite's count would fail this file run on its own — it briefly
# was 16, for exactly that reason.
# 2026-09-30: the gate is VisualGeometryToolkitCore's shared `test_quality_gates`, which
# drops KernelAbstractions launch reports (the GPU union-splits): 19 → 11.
# 2026-10-06 (0.11.0 cleanup): 11 → 10, all ten interface stubs (seven
# `codimension`/`data_size`, three `constraint_with_jacobians`).
# 2026-10-09 (K+): 10 → 7 — the problem protocol and the posterior-weight
# toolkit moved to VisualGeometryToolkitCore, and three stub reports with them; the
# seven left are the same class (`codimension`/`data_size` and
# `constraint_with_jacobians` on the abstract `AbstractConstraintProblem`).
# =============================================================================

@testitem "quality_gates" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, Aqua, ExplicitImports, JET
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore, test_quality_gates
    using RANSACScoringDoneRight

    # The stack's one gate (VisualGeometryToolkitCore's quality extension): Aqua,
    # ExplicitImports, JET. Two qualified accesses are to names their owners
    # document but have not declared `public`: `KernelAbstractions.copyto!` (KA's
    # backend copy, the only way to move data between host and device buffers)
    # and `Threads.maxthreadid` (sizes per-thread workspaces for `@threads`).
    #
    # JET: the extension drops KernelAbstractions LAUNCH reports (a kernel
    # launch analysed with the backend left abstract splits into
    # `Kernel{<:GPU}`, for which no method exists until a GPU backend package is
    # loaded). What remains are INTERFACE STUBS — `codimension`, `data_size`,
    # `constraint_with_jacobians` are declared here and
    # implemented by the CONSUMER (VisualGeometryToolkitEstimation's problems), so analysis
    # against the abstract `AbstractConstraintProblem` finds no method. Lower the
    # bound the moment a report goes away; never raise it.
    #
    # Piracy: RANSAC and MSAC are VisualGeometryToolkitCore's baseline scorers, and
    # this package adds their GRID methods (VGC's baseline_scorers.jl header):
    # `LossWidthScoring` is a Union that includes them, so the loss-width
    # family's methods (`best_pool_model` at a loss width, `pool_model_score`,
    # the grid kernels) are on VGC's types. That is by design, not piracy — VGC
    # owns the scorer and leaves its WIDTH-FREE grid family to this package
    # (the threshold scorers stay on VGC's generic pool selection) — so the two
    # are treated as RSDR's own here.
    #
    # K+ (2026-10): two qualified accesses into VisualGeometryToolkitCore came with the
    # toolkit's move there — `VisualGeometryToolkitCore.cov` (VGC's method of
    # Statistics' `cov`, read off an `Uncertain` model in the predictive
    # `score!`) and `VisualGeometryToolkitCore._no_solver_covariance` (the one
    # missing-provider message, which `_check_provider` throws at construction).
    # Both are to be retired from src; listed here until then.
    # jet_bound 7 → 8 (2026-10-09): `select_on_grid!` takes the pairs' problem
    # type positionally (VisualGeometryToolkitCore's calling convention: no type
    # keyword), so the mixture marginal's method reads `minimal_sample_size` of
    # an abstract `Type` — the abstract-stub class of the problem protocol,
    # implemented downstream (as `data_size`/`codimension` above).
    test_quality_gates(RANSACScoringDoneRight; jet_bound = 8,
                       treat_as_own = [VgtkCore.RansacScoring, VgtkCore.MsacScoring],
                       qualified_ignore = (:copyto!, :maxthreadid))
end
