module RANSACScoringDoneRight

import VisualGeometryToolkitCore as VgtkCore

# =============================================================================
# RANSACScoringDoneRight — scale-free marginal RANSAC scoring
# =============================================================================
#
# The paper scorers on VisualGeometryToolkitCore's scoring contract: the sort-and-sweep
# Student-t + uniform marginal score (Algorithm 1, `StudentTUniformScoring`), its
# CPU/GPU grids, and the paper baselines. The scoring core
# depends on VisualGeometryToolkitCore (for Uncertain/whiten/constraint_with_jacobians)
# and StaticArrays/SpecialFunctions; the GPU sweep port adds KernelAbstractions
# (and GPUArrays, to free its device scratch eagerly),
# and the scoring/tuning VOCABULARY layer (with SIMFIT++) adds Distributions,
# Random and Statistics. The bootstrap harness that once lived here —
# and the JLD2/HDF5/DrWatson/DataFrames/Bootstrap/Optim deps it carried — moved
# to the evaluation engine on 2026-07-18 (now VisualGeometryToolkitEval) and is NOT a dependency of this
# package. Problems implement VisualGeometryToolkitCore's `AbstractConstraintProblem`.
#
# VOCABULARY. VisualGeometryToolkitCore's robust layer is authoritative for the loss
# words this package shares with it: the shape math ρ/ψ/weight,
# `scale_multiple` the dimensionless multiple of the scale a width sits at,
# `s2` the noise VARIANCE, `loss_width` a loss's scale parameter and
# `sq_loss_width` its square (`scale_multiple^2 * s2` for a deployed loss).
# Note VGC's `ρ(shape, s, sq_loss_width)` spells the squared RESIDUAL `s`; this
# package spells it `sq_r`/`sq_residuals` and never `s`. On top of that RSDR
# owns one layer of its own: `outlier_halfwidth` the half-width of the uniform
# outlier support, `decision_threshold` the residual at which the MAP rule
# flips (`decision_threshold`/`decision_width` are the ONE map between them — a
# grid of outlier half-widths is never called a grid of thresholds),
# `ŝ`/`s_hat_sq` the scale estimate and its square, `d_g`/`dg` the codimension,
# `inlier_prior` the scalar mixing proportion π and `inlier_posterior` the
# per-point P(inlier|r).
# =============================================================================

# ALL imports live here, every one EXPLICIT (gated by test/quality_gates_tests.jl).
# Here and not beside the file that uses a name: imports are module-wide, but
# a name used in a method SIGNATURE must be bound before the file that declares
# it is included, so per-file import lines silently depended on include order.
# A bare `using VisualGeometryToolkitCore` once sat in a pool-record file and pulled
# VGC's whole export list into this namespace — its `m` and `mm` are the metre
# and millimetre UNITS, and several local variables here share those names.
using LinearAlgebra: I, det, dot
using StaticArrays: SMatrix, SVector, StaticMatrix, StaticVector
using SpecialFunctions: loggamma, gamma, gamma_inc
using LogExpFunctions: log1pexp # log(1 + eˣ) (MLESAC's quality)
using Distributions: Chisq, cdf, quantile
using Random: AbstractRNG, default_rng, randperm
using Statistics: Statistics, median

# GPU scoring (KernelAbstractions port — CPU + CUDA backends).  The loss-width
# baseline and IG/marginal sweep kernels in ig_sweep_ka.jl / loss_width_sweep_ka.jl use these;
# the module name is bound because nearly every call is qualified.
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
# Eager release of the grid evaluators' function-local device buffers
# (`_free_local!`, marg_sweep_ka.jl): GPUArrays owns `unsafe_free!`, which every
# GPU array type implements, and `AbstractGPUArray`. Bound by module name,
# because both uses are qualified. (KernelAbstractions' own `unsafe_free!` is
# not public, and CUDA.jl 5.11 does not extend it: on CUDA it frees nothing.)
using GPUArrays: GPUArrays

# `whiten` is the ONE whitening (and degeneracy rule) of the residual path; the
# restated `_mahalanobis` that needed `propagate`/`_propagate_scalar` beside it
# now goes through it, so neither — the second a VGC internal — is imported.
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, Uncertain, CovMat, whiten, geom
# Geometry the evaluation subsystem scores against (scoring_eval_common.jl,
# simfitpp.jl).
using VisualGeometryToolkitCore:
    # the loss shapes the inlier-loss layer and the scale marginals dispatch on
    AbstractLossShape, WelschShape, GemanMcclureShape, SmoothTLSShape,
    sampson_distance, sources, targets,
    # a pool model's frame change is composition with the normalization's two
    # sides (`pixel_residuals`)
    source_map, destination_map
# The benchmark pool record and its accessors are VisualGeometryToolkitCore's (its
# `Pools` module): RSDR's pool scorers take the light `PoolPair` together with
# its residual matrix, and read the record only through these.
using VisualGeometryToolkitCore.Pools: PoolPair, num_points, num_models, pool_imc, pool_normalization,
    pixel_csponds
# The width axes, their native maps and the native-axis conversions are
# VisualGeometryToolkitCore's; this package states the axis of its grid families.
using VisualGeometryToolkitCore: LossWidth, OutlierHalfwidth
using VisualGeometryToolkitCore: constraint_with_jacobians
# The contracts this package implements are VisualGeometryToolkitCore's: the
# constraint-problem protocol, the scoring contract, and the robust-statistics
# toolkit (scale, the Student-t + uniform mixture, the posterior-weight state).
# RSDR implements the scoring contract for its own scorers and adds its grid
# algorithms and paper baselines; it re-exports nothing VGC owns.
using VisualGeometryToolkitCore: AbstractConstraintProblem, data_size,
    solver_covariance, provides_solver_covariance, sq_norm, whitened_constraint_residual,
    whitened_distance, IGPrior, JeffreysPrior,
    square_residuals!, zero_sample!, GauInlier, decision_threshold,
    truncated_log_normaliser, consensus_s2, ScoreBuffers, residuals,
    scoring_workspace, RansacScoring, MsacScoring
import VisualGeometryToolkitCore: AbstractScoring, loss_shape, codimension, score!,
    inlier_count, consensus_order, inlier_threshold, has_inlier_threshold, is_predictive,
    scored_model, scale_prior, outlier_halfwidth, check_workspace, with_width, width_axis, minimal_sample_size,
    residual_quality, scoring_workspace,
    # the pool-selection contract (VisualGeometryToolkitCore robust/pool_selection.jl):
    # the grid-family form of best_pool_model and the staged grid selection
    best_pool_model, selection_staging, stage_selection, select_on_grid!

# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

# The sweep and its workspace.
export ScoringWorkspace
public precompute_pair

# The Student-t + uniform scorer (scoring.jl): model-certain, or Predictive.
export StudentTUniformScoring, Predictive
export score_residuals!, score_sq_residuals!

# GPU scoring (KernelAbstractions): flat-layout staging + THE grid verb.
# `grid_evaluate` dispatches on the scoring: the loss-width families
# (RANSAC/MSAC/MLESAC/MAGSAC++ and the losses), the Student-t + uniform marginal,
# the full-data scale marginals and the mixture marginal of a redescending loss.
export IGSharedData, stage_ig_shared_from_pairs
export grid_evaluate
# The scorer fact a grid reads, the width axis its grid is native on, is
# VisualGeometryToolkitCore's `width_axis(scoring)` (methods in native_widths.jl).
# The paper baselines (RANSAC and MSAC are VisualGeometryToolkitCore's baseline
# scorers; this package adds their grid methods). "Takes a loss width" cuts
# across the mixture and the loss-cost branches, so it is the `LossWidthScoring`
# union, not a supertype.
export LossWidthScoring
export MlesacScoring, MagsacScoring
export WelschCostScoring, SmoothTLSScoring, GemanScoring
# `loss_width_scoring(tag[, dg])` builds a `LossWidthScoring` member by its tag.
export loss_width_scoring

# Per-pair model selection over a width-free loss-width family
# (scoring_eval_common.jl; a scorer AT its width selects through
# VisualGeometryToolkitCore's `best_pool_model`), and the staged grid selection of
# every grid family (selection.jl, VisualGeometryToolkitCore's `select_on_grid!`).
export WelschMarginal,
    pool_model_score   # one stored model's loss-width-family score (the per-model term of best_pool_model)
public pair_chunk_budget
# SIMFIT++ (simfitpp.jl): loss-width estimation from held-out residuals + the pool adapter
export SimFitPP, unbiased_median_variance, simfitpp_loss_width, simfitpp_select, pixel_residuals,
    residual_unit_scale
# The SIMFIT++ bounds in another residual unit (`simfitpp_select`'s matrix method).
public with_residual_unit
# The two aggregations of the loss-width grid (public; generic names)
public Aggregation, Exact, LogBinned

# Track B — inlier/outlier mixture reformulation of redescending losses
public loss_scale_bounds, LossSpec, loss_spec

# Accessors of the sweep workspace (public, not exported: generic nouns).
public sweep_order, penalties, sort_key, lg_table
public sweep!, default_scoring_workspace, alloc_scoring_workspace

# -----------------------------------------------------------------------------
# Include order: sweep → scoring
# -----------------------------------------------------------------------------

include("sweep.jl")                   # ScoringWorkspace, sweep!
include("scoring.jl")                 # StudentTUniformScoring + its scoring contract + score_residuals!/score_sq_residuals!

# -----------------------------------------------------------------------------
# GPU scoring (KernelAbstractions port)
# -----------------------------------------------------------------------------
# MAGSAC++ constants used by `residual_quality(::MagsacScoring, …)` in
# loss_width_sweep_ka.jl.  DOF = 4 (homography/relative-pose constraint
# codimension); the scale multiple sqrt(chi2_{0.99}(4)) is the MAGSAC++ inlier
# cutoff, and this is THE one copy of it (the lookup-table scorer, whose per-DOF
# table read its dof == 4 entry from here, is deleted).  The value is
# superansac's `getK`; the exact quantile is 3.6437212, so it carries a 4.5e-4
# residual — recorded here rather than rediscovered.  MAGSAC++ parametrises its
# maximal scale as `σ̄ = loss_width / _MAGSAC_SCALE_MULTIPLE` (Shekhovtsov 2025
# §5.2.4 Fig 5 line 881; Prop 6).
const _MAGSAC_DOF = 4
const _MAGSAC_SCALE_MULTIPLE = 3.644173432

include("ig_sweep_ka.jl")             # IGSharedData, ig_sweep_kernel!, grid_evaluate(::StudentTUniformScoring), staging
include("loss_width_sweep_ka.jl")     # RANSAC/MSAC/GaU/MAGSAC++ device kernels + grid_evaluate(::LossWidthScoring)

# -----------------------------------------------------------------------------
# Evaluation subsystem (extracted from VisualGeometryToolkitEstimation/eval)
# -----------------------------------------------------------------------------
# The pool scorers over VisualGeometryToolkitCore's light pool record + its residual
# matrix. Concrete problem types (HomographyProblem/…) stay in the consumer and
# are given as the leading `problem_type` argument where one is needed.
include("scoring_eval_common.jl")     # per-pair scorers, pool scorers, per-pair staging precompute
include("simfitpp.jl")                # SIMFIT++ loss-width estimation (Edstedt 2025) + pool adapter
include("inlier_potentials.jl")       # canonical depths of the loss shapes; ρ̃ = (Δ/sq_loss_width)·VGC's ρ
include("welsch_marginal.jl")         # ScaleMarginal{K} + shared evidence machinery; Welsch methods
include("smoothtls_marginal.jl")      # marginal smooth-TLS spec: SmoothTLSMarginal
include("inlier_losses.jl")           # Track B: redescending losses as inlier/outlier mixtures
include("marg_sweep_ka.jl")           # KA marginal kernels: grid_evaluate(::ScaleMarginal), grid_evaluate(::LossSpec)
include("native_widths.jl")           # width_axis of the grid families
include("selection.jl")               # the grid families on the pool-selection contract

end # module RANSACScoringDoneRight
