module RANSACScoringDoneRight

import VisualGeometryToolkitCore as VgtkCore

# =============================================================================
# RANSACScoringDoneRight — scale-free marginal RANSAC scoring
# =============================================================================
#
# The paper's research scorers on VisualGeometryToolkitCore's contracts. The
# sort-and-sweep Student-t + uniform marginal score itself (Algorithm 1,
# `StudentTUniformScoring`: its sweep and the one prefix kernel, its workspace,
# label priors, the supplied half-width) is VisualGeometryToolkitCore's (moved
# there 2026-11-11, the owner's ruling); this package extends it by dispatch —
# the outlier models that estimate the support, its CPU/GPU grids — and adds the
# paper baselines. The scoring core
# depends on VisualGeometryToolkitCore (for Uncertain and the constraint-problem protocol)
# and StaticArrays/SpecialFunctions; the GPU sweep port adds KernelAbstractions,
# and the scoring VOCABULARY layer adds Statistics (`cov` of an `Uncertain`).
# SIMFIT++ is VisualGeometryToolkitEstimation's (`SimFitPP`, `simfitpp_select`;
# moved there on 2026-11-11): this package only gives it a grid on the
# pool-selection contract (selection.jl). The bootstrap harness that once lived here —
# and the JLD2/HDF5/DrWatson/DataFrames/Bootstrap/Optim deps it carried — moved
# to the evaluation engine on 2026-07-18 (now VisualGeometryToolkitEval) and is NOT a dependency of this
# package. Problems implement VisualGeometryToolkitCore's `AbstractConstraintProblem`.
#
# VOCABULARY. VisualGeometryToolkitCore's robust layer is authoritative for the loss
# words this package shares with it: the shape math ρ/ψ/weight,
# `efficiency_constant` the dimensionless multiple of the scale a width sits at,
# `s2` the noise VARIANCE, `loss_width` a loss's scale parameter and
# `sq_loss_width` its square (`efficiency_constant^2 * s2` for a deployed loss).
# Note VGC's `ρ(shape, s, sq_loss_width)` spells the squared RESIDUAL `s`; this
# package spells it `sq_r`/`sq_distances` and never `s`. On top of that RSDR
# owns one layer of its own: `outlier_halfwidth` the half-width of the uniform
# outlier support, `decision_boundary` the residual at which the MAP rule
# flips (`decision_boundary`/`decision_width` are the ONE map between them — a
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
using SpecialFunctions: gamma, gamma_inc
using LogExpFunctions: log1pexp   # log(1 + eˣ) (MLESAC's quality)
using Statistics: Statistics

# GPU scoring (KernelAbstractions port — CPU + CUDA backends).  The loss-width
# baseline and IG/marginal sweep kernels in ig_sweep_ka.jl / loss_width_sweep_ka.jl use these;
# the module name is bound because nearly every call is qualified.
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
# The one host → device upload (allocate + copy, converted to the element type)
# and the eager release of the grid evaluators' function-local device buffers
# (GPUArrays' `unsafe_free!`; this package's `_free_local!` copy is deleted,
# 2026-11-12) are VisualGeometryToolkitCore's.
using VisualGeometryToolkitCore: ka_upload, free_device_buffers!
using VisualGeometryToolkitCore: OnBackend   # the one kernel-launch policy: every grid kernel is instantiated through it

# This package computes no residual of its own: the whitened residuals and the
# predictive score are VisualGeometryToolkitCore's (with its StudentTUniformScoring).
using VisualGeometryToolkitCore: VisualGeometryToolkitCore,
    # Geometry the evaluation subsystem scores against (scoring_eval_common.jl):
    # the loss shapes the inlier-loss layer and the scale marginals dispatch on
    AbstractLossShape, WelschShape, GemanMcclureShape, SmoothTLSShape
# The benchmark pool record and its accessors are VisualGeometryToolkitCore's (its
# `Pools` module): RSDR's pool scorers take the light `PoolPair` together with
# its residual matrix, and read the record only through these. `pixel_residuals`
# is the pool's one exit to pixels (the pair's image-plane Sampson residuals).
using VisualGeometryToolkitCore.Pools: PoolPair, num_points, num_models, pool_imc, pixel_residuals,
    # the record's residual-matrix contract and the one
    # pixel → pool-frame scale
    check_residual_matrix, pool_pixel_scale
# SIMFIT++ is VisualGeometryToolkitEstimation's: its configuration and its pool
# adapter. This package gives `SimFitPP` a width axis, a tag and a grid on the
# pool-selection contract (native_widths.jl, selection.jl) and re-exports
# nothing of it: callers take both names from VisualGeometryToolkitEstimation.
using VisualGeometryToolkitEstimation: SimFitPP, simfitpp_select
import VisualGeometryToolkitEstimation as VgtkEstimation   # its scorer TOML reader and writer (`scoring_from_toml`, `config_dict`), extended for this package's scorers
# The scoring families' tag registry, merged with this package's (native_widths.jl).
using VisualGeometryToolkitCore: TagRegistry
# The stack's defaults mechanism (config_defaults.jl): configs/default.toml, parsed once.
using VisualGeometryToolkitCore: parse_defaults, toml_default
# The width axes, their native maps and the native-axis conversions are
# VisualGeometryToolkitCore's; this package states the axis of its grid families.
using VisualGeometryToolkitCore: LossWidth, OutlierHalfwidth
# The contracts this package implements are VisualGeometryToolkitCore's: the
# constraint-problem protocol, the scoring contract, the pool-selection contract
# and the robust-statistics toolkit. RSDR implements the scoring contract for
# its own scorers and adds its grid algorithms and paper baselines; it
# re-exports nothing VGC owns.
using VisualGeometryToolkitCore: AbstractConstraintProblem,
    IGPrior,
    truncated_log_normaliser,
    RansacScoring, MsacScoring, AbstractThresholdScoring, @threshold_scoring_constructors,
    argmax_models_kernel!, sorted_squared_residuals   # the grids' model argmax and staging (Core's)
# The Student-t + uniform marginal scorer and its partition sweep (Core's): this package's grids,
# outlier models and paper baselines use and extend them.
using VisualGeometryToolkitCore: StudentTUniformScoring, AbstractLabelPrior, FlatLabelPrior,
    AbstractOutlierModel, FixedHalfwidth, ig_sweep_kernel!, log_gamma_table,
    suffix_sums!
import VisualGeometryToolkitCore: loss_shape, codimension, score_bound,
    decision_boundary, has_decision_boundary,
    scale_prior, outlier_halfwidth, width_axis, minimal_sample_size,
    residual_quality,
    # the estimated-support sweep's hooks, for this package's outlier models (extreme_value_sweep.jl)
    outlier_context!, outlier_cost, scored_outlier_halfwidth,
    # the pool-selection contract (VisualGeometryToolkitCore robust/pool_selection.jl):
    # the staged grid selection (a scorer at its width selects through
    # VisualGeometryToolkitCore's `best_pool_model`, by `score!`)
    selection_staging, stage_selection, select_on_grid!

# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

# The pool record's per-pair staging precompute.
public precompute_pair

# The outlier models that estimate the support of VisualGeometryToolkitCore's
# StudentTUniformScoring (extreme_value.jl; the scorer, its label priors and the
# supplied half-width `FixedHalfwidth` are Core's).
export TailRules, ExtremeValueHalfwidth, GeneralizedParetoOutliers, OutlierTerm
public gpd_pwm_fit, tail_fit, gpd_support, gpd_outlier_cost

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
# `loss_width_scoring(tag[, dg])` builds a `LossWidthScoring` member by its tag
# of `ALL_SCORING_TAGS`, the scoring families' registry (VisualGeometryToolkitCore's merged
# with this package's).
export loss_width_scoring
public ALL_SCORING_TAGS

# A loss-width family at its width selects through VisualGeometryToolkitCore's
# `best_pool_model` (its `score!`, scoring_eval_common.jl); the staged grid
# selection of every grid family is selection.jl's (VisualGeometryToolkitCore's
# `select_on_grid!`).
export WelschMarginal
public pair_chunk_budget
# The loss-width grid's log-binned aggregation (its exact selection is
# VisualGeometryToolkitCore's `select_on_grid!` on the `:pool` staging)
public LogBinned

# Track B — inlier/outlier mixture reformulation of redescending losses
public loss_scale_bounds, LossSpec, loss_spec

# -----------------------------------------------------------------------------
# Include order: the outlier models → their hooks on Core's sweep
# -----------------------------------------------------------------------------

include("config_defaults.jl")         # configs/default.toml → `_DEFAULTS_OF_RECORD`, `_config_default` (first: every default reads it)
include("extreme_value.jl")           # the outlier models that estimate the support, the generalized Pareto fit
include("scoring.jl")                 # the loss-cost branch's supertype (LossCostScoring)
include("extreme_value_sweep.jl")     # their methods of Core's estimated-support sweep hooks

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

include("ig_sweep_ka.jl")             # IGSharedData, grid_evaluate(::StudentTUniformScoring) on Core's ig_sweep_kernel!, staging
include("loss_width_sweep_ka.jl")     # RANSAC/MSAC/GaU/MAGSAC++ device kernels + grid_evaluate(::LossWidthScoring)

# -----------------------------------------------------------------------------
# Evaluation subsystem (extracted from VisualGeometryToolkitEstimation/eval)
# -----------------------------------------------------------------------------
# The pool scorers over VisualGeometryToolkitCore's light pool record + its residual
# matrix. Concrete problem types (HomographyProblem/…) stay in the consumer and
# are given as the leading `problem_type` argument where one is needed.
include("scoring_eval_common.jl")     # per-pair scorers, pool scorers, per-pair staging precompute
include("inlier_potentials.jl")       # canonical depths of the loss shapes; ρ̃ = (Δ/sq_loss_width)·VGC's ρ
include("welsch_marginal.jl")         # ScaleMarginal{K} + shared evidence machinery; Welsch methods
include("smoothtls_marginal.jl")      # marginal smooth-TLS spec: SmoothTLSMarginal
include("inlier_losses.jl")           # Track B: redescending losses as inlier/outlier mixtures
include("marg_sweep_ka.jl")           # KA marginal kernels: grid_evaluate(::ScaleMarginal), grid_evaluate(::LossSpec)
include("native_widths.jl")           # width_axis of the grid families
include("selection.jl")               # the grid families on the pool-selection contract

end # module RANSACScoringDoneRight
