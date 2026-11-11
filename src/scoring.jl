# =============================================================================
# Scoring — the paper's two branches of scorers. The Student-t + uniform
# marginal score itself (`StudentTUniformScoring`, its sweep, label priors and
# supplied half-width) is VisualGeometryToolkitCore's (moved there 2026-11-11);
# this package keeps its loss-cost branch, its outlier models that estimate the
# support (extreme_value.jl) and its grids.
# =============================================================================

# =============================================================================
# THE TWO BRANCHES
# =============================================================================
#
# What separates them is whether a GENERATIVE MODEL sits underneath, and that
# decides which questions a member can answer — not merely how it computes. So
# it is a hierarchy, not a trait.
#
#   GauScoring     the Gaussian-Uniform mixture of Eq. (1): a Gaussian inlier
#                  density and a uniform outlier density on [-h, h]^d_g (h the
#                  outlier half-width), with a
#                  latent label z_i and prior gamma = Pr(z_i = 1). EVERY member
#                  posits this one model; they differ only in how the two
#                  nuisances are eliminated — see the elimination table below.
#                  RANSAC and MSAC are not outside the family: they profile the
#                  labels where MLESAC marginalises them.
#
#   LossCostScoring a robust loss rho from VisualGeometryToolkitCore, with NO mixture.
#                  The half-quadratic lift does give rho(s) = min_z[z*s + psi(z)],
#                  but psi is a PENALTY, not a normalised log-prior: z is not a
#                  posterior, there is no gamma and no f_in. These cannot answer
#                  a probability question at all, which is why they are a branch
#                  rather than a trait value on one.
#
# The loss width each branch runs at comes from a DIFFERENT THEORY — MAP for
# the mixture, asymptotic efficiency for the loss — which is the substantive
# reason not to give them a common supertype that implies a mixture.
#
# The TYPES follow what the scorers compute, because Julia has one supertype:
# every family at a loss width is a VisualGeometryToolkitCore
# `AbstractThresholdScoring{T}` (`T` its width's type), whose one loop scores
# it. So `LossCostScoring{T} <: AbstractThresholdScoring{T}`, and MLESAC and
# MAGSAC++ subtype `AbstractThresholdScoring` directly; `GauScoring` holds the
# one mixture scorer with no loss width, `StudentTUniformScoring` — now
# VisualGeometryToolkitCore's, a plain `AbstractScoring` (the empty `GauScoring`
# supertype went with the move: nothing dispatched on it, 2026-11-07: this
# package, LODR, BlobBoards, the neurips2026 code). The branch a family belongs
# to is the table above, not its supertype.

abstract type LossCostScoring{T} <: AbstractThresholdScoring{T} end

# The Gaussian-uniform mixture's boundary, both ways — `decision_boundary`
# (the residual at which the MAP rule flips) and `decision_width(::GauInlier, …)`
# (the outlier half-width that puts it there) — is VisualGeometryToolkitCore's, with
# `GauInlier`. `outlier_halfwidth_to_loss_width`/`loss_width_to_outlier_halfwidth`
# (welsch_marginal.jl) are a DIFFERENT question — what one family's parameter is
# worth in the other's units at a fixed operating scale multiple.

# -----------------------------------------------------------------------------
# How each branch eliminates its nuisances — a TABLE, not a trait family.
# -----------------------------------------------------------------------------
#
# The labels `z` are eliminated by profiling (a max: MSAC, and RANSAC keeping
# only the count) or by marginalising (a sum: MLESAC); the scale `sigma` is
# either supplied as a loss width `scale_multiple*sigma` or integrated out under the conjugate
# Inverse-Gamma prior. The paper's contribution is the cell nothing else
# occupies:
#
#                 scale supplied            scale marginalised
#   profile z     RANSAC, MSAC              StudentTUniformScoring  <-- here
#   marginal z    MLESAC, MAGSAC++          intractable (2^N partitions)
#
# MAGSAC++ sits under "scale supplied": its uniform-prior "marginalisation" over
# `sigma ~ U[0, sigma_bar]` is numerically MLESAC at fixed `sigma_bar`, and the
# parameter the user sets is the loss width `kappa*sigma_bar` — a proxy hyperparameter in
# disguise. What the score IS, not what its derivation claimed.
#
# This was briefly a `LabelRule`/`ScaleRule` trait family with an
# `is_threshold_based` predicate over it. Deleted: nothing dispatched on any of
# it, in this package or in VisualGeometryToolkitCore, VisualGeometryToolkitEstimation,
# VisualGeometryToolkitEval or BlobBoards — and traits are added only when a
# dispatch site consumes them. Its `FixedScale` also shadowed VGC's exported
# `FixedScale` (a `ScaleInterface` that RVG really does dispatch on), so one
# name meant two unrelated things across a single dependency chain. The claim
# the traits encoded is the table above, which costs nothing and cannot drift
# out of step with a dispatch that does not exist.

# `loss_shape(::LossCostScoring)` -> the VisualGeometryToolkitCore shape it scores by.
# Methods go on VGC's OWN exported `loss_shape` (imported in the entry point)
# rather than on a fresh generic: VGC already asks this exact question of a
# `DeployedLoss`/`AnnealedLoss`/`AbstractLossShape`, and a scoring is just one
# more thing that has a shape. Declaring a second `loss_shape` here — and
# exporting it — put two unrelated bindings of one name in a single dependency
# chain, which becomes an ambiguity for any module that `using`s both.

