# =============================================================================
# Scoring — the Student-t + uniform marginal score on VisualGeometryCore's
# scoring contract
# =============================================================================
#
# Contents:
#   1. the two branches of the paper's scorers (GauScoring, LossCostScoring)
#   2. StudentTUniformScoring{P,DG} (P=Nothing|Predictive, DG=codimension)
#   3. its scoring contract (scoring_workspace, score!, inlier_count,
#      consensus_order, inlier_threshold, …)
#   4. buffer-based score_residuals!/score_sq_residuals! + sweep wrappers
#
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
#   LossCostScoring a robust loss rho from VisualGeometryCore, with NO mixture.
#                  The half-quadratic lift does give rho(s) = min_z[z*s + psi(z)],
#                  but psi is a PENALTY, not a normalised log-prior: z is not a
#                  posterior, there is no gamma and no f_in. These cannot answer
#                  a probability question at all, which is why they are a branch
#                  rather than a trait value on one.
#
# The loss width each branch runs at comes from a DIFFERENT THEORY — MAP for
# the mixture, asymptotic efficiency for the loss — which is the substantive
# reason not to give them a common supertype that implies a mixture.

abstract type GauScoring <: AbstractScoring end
abstract type LossCostScoring <: AbstractScoring end

# The Gaussian-uniform mixture's boundary, both ways — `decision_threshold`
# (the residual at which the MAP rule flips) and `decision_width(::GauInlier, …)`
# (the outlier half-width that puts it there) — is VisualGeometryCore's, with
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
# it, in this package or in VisualGeometryCore, RobustVisualGeometry,
# VisualGeometryEval or BlobBoards — and traits are added only when a
# dispatch site consumes them. Its `FixedScale` also shadowed VGC's exported
# `FixedScale` (a `ScaleInterface` that RVG really does dispatch on), so one
# name meant two unrelated things across a single dependency chain. The claim
# the traits encoded is the table above, which costs nothing and cannot drift
# out of step with a dispatch that does not exist.

# `loss_shape(::LossCostScoring)` -> the VisualGeometryCore shape it scores by.
# Methods go on VGC's OWN exported `loss_shape` (imported in the entry point)
# rather than on a fresh generic: VGC already asks this exact question of a
# `DeployedLoss`/`AnnealedLoss`/`AbstractLossShape`, and a scoring is just one
# more thing that has a shape. Declaring a second `loss_shape` here — and
# exporting it — put two unrelated bindings of one name in a single dependency
# chain, which becomes an ambiguity for any module that `using`s both.

# AbstractScoring now lives in VisualGeometryCore (src/robust/estimation.jl) —
# imported in RansacScoringDoneRight.jl and re-exported. Scoring functions
# determine how candidate models are scored. The scale-free marginal score
# (Eq. 12) eliminates σ by marginalizing under the Jeffreys prior π(σ²) ∝ 1/σ²,
# then optimizes over partitions I. Higher = better. Built-in strategies:
# - `StudentTUniformScoring`: Model-certain score S(θ,I) (Section 3, Eq. 12)
# - `StudentTUniformScoring{Predictive}`: Predictive score S_pred(θ̂,I) incorporating
#   per-point leverages (Section 4, Eq. 20)

# Scale prior `IGPrior(α₀, β₀)` defined in estimators/sweep.jl. Jeffreys =
# IGPrior(0, 0); the two were previously distinct types and `_alpha0`
# dispatched on each — now a single field read.
_alpha0(p::IGPrior) = p.alpha0

# NOTE: the local-optimization strategy types (AbstractLocalOptimization,
# NoLocalOptimization, PosteriorIrls, PoseLibLm) and `default_local_optimization`
# live in RobustVisualGeometry — they are RANSAC-loop concerns, not scoring.
# Scoring only produces (S, k) from residuals; the loop decides how to refine.

# The Student-t + uniform mixture's toolkit (StudentTMixture, the posterior
# weights of a partition, PosteriorWeightState) is VisualGeometryCore's.

# -----------------------------------------------------------------------------
# Uncertain{M} Dispatch (Section 3.3: Specializations)
# -----------------------------------------------------------------------------
#
# Model uncertainty dispatch (type dispatch on model argument):
#   model::M             → model-certain (Σ_θ = 0, treat θ as exact)
#   model::Uncertain{M}  → model-uncertain (Σ_θ = cov(model))
#
# The model carries its own covariance via VGC's Uncertain{V,T,N} wrapper.
# `scored_model` (the scoring contract) wraps after a minimal solve; the
# model-certain scorer never wraps → zero overhead.
#
# All residuals!() implementations return whitened residuals such that
# ‖rᵢ‖² = Mahalanobis distance. The two score! methods:
#
#   model::M (certain)     → scores[i] = rᵢ²,         penalties[i] = 0
#   model::Uncertain{M}   → scores[i] = rᵢᵀ Ṽᵢ⁻¹ rᵢ, penalties[i] = log|Ṽᵢ|
#
# Measurement-space ℓᵢ = 0: the outlier density in whitened constraint space
# rᵢ = Lᵢ⁻¹gᵢ is (2·outlier_halfwidth)^{-dg}, constant and θ-independent. The inlier
# Gaussian normalization |Σ̃_{gᵢ}|^{-1/2} cancels with the whitening
# Jacobian, so no measurement-space log-det correction is needed.
#
# Model-uncertainty ℓᵢ = log|Ṽᵢ|: for Uncertain{M}, the predictive variance
# inflation Ṽᵢ = I + G_{θ,w} Σ̃_θ G_{θ,w}ᵀ ≥ I deflates the per-point score
# qᵢ = rᵢᵀ Ṽᵢ⁻¹ rᵢ ≤ ‖rᵢ‖². Without the penalty, an uncertain model
# always scores at least as well as a certain one. The penalty is accumulated
# in sweep! as −½ Σᵢ∈I ℓᵢ to prevent this.
# -----------------------------------------------------------------------------
#
# (Removed: MeasurementCovariance trait — all residuals! return whitened r, so
#  the Isotropic/Heteroscedastic axis was redundant. See git log for history.)

# -----------------------------------------------------------------------------
# Concrete Scoring Strategies
# -----------------------------------------------------------------------------

"""
    Predictive

Type tag for the predictive variant of `StudentTUniformScoring`.

`StudentTUniformScoring{Predictive}` incorporates model estimation uncertainty
(Σ̃_θ from `solver_covariance`, of a minimal-sample model or of a refit on its
consensus set) into per-point scores.
`StudentTUniformScoring{Nothing}` treats the model as exact (Σ_θ = 0).

See also: [`StudentTUniformScoring`](@ref), [`StudentTUniformScoring{Predictive}`](@ref)
"""
struct Predictive end

"""
    StudentTUniformScoring{P, DG} <: GauScoring

The scale-free marginal score of the Student-t inlier + uniform outlier model
(Section 3, Eq. 12) — **spec only**: the immutable configuration of the score
(outlier half-width, model DOF, scale prior, profile-prior flag). Its scratch
(perm, lg_table, penalties, sort_key, prior_table) lives on its
[`ScoringWorkspace`](@ref), the scoring contract's `scoring_workspace`.

Parameterized by:
- `P`: `Nothing` (model-certain) or `Predictive` (model-uncertain)
- `DG::Int`: Codimension dg of the model manifold (compile-time constant)

The score marginalizes σ² under the scale prior (Jeffreys `π(σ²) ∝ 1/σ²` by
default):

    S(θ, I) = log Γ((nᵢ·dg − n_θ)/2) − ((nᵢ·dg − n_θ)/2) log(π·RSS_I) − nₒ·dg·log(2h)

where nᵢ = |I|, RSS_I = Σᵢ∈I rᵢ², nₒ = N − nᵢ, n_θ = m·dg (model parameters),
h is the outlier half-width, and dg the codimension.

Per-point scores ‖rᵢ‖² = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ, where rᵢ = Lᵢ⁻¹gᵢ is the
shape-whitened residual. For the predictive variant, Σ̃_{gᵢ} includes
model estimation uncertainty.

Build it with `StudentTUniformScoring(problem | P | model_dof, outlier_halfwidth;
prior = JeffreysPrior(), profile_prior = false)` (the `model_dof` form takes
`codimension`), or `StudentTUniformScoring{Predictive}(problem | P, outlier_halfwidth; …)`.

# Fields
- `outlier_halfwidth::Float64`: the half-width of the uniform outlier support
  (the sweep charges `log(2·outlier_halfwidth)` per point and axis)
- `model_dof::Int`: Minimal sample size m = ⌈n_θ/dg⌉
- `profile_prior::Bool`: Whether the profiled-γ correction is used (the
  workspace's `prior_table`): the profiled label prior γ̂ = nᵢ/N instead of
  the flat γ = ½.
- `scale_prior::IGPrior`: Scale prior for σ² marginalization. Jeffreys
  is `IGPrior(0.0, 0.0)`; weakly-informative IG is `IGPrior(α₀, β₀)`.
"""
struct StudentTUniformScoring{P, DG} <: GauScoring
    outlier_halfwidth::Float64
    model_dof::Int
    profile_prior::Bool
    scale_prior::IGPrior
    # TYPED field constructor, replacing the default one that `convert`s every
    # argument: a stale positional call would otherwise convert silently.
    StudentTUniformScoring{P, DG}(outlier_halfwidth::Float64, model_dof::Int, profile_prior::Bool,
                                  scale_prior::IGPrior) where {P, DG} =
        new{P, DG}(outlier_halfwidth, model_dof, profile_prior, scale_prior)
end

codimension(::StudentTUniformScoring{P, DG}) where {P, DG} = DG

# Accessors — downstream reads the spec through these, never by field name.
"""
    outlier_halfwidth(scoring::StudentTUniformScoring) -> Float64

The half-width of the uniform outlier support per axis. The sweep charges
`log(2·outlier_halfwidth)` per point and axis.
"""
outlier_halfwidth(s::StudentTUniformScoring) = s.outlier_halfwidth

# The model DOF `m` of `ν = (k − m)·d_g` — the problem's minimal sample size,
# which is what every problem-aware constructor fills the field from.
minimal_sample_size(s::StudentTUniformScoring) = s.model_dof

scale_prior(s::StudentTUniformScoring) = s.scale_prior

"""
    Predictive scoring: `StudentTUniformScoring{Predictive}`

Model-uncertain scoring that incorporates estimation uncertainty Σ̃_θ (Section
4, Eq. 20). An OPTION, not a default: measured against model-certain scoring
it ties or loses on pooled ΔAUC@10° and costs 13–565× per hypothesis
(RobustVisualGeometry's docs/src/guide/scoring.md has the tables).
"""
const _PredictiveStudentTUniform = StudentTUniformScoring{Predictive}

"""
    _build_prior_table(n)

Precompute profiled-γ correction for k = 1..N:

    prior_table[k] = k·log(k/N) + (N−k)·log((N−k)/N)

This is the log of the profiled label prior γ̂^{nᵢ}(1−γ̂)^{nₒ} with
γ̂ = k/N (MLE of the inlier ratio for a given partition size k).
Equivalent to −N·H(k/N) where H is binary entropy.

Convention: 0·log(0) = 0, so prior_table[N] = 0.
"""
function _build_prior_table(n::Int)
    pt = Vector{Float64}(undef, n)
    n_f = Float64(n)
    @inbounds for k in 1:n-1
        k_f = Float64(k)
        pt[k] = k_f * log(k_f / n_f) + (n_f - k_f) * log((n_f - k_f) / n_f)
    end
    if n >= 1
        pt[n] = 0.0  # all inliers: n·log(1) + 0·log(0) = 0
    end
    return pt
end

# -----------------------------------------------------------------------------
# Constructors: the prior is a keyword (Jeffreys by default)
# -----------------------------------------------------------------------------
#
# The spec carries no per-point state: only `model_dof` and the outlier
# half-width are consumed. The data count belongs to the WORKSPACE, which is
# sized by `scoring_workspace(scoring, n, d_g)`.

function StudentTUniformScoring{P, DG}(model_dof::Int, outlier_halfwidth::Real;
                                       prior::IGPrior = JeffreysPrior(), profile_prior::Bool = false) where {P, DG}
    outlier_halfwidth > 0 ||
        throw(ArgumentError("outlier_halfwidth must be positive, got $outlier_halfwidth"))
    return StudentTUniformScoring{P, DG}(Float64(outlier_halfwidth), model_dof, profile_prior, prior)
end

# Bare model DOF: the codimension is a keyword (1 by default).
StudentTUniformScoring{P}(model_dof::Int, outlier_halfwidth::Real; codimension::Int = 1,
                          prior::IGPrior = JeffreysPrior(), profile_prior::Bool = false) where {P} =
    StudentTUniformScoring{P, codimension}(model_dof, outlier_halfwidth; prior, profile_prior)
StudentTUniformScoring(model_dof::Int, outlier_halfwidth::Real; codimension::Int = 1,
                       prior::IGPrior = JeffreysPrior(), profile_prior::Bool = false) =
    StudentTUniformScoring{Nothing, codimension}(model_dof, outlier_halfwidth; prior, profile_prior)

# Problem-aware: m and d_g from the problem (an instance or its type).
StudentTUniformScoring(problem::Union{AbstractConstraintProblem, Type{<:AbstractConstraintProblem}},
                       outlier_halfwidth::Real; prior::IGPrior = JeffreysPrior(), profile_prior::Bool = false) =
    StudentTUniformScoring{Nothing, codimension(problem)}(minimal_sample_size(problem), outlier_halfwidth;
                                                          prior, profile_prior)
StudentTUniformScoring{P}(problem::Union{AbstractConstraintProblem, Type{<:AbstractConstraintProblem}},
                          outlier_halfwidth::Real; prior::IGPrior = JeffreysPrior(),
                          profile_prior::Bool = false) where {P} =
    (_check_provider(P, problem isa Type ? problem : typeof(problem));
     StudentTUniformScoring{P, codimension(problem)}(minimal_sample_size(problem), outlier_halfwidth;
                                                     prior, profile_prior))

_check_provider(::Type, ::Type) = nothing
_check_provider(::Type{Predictive}, ::Type{P}) where {P} =
    provides_solver_covariance(P) ? nothing : throw(ArgumentError(
    "predictive scoring needs a solver covariance, and $(nameof(P)) provides none: define " *
    "`VisualGeometryCore.solver_covariance(::$(nameof(P)), model, sample_indices)` and declare " *
    "`VisualGeometryCore.provides_solver_covariance(::Type{<:$(nameof(P))}) = true`, or score $(nameof(P)) model-certain"))

# =============================================================================
# alloc_scoring_workspace — build ScoringWorkspace from problem + scoring
# =============================================================================

"""
    alloc_scoring_workspace(problem, scoring::StudentTUniformScoring) -> ScoringWorkspace

Allocate scratch buffers for Algorithm 1 sweep. Depends on:
- `data_size(problem)` — buffer lengths
- `codimension(problem)` — lg_table step size
- `scoring.scale_prior` — lg_table α₀ offset (Jeffreys → 0.0)
- `scoring.profile_prior` — whether the profiled-γ correction table is built

Called by `RansacWorkspace(problem, scoring)` so buffers live on
`RansacWorkspace.scoring_ws`.
"""
function alloc_scoring_workspace(problem::AbstractConstraintProblem,
                                 scoring::StudentTUniformScoring)
    alloc_scoring_workspace(data_size(problem), codimension(problem), scoring)
end

"""
    alloc_scoring_workspace(n::Int, dg::Int, scoring::StudentTUniformScoring) -> ScoringWorkspace

Size-only overload: build a `ScoringWorkspace` from raw `n` / `dg` instead of
a live problem instance. Useful for benchmarks and pre-allocation contexts
where the sweep buffers must be sized for the maximum dataset a thread will
see, before any particular problem is in hand.

The resulting sws carries `scoring.scale_prior`, so subsequent
`sweep!(sws, ...)` calls use that prior without any kwarg plumbing.
"""
function alloc_scoring_workspace(n::Int, dg::Int, scoring::StudentTUniformScoring)
    lg = _build_lg_table(n, dg; alpha0=_alpha0(scoring.scale_prior))
    pt = scoring.profile_prior ? _build_prior_table(n) : Float64[]
    ScoringWorkspace(
        Vector{Int}(undef, n),
        Vector{Int}(undef, n),
        lg,
        zeros(Float64, n),
        Vector{Float64}(undef, n),
        pt,
        scoring.scale_prior,
    )
end

"""
    score_sq_residuals!(sws, q, scoring, problem, model, n) -> (S, k)

Score a model by computing per-point squared whitened residuals and sweeping
for the optimal partition (Algorithm 1, Eq. 12).

Fills `q[1:n]` with squared whitened residuals, `q[n+1:end]` with `typemax`,
and overwrites the `sws` sweep buffers.

Returns `(S, k)`: sweep score and optimal inlier count.
"""
function score_sq_residuals!(sws::ScoringWorkspace, q::AbstractVector{T}, s::StudentTUniformScoring,
                             problem::AbstractConstraintProblem, model,
                             n::Int) where T
    @inbounds for i in 1:n
        # squared norm of the whitened residual itself, not of its norm — no
        # sqrt round trip for a vector constraint
        q[i] = sq_norm(whitened_constraint_residual(constraint_with_jacobians(problem, model, i)))
    end
    @inbounds for i in n+1:length(q)
        q[i] = typemax(T)
    end
    _zero_penalties!(sws, s)
    return sweep!(sws, s, q, n)
end

"""
    score_residuals!(sws, q, scoring, residuals, n; sample_idx = nothing) -> (S, k)

Score from precomputed residuals (model-certain, isotropic noise).

Squares `residuals[1:n]` into `q`, pads `q[n+1:end]` with `typemax`, zeros the
`sws` penalties, and sweeps for the optimal partition (overwriting the `sws`
sweep buffers). The `sample_idx` points get `q = 0`.

Returns `(S, k)`: sweep score and optimal inlier count.
"""
function score_residuals!(sws::ScoringWorkspace, q::AbstractVector{T}, s::StudentTUniformScoring,
                          residuals::AbstractVector{<:Real},
                          n::Int;
                          sample_idx::Union{Nothing,AbstractVector{Int}}=nothing) where T
    square_residuals!(q, residuals, n)
    @inbounds for i in n+1:length(q)
        q[i] = typemax(T)
    end
    zero_sample!(q, sample_idx)     # bounds-checked: an invalid index throws
    _zero_penalties!(sws, s)
    return sweep!(sws, s, q, n)
end

"""
    score_sq_residuals!(sws, q, scoring::StudentTUniformScoring{Predictive}, problem, model, sample_idx, n) -> (S, k)

Predictive variant: compute model-uncertain scores qᵢ from solver Jacobian,
then sort-and-sweep (Algorithm 1, Eq. 20).

Computes Σ̃_θ from `solver_covariance(problem, model, sample_idx)`, then
fills `q[i]` with the predictive squared residual rᵢᵀ Ṽᵢ⁻¹ rᵢ.
A problem with no `solver_covariance` throws (see its docstring).

Returns `(S, k)`: sweep score and optimal inlier count. A DEGENERATE sample —
the provider returns `nothing` — is no hypothesis: `(-Inf, 0)`, as for a failed
solve, with `q` and `sws` untouched. It is never scored model-certain.
Otherwise it is the covariance-given method below at that Σ̃_θ.
"""
function score_sq_residuals!(sws::ScoringWorkspace, q::AbstractVector,
                             s::StudentTUniformScoring{Predictive},
                             problem::AbstractConstraintProblem, model,
                             sample_idx::AbstractVector{Int}, n::Int)
    Σ̃_θ = solver_covariance(problem, model, sample_idx)
    isnothing(Σ̃_θ) && return (-Inf, 0)
    return score_sq_residuals!(sws, q, s, problem, model, Σ̃_θ, sample_idx, n)
end

"""
    score_sq_residuals!(sws, q, scoring::StudentTUniformScoring{Predictive}, problem, model, Σ̃_θ,
                        sample_idx, n; residuals = nothing) -> (S, k)

Predictive scoring at a GIVEN unit-scale model covariance `Σ̃_θ`. The
convention is multiplicative, `Σ_θ = s²·Σ̃_θ`, and `Σ̃_θ` is a minimal sample's
`solver_covariance`, or a refit's on its consensus set, as an `Uncertain`
model carries it. Fills `q[i]` with the predictive squared residual
`rᵢᵀ Ṽᵢ⁻¹ rᵢ` and the `sws` penalties with `log|Ṽᵢ|`, then sort-and-sweeps
(Algorithm 1, Eq. 20).

`sample_idx` (or `nothing`): the minimal sample's points, which satisfy their
constraints exactly — their `q` and penalty are set to zero, as `q` is in
[`score_residuals!`](@ref). When `residuals` is supplied it receives each
point's [`whitened_residual`](@ref) from the same constraint evaluation — SIGNED
for a scalar constraint, exactly what `residuals!` writes.

Returns `(S, k)`: sweep score and optimal inlier count.
"""
function score_sq_residuals!(sws::ScoringWorkspace, q::AbstractVector{T},
                             s::StudentTUniformScoring{Predictive},
                             problem::AbstractConstraintProblem, model, Σ̃_θ,
                             sample_idx::Union{Nothing,AbstractVector{Int}}, n::Int;
                             residuals::Union{Nothing,AbstractVector{<:Real}} = nothing) where T
    @inbounds for i in 1:n
        rᵢ, ∂θgᵢ_w, _ = whiten(constraint_with_jacobians(problem, model, i)...)
        residuals === nothing || (residuals[i] = whitened_distance(rᵢ))
        q[i], sws.penalties[i] = _predictive_score_penalty(rᵢ, ∂θgᵢ_w, Σ̃_θ)
        # sort_key = q (unit-scale Mahalanobis). The historical raw-r²
        # key ("block high-leverage infiltration") is RETIRED: with
        # heterogeneous penalties sweep! takes the exact parametric path
        # (generalized sweep lemma, uncertain_evidence 35ae25c) whose key
        # t_i(μ) = g_i − μq_i carries the covariance penalty — the guard
        # is subsumed by the ordering itself (lemma consequence (iv)).
        sws.sort_key[i] = q[i]
    end
    # Sample points satisfy constraints exactly — their prediction variance
    # is zero (model is determined by them), so their penalty is spurious.
    # Bounds-checked on purpose, as in `score_residuals!`.
    zero_sample!(q, sample_idx)
    zero_sample!(sws.penalties, sample_idx)
    @inbounds for i in n+1:length(q)
        q[i] = typemax(T)
        sws.sort_key[i] = typemax(T)
    end
    return sweep!(sws, s, q, n)
end

_maybe_prior(sws::ScoringWorkspace) = isempty(sws.prior_table) ? nothing : sws.prior_table

sweep!(sws::ScoringWorkspace, s::StudentTUniformScoring{Nothing, DG},
       scores::Vector{T}, n::Int) where {DG, T} =
    sweep!(sws.perm, sws.lg_table, s.model_dof, log(2 * s.outlier_halfwidth), DG, scores, n;
           scratch=sws.scratch, prior_table=_maybe_prior(sws), scale_prior=s.scale_prior,
           radix=sws.radix)
sweep!(sws::ScoringWorkspace, s::StudentTUniformScoring{Predictive, DG},
       scores::Vector{T}, n::Int) where {DG, T} =
    sweep!(sws.perm, sws.lg_table, s.model_dof, log(2 * s.outlier_halfwidth), DG, scores, n;
           scratch=sws.scratch, penalties=sws.penalties,
           prior_table=_maybe_prior(sws), scale_prior=s.scale_prior,
           radix=sws.radix)

# (`sweep_marginal!` and `score_residuals_marginal!` — the logsumexp over
#  nested partitions instead of their max — are deleted with the `:marginal_soft`
#  grid they served: no caller in RSDR, RVG, VisualGeometryEval or BlobBoards.)

# =============================================================================
# Model Covariance Wrapping — Uncertain{M} construction
# =============================================================================

"""
    _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T}, Σ̃_unit) -> (qᵢ, ℓᵢ)

Model-uncertain score and penalty for a scalar constraint (dg=1). It uses the
MULTIPLICATIVE predictive-covariance convention of the generalized sweep
lemma (uncertain_evidence 35ae25c): the full predictive variance is
Ṽᵢ = s²·Vᵢ with the UNIT-SCALE shape

  Vᵢ = 1 + (∂θgᵢ_w)ᵀ Σ̃_unit (∂θgᵢ_w),

where Σ̃_unit is the propagation of UNIT measurement noise through the
solver/fit (the physical Σ_θ = s²·Σ̃_unit — the hypothesis derives from the
same noise process the measurements carry). Returned qᵢ = rᵢ²/Vᵢ and
ℓᵢ = log Vᵢ are therefore s-FREE; the s-dependence factors into the common
scale marginal exactly as the lemma's separation requires. (The former
`s2 +` argument slot read as an ADDITIVE convention with the model
covariance frozen at a reference scale — a latent modeling inconsistency
while s is profiled; numerically identical at s2 = 1, semantically
corrected here.)
"""
@inline function _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T},
                                            Σ̃_unit) where {n_θ,T}
    Vᵢ = one(T) + dot(∂θgᵢ_w, Σ̃_unit * ∂θgᵢ_w)
    Vᵢ > eps(T) || return (typemax(T), zero(T))
    qᵢ = rᵢ^2 / Vᵢ
    return (qᵢ, log(Vᵢ))
end

"""
    _predictive_score_penalty(rᵢ::StaticVector{dg,T}, ∂θgᵢ_w::StaticMatrix{dg,n_θ,T}, Σ̃_unit)
        -> (qᵢ, ℓᵢ)

Vector-constraint (dg≥2) form of the MULTIPLICATIVE convention (see the
scalar method). Here Vᵢ = I + (∂θgᵢ_w) Σ̃_unit (∂θgᵢ_w)ᵀ (unit-scale dg×dg shape),
qᵢ = rᵢᵀ Vᵢ⁻¹ rᵢ and ℓᵢ = log|Vᵢ| — both s-free; Ṽᵢ = s²·Vᵢ.
"""
@inline function _predictive_score_penalty(rᵢ::StaticVector{dg,T}, ∂θgᵢ_w::StaticMatrix{dg,n_θ,T},
                                            Σ̃_unit) where {dg,n_θ,T}
    # `rᵢ`/`∂θgᵢ_w` are `StaticVector`/`StaticMatrix`, not narrowly `SVector`/
    # `SMatrix`: the whitened homography residual arrives as a
    # `GeometryBasics.Vec{2}` (a StaticVector, but NOT an SVector), so the
    # narrow signature threw a MethodError on the Uncertain{HomographyMat}
    # path. Behavior-identical for existing SVector callers (same body,
    # same SMatrix arithmetic via the `SMatrix{dg,dg,T}(I)` promotion).
    Vᵢ = SMatrix{dg,dg,T}(I) + ∂θgᵢ_w * Σ̃_unit * ∂θgᵢ_w'
    det_Vᵢ = det(Vᵢ)
    det_Vᵢ > eps(T) || return (typemax(T), zero(T))
    qᵢ = dot(rᵢ, Vᵢ \ SVector{dg,T}(rᵢ))
    return (qᵢ, log(det_Vᵢ))
end

# =============================================================================
# Per-point scoring helpers used by both score_sq_residuals! (here) and the
# RansacWorkspace-typed score!/mask! glue (which lives in RobustVisualGeometry,
# since it reads/writes the RANSAC workspace buffers).
# =============================================================================

_zero_penalties!(::ScoringWorkspace, ::StudentTUniformScoring{Nothing}) = nothing
_zero_penalties!(sws::ScoringWorkspace, ::StudentTUniformScoring{Predictive}) = fill!(sws.penalties, 0.0)

# RobustVisualGeometry's RANSAC loop scores through `score_residuals!` (a plain
# model) and the covariance-given `score_sq_residuals!` (an Uncertain{M}),
# naming its workspace buffers; it computes no score of its own.


# =============================================================================
# The scoring contract (VisualGeometryCore src/robust/scoring_contract.jl)
# =============================================================================
#
# The sort-and-sweep IS the contract's fused path: one `score!` produces the
# score, k* (cached on the workspace for `inlier_count`) and the sorted order
# (`consensus_order`), whose first k* entries are the consensus set.

scoring_workspace(s::StudentTUniformScoring, n::Int, d_g::Int) = alloc_scoring_workspace(n, d_g, s)

# Model-certain: the residuals are given; `q` receives rᵢ² (the sample's points
# zeroed). Under predictive scoring a PLAIN model is scored this way too (a
# seed scored model-certain to find its consensus set): the penalties are zeroed.
function score!(sws::ScoringWorkspace, s::StudentTUniformScoring, q::AbstractVector,
                residuals::AbstractVector, n::Int; sample = nothing, bound::Float64 = -Inf)
    S, k = score_residuals!(sws, q, s, residuals, n; sample_idx = sample)
    sws.k = k
    return S
end

# Predictive: the model carries its unit-scale covariance (`Uncertain`); the
# residuals receive each point's whitened residual from the same constraint
# evaluation.
function score!(sws::ScoringWorkspace, s::StudentTUniformScoring{Predictive}, q::AbstractVector,
                residuals::AbstractVector, problem, θ::Uncertain, n::Int;
                sample = nothing, bound::Float64 = -Inf)
    S, k = score_sq_residuals!(sws, q, s, problem, geom(θ), Statistics.cov(θ),
                               sample, n; residuals)
    sws.k = k
    return S
end

inlier_count(::StudentTUniformScoring, sws::ScoringWorkspace) = sws.k
consensus_order(::StudentTUniformScoring, sws::ScoringWorkspace) = sws.perm
is_predictive(::StudentTUniformScoring{Predictive}) = true

"""
    scored_model(scoring::StudentTUniformScoring{Predictive}, problem, θ, sample) -> Uncertain | nothing

The model with the unit-scale covariance of its fit on the points `sample`
(`solver_covariance`); `nothing` for a degenerate sample — no hypothesis,
skipped like a failed solve, never scored model-certain.
"""
function scored_model(::StudentTUniformScoring{Predictive}, problem, θ, sample)
    Σ̃_θ = solver_covariance(problem, θ, sample)
    isnothing(Σ̃_θ) && return nothing
    return Uncertain(θ, CovMat(Σ̃_θ))
end

# The decision point of the last score: the Gaussian-uniform MAP crossover of the
# consensus set's Student-t scale (at the local optimizer's dof, m + 1) against
# the uniform outlier box.
function inlier_threshold(s::StudentTUniformScoring{P, DG}, buf::ScoreBuffers) where {P, DG}
    sws = buf.scoring_ws
    k = sws.k
    s2 = consensus_s2(s, residuals(buf), sws.perm, k; model_dof = s.model_dof + 1, d_g = DG)
    return decision_threshold(GauInlier(s2; d_g = DG), s.outlier_halfwidth)
end
has_inlier_threshold(::Type{<:StudentTUniformScoring}) = true

# A workspace's tables are built for one prior and one profile flag.
function check_workspace(s::StudentTUniformScoring, sws::ScoringWorkspace)
    sws.scale_prior == s.scale_prior && isempty(sws.prior_table) == !s.profile_prior && return nothing
    throw(ArgumentError(
        "the workspace's scoring tables were built for scale prior $(sws.scale_prior) " *
        "(profile_prior = $(!isempty(sws.prior_table))), but the scoring has $(s.scale_prior) " *
        "(profile_prior = $(s.profile_prior)); build the workspace from this scoring"))
end

"""
    with_width(scoring::StudentTUniformScoring, a::OutlierHalfwidth) -> StudentTUniformScoring

The same scorer at the outlier half-width `a` (every other field kept).
"""
with_width(s::StudentTUniformScoring{P, DG}, a::OutlierHalfwidth{Float64}) where {P, DG} =
    StudentTUniformScoring{P, DG}(a.value, s.model_dof, s.profile_prior, s.scale_prior)
width_axis(::StudentTUniformScoring) = OutlierHalfwidth()
