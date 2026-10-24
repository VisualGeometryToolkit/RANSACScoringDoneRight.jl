# RANSACScoringDoneRight.jl

[![CI](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-ransacscoringdoneright.github.io-blue.svg)](https://ransacscoringdoneright.github.io)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/RANSACScoringDoneRight.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/VisualGeometryToolkit/RANSACScoringDoneRight.jl)

**[RANSAC Scoring Done Right](https://arxiv.org/abs/2606.27385)**
James Pritts, Felix Seegräber and Kevin Köser. *NeurIPS 2026*.
[Project page](https://ransacscoringdoneright.github.io).

The most widely used RANSAC variants score candidate models by counting inliers
or summing per-point scores that saturate beyond a residual threshold. Every
such score requires a user-supplied parameter that is a function of the inlier
scale, which must itself be estimated from contaminated data. We remove this
dependence by reversing the usual order of inference: rather than estimating
the scale and then scoring against it, we marginalize the inlier scale
analytically in closed form under a conjugate Inverse-Gamma prior for a fixed
inlier partition, then optimize over partitions. A single closed-form
expression spans the non-informative Jeffreys limit and informative
empirical-Bayes priors, so the same score adapts across data-rich and
data-scarce regimes without any change to the algorithm. The proposed RANSAC
score is the first in which the inlier scale is genuinely absent from the
formula. The score admits O(N log N) computation via sort-and-sweep. On a
benchmark of nearly 70 000 image pairs spanning different two-view estimation
problems and both engineered and learned feature pipelines, the proposed score
exceeds the state of the art (RANSAC, MSAC, GaU, MAGSAC): it stays nearly flat
under threshold miscalibration where baselines degrade, reaches near-optimal
accuracy from as few as two validation pairs where baselines need on the order
of 100 times more, and tightens its prior regularization as validation data
grows scarce.

## What the package does

RANSACScoringDoneRight implements the paper's scale-free marginal score for
RANSAC, `StudentTUniformScoring`. Given the residuals of one model hypothesis,
it finds the best inlier partition and its score with the inlier scale
integrated out, so no inlier threshold has to be set. The package also has the
paper's baseline scorers (RANSAC, MSAC, GaU, MAGSAC++ and three robust losses)
and CPU and GPU evaluators that score a stored pool of hypotheses at every
value of a hyperparameter grid.

The scorers plug into the scoring contract of
[VisualGeometryToolkitCore](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore),
so any RANSAC written against that contract can use them, including
VisualGeometryToolkitEstimation's. The repository also holds the paper's
research project in `neurips2026/`.

## Installation

The package requires Julia 1.12 or later. The quick start below also uses
VisualGeometryToolkitCore, which defines the scoring contract:

```julia-repl
pkg> add RANSACScoringDoneRight VisualGeometryToolkitCore
```

## Quick start

Score one hypothesis from its residuals. The sweep sorts them, scores every
candidate inlier partition with the scale marginalized out, and returns the
best score with its inlier count. The scale estimate of that partition then
follows in closed form.

```julia
using RANSACScoringDoneRight
using VisualGeometryToolkitCore: scoring_workspace, score!, inlier_count, consensus_order, consensus_s2

# The residuals of one line hypothesis (codimension 1): 80 inliers with noise
# scale 0.01 and 20 outliers spread uniformly over [-0.5, 0.5].
n = 100
r = [0.01 .* randn(80); rand(20) .- 0.5]

scoring = StudentTUniformScoring(2, 0.5)          # a line has 2 parameters; outlier half-width 0.5
W = scoring_workspace(scoring, n, 1)              # sweep buffers, reusable across hypotheses
q = Vector{Float64}(undef, n)
S = score!(W, scoring, q, r, n)                   # the score
k = inlier_count(scoring, W)                      # the selected inlier count
s2 = consensus_s2(scoring, r, consensus_order(scoring, W), k; model_dof = 2, d_g = 1)   # its scale ŝ²
```

`StudentTUniformScoring(2, 0.5; prior = IGPrior(α₀, β₀))`, with
VisualGeometryToolkitCore's `IGPrior`, scores under an informative scale prior
instead of the Jeffreys prior.

## Links

- [Documentation](https://ransacscoringdoneright.github.io): the overview, the
  API reference, the conventions and the guide for people working on the stack.
- [Paper](https://arxiv.org/abs/2606.27385) on arXiv.
- [`neurips2026/REPRODUCE.md`](neurips2026/REPRODUCE.md): how each figure of the
  paper is reproduced.
- [VisualGeometryToolkit](https://github.com/VisualGeometryToolkit), the stack
  of Julia packages this one belongs to.

## Citation

```bibtex
@inproceedings{Pritts2026RansacScoring,
  title         = {{RANSAC} Scoring Done Right},
  author        = {Pritts, James and Seegr{\"a}ber, Felix and K{\"o}ser, Kevin},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS)},
  year          = {2026},
  eprint        = {2606.27385},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

MIT
