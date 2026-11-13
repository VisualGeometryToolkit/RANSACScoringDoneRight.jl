# RANSACScoringDoneRight.jl

[![CI](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-ransacscoringdoneright.github.io-blue.svg)](https://ransacscoringdoneright.github.io)
[![Coverage](https://codecov.io/gh/VisualGeometryToolkit/RANSACScoringDoneRight.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/VisualGeometryToolkit/RANSACScoringDoneRight.jl)

RANSACScoringDoneRight scores RANSAC hypotheses with the scale-free marginal
score of the NeurIPS 2026 paper
[RANSAC Scoring Done Right](https://arxiv.org/abs/2606.27385): the inlier scale
is integrated out in closed form, and the best inlier partition is found by one
sort-and-sweep, so no inlier threshold has to be set. It also holds the paper's
baseline scorers (RANSAC, MSAC, GaU, MAGSAC++ and three robust losses) and CPU
and GPU evaluators that score a stored hypothesis pool at every value of a
hyperparameter grid.

The scorers implement the scoring contract of
[VisualGeometryToolkitCore](https://github.com/VisualGeometryToolkit/VisualGeometryToolkit.jl/tree/main/VisualGeometryToolkitCore),
so any RANSAC written against that contract can use them, including
VisualGeometryToolkitEstimation's.

## Installation

The package requires Julia 1.13 or later. The example also loads
VisualGeometryToolkitCore, which defines the scoring contract:

```julia-repl
pkg> add RANSACScoringDoneRight VisualGeometryToolkitCore
```

## Example

Score one hypothesis from its residuals: the sweep sorts them, scores every
candidate inlier partition with the scale marginalized out, and returns the
best score with its inlier count, whose scale estimate follows in closed form.

```julia
using RANSACScoringDoneRight
using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, FlatLabelPrior,
    scoring_workspace, score!, inlier_count, consensus_order, consensus_s2
using Random

# The residuals of one line hypothesis (codimension 1): 80 inliers with noise
# scale 0.01 and 20 outliers spread uniformly over [-0.5, 0.5].
rng = Xoshiro(7)
n = 100
r = [0.01 .* randn(rng, 80); rand(rng, 20) .- 0.5]

# a line has 2 parameters; outlier half-width 0.5; the scale and label priors (Jeffreys, flat) are required
scoring = StudentTUniformScoring(2, 0.5; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
W = scoring_workspace(scoring, n, 1)      # sweep buffers, reusable across hypotheses
q = Vector{Float64}(undef, n)
S = score!(W, scoring, q, r, n)           # the score, with the scale integrated out
k = inlier_count(scoring, W)              # the selected inlier count: about 80
s2 = consensus_s2(scoring, r, consensus_order(scoring, W), k; model_dof = 2, d_g = 1)
sqrt(s2)                                  # the partition's scale ŝ: about 0.01
```

`StudentTUniformScoring(2, 0.5; scale_prior = IGPrior(α₀, β₀), label_prior = FlatLabelPrior())`, with
VisualGeometryToolkitCore's `IGPrior`, scores under an informative scale prior
instead of the Jeffreys prior. The [Overview](https://ransacscoringdoneright.github.io/overview/)
continues from here.

## Documentation

- [Overview](https://ransacscoringdoneright.github.io/overview/): scoring
  versus selection, the scorers, the scale priors and the pool frame.
- [Experiments](https://ransacscoringdoneright.github.io/guide/experiments/):
  the scoring studies shipped with the repository.
- [Conventions](https://ransacscoringdoneright.github.io/conventions/): units,
  frames and widths, and why they are named as they are.
- [API Reference](https://ransacscoringdoneright.github.io/api/scoring/).

The package is part of the
[VisualGeometryToolkit](https://github.com/VisualGeometryToolkit) stack:
VisualGeometryToolkitCore defines the contracts it implements, and
[BlobBoards](https://blobboards.github.io) and the ACCV 2026 study of
[Robust Local Optimization Done Right](https://localoptimizationdoneright.github.io)
use its scorers. The paper's research project, with the drivers of every
figure, is in [`neurips2026/`](neurips2026/REPRODUCE.md).

## Citing

James Pritts, Felix Seegräber and Kevin Köser. RANSAC Scoring Done Right.
*NeurIPS 2026* ([arXiv](https://arxiv.org/abs/2606.27385)).

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
