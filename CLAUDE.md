# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RansacScoringDoneRight: scale-free marginal RANSAC scoring, as an
implementation of VisualGeometryToolkitCore's contracts — the scoring contract
(`StudentTUniformScoring`, the sort-and-sweep marginal score) and the
pool-selection contract (`select_on_grid!` over its CPU/GPU grid families:
the marginal, the loss-width baselines, `LossSpec`, `ScaleMarginal`,
SIMFIT++). It depends only on VisualGeometryToolkitCore (never on
VisualGeometryToolkitEstimation, LocalOptimizationDoneRight or VisualGeometryToolkitEval).
Problem-agnostic: problems implement VisualGeometryToolkitCore's
`AbstractConstraintProblem`, and the Student-t posterior-weighting toolkit,
the width axes and the MSAC/RANSAC baseline scorers are VisualGeometryToolkitCore's.
The repository has two halves, like LocalOptimizationDoneRight's: the
registered package (`src/`, `test/`, `docs/`) and the research project
`neurips2026/` — the NeurIPS 2026 paper, its figure drivers, configs of record,
the REPRODUCE.md asset index and its own environment (a DrWatson project that
devs this package by path; see `neurips2026/CLAUDE.md`). The package never
depends on the project; paper material (drivers, studies, write-ups, stored
results) never goes in `src/`.

## Testing

ReTestItems (`test/*_tests.jl`, one item per file):
`Pkg.test(; test_args = ["simfitpp"])` filters items by name. Check the reported
total: 0 tests means nothing matched. The research project's suite is separate:
`cd neurips2026 && julia --project=. test/runtests.jl`.

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "🤖 Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself

## Stack Conventions

The conventions every stack package follows — naming and the estimator verbs,
signatures, data arguments, coordinate frames and normalization ("frame in =
frame out"), pixel origins, bearings, Jacobians, units and the stack
contracts — live ONCE, in VisualGeometryToolkitCore's `conventions/` folder
(`conventions/README.md` indexes it). Read them before changing code here.
This file carries no copy of them, only how they land in this package. Where
the folder is:

- in a checkout of the stack's monorepo: `../VisualGeometryToolkitCore/conventions/`
  (the root CLAUDE.md imports it);
- in VisualGeometryToolkitCore's own repository: `conventions/`;
- from Julia, the installed copy: `VisualGeometryToolkitCore.conventions_dir()`, i.e.
  `joinpath(pkgdir(VisualGeometryToolkitCore), "conventions")`.

The agent-docs test checks that this file points there and holds no copy.

## Coordinate Frames and Normalization

The rule is the stack's (`conventions/frames.md`); how it lands here:

### The pool frame here

The pool record is VisualGeometryToolkitCore's `PoolPair` (`VisualGeometryToolkitCore.Pools`):
csponds and models in `NormalizedImageFrame`, the maps its `source_map`/
`target_map`, the two sides of `pool_normalization(pair)` (VGC also builds the
pool frame of two image sizes there). The residual matrix is NOT in the record:
every pool scorer here takes the record AND its `num_points × num_models`
Float32 matrix `R` in the same unit — `best_pool_model(scoring, pair, R, …)`,
`pool_model_score`, `precompute_pair(pair, R)`,
`stage_ig_shared_from_pairs(pairs, residuals)` — and never opens a file. Every
scorer width on pool data (the outlier half-width, the loss width, every
`grid_evaluate` grid, the `SimFitPP` bounds, `POOL_FRAME_R_MAX`) is in that
frame's unit and says so. `POOL_FRAME_R_MAX = 1` is a declared outlier-support
constant in that unit, the one every tuned constant was cross-validated with.
`pixel_residuals(pair)` is the one exit to pixels: the image-plane Sampson of
the stored models, from the light record alone (it never reads `R`, so it does
not depend on the stored residual kind — tangent Sampson for E pools).
