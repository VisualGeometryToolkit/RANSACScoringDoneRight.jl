docs-pass: 707f0bc
landing: RansacScoringDoneRight/ransacscoringdoneright.github.io a44d59e (no change: a compat patch; nothing the site shows moved)

# 0.7.1 docs pass (julia-docs-pass) — on VisualGeometryToolkitEstimation 0.10

RANSACScoringDoneRight 0.7.1 is a patch: `[compat]` VisualGeometryToolkitEstimation "0.10" in every member that declares it, and the
CHANGELOG section. No public name added, removed or re-signatured.
- API diff since the last release: none.
- docs build: the overview's `score_sq_residuals!` link pointed at a name Core's inventory holds only per method (three
  method docstrings, no generic one), so `@extref` was ambiguous and the build failed; it links Core's Student-t scoring
  page now.
- stale mentions: nothing here names the SplineLens image-space refine Estimation 0.10.0 retired (its free route,
  `penalty`, the GCV and window metadata): checked by grep over src, docs, README and the paper projects.
- README / install lines: unchanged (no version is named); the README rows resolve the registry and pass once
  Estimation 0.10.0 is registered.

---

# Docs pass: RANSACScoringDoneRight 0.7.0 (2026-11-12)

The julia-docs-pass skill on the 0.7.0 release tree (rsdr-09, the package branch with paper-09-rsdr merged).

## API diff since v0.6.0 (the last reviewed docs)

- Removed from this package (moved to the stack, or deleted): `StudentTUniformScoring`, `Predictive`, its label priors,
  `FixedHalfwidth`, `AbstractOutlierModel`, `ScoringWorkspace`, `alloc_scoring_workspace`, `default_scoring_workspace`,
  `sweep!`, `sweep_order`, `penalties`, `sort_key`, `lg_table`, `label_prior`, `outlier_model`, `score_residuals!`,
  `score_sq_residuals!` (all VisualGeometryToolkitCore's now); `SimFitPP`, `simfitpp_loss_width`, `simfitpp_select`,
  `unbiased_median_variance`, `with_residual_unit` (VisualGeometryToolkitEstimation's); `SCORING_TAGS` (renamed
  `ALL_SCORING_TAGS`); `Exact` and `Aggregation` (deleted: the exact selection is Core's `select_on_grid!`).
- Added: `ALL_SCORING_TAGS`.
- Every remaining mention of a moved name in docs/src and README.md names its new owner through `@extref`. The twelve
  `@extref` targets (Core: `GauInlier`, `ScoringWorkspace`, `StudentTUniformScoring`, `best_pool_model`,
  `decision_boundary`, `decision_width`, `residual_quality`, `score_residuals!`, `score_sq_residuals!`; Estimation:
  `SimFitPP`, `simfitpp_loss_width`, `simfitpp_select`) are public in the registered 0.9 stack and documented in its
  manuals. No mention of `Exact`, `Aggregation`, `SCORING_TAGS` or another removed name remains.

## Docstrings

- The suite's `docs` item (VisualGeometryToolkitCore's `test_docs`: undocumented public names, signature drift,
  detached docstrings) passes, with the rest of the suite (10653 / 10653, resolved from General, job 9896).
- Fixed: eight docstrings (the extreme-value outlier models, `TailRules`, `gpd_pwm_fit`, `tail_fit`, `gpd_support`,
  `gpd_outlier_cost`) open with a one-sentence summary; the docs build's CodeBlocks warned that their first paragraph
  had no sentence boundary for the tooltip brief (f0ba187).

## README and the readme test

The README example uses only VisualGeometryToolkitCore names (the Student-t scorer is Core's); its install line adds
RANSACScoringDoneRight and VisualGeometryToolkitCore. The `readme` item now PASSES from the registry (it failed only
while the 0.9 stack was unregistered).

## Docs build

`docs/make.jl` (checkdocs = :public, doctest, linkcheck, no warnonly), resolved from General:

- Everything except the external cross-references builds clean (job 9903: a copy of make.jl with only
  `warnonly = [:external_cross_references]` added, rc 0; the CodeBlocks warnings fixed in f0ba187).
- **Open, outside this repository:** the `@extref` links fail against the PUBLISHED inventories, which are still the old
  releases (`VisualGeometryToolkitCore/dev/objects.inv` is 0.6.0, `VisualGeometryToolkitEstimation/dev/objects.inv`
  0.6.1 at 2026-11-12 12:45Z). The monorepo's 0.9 docs are not deployed yet (its Docs runs for the Core and
  Estimation 0.9.0 tags were not the ones that deployed). Every target exists in the 0.9 sources and their docs pages,
  so the build passes as written once Core's and Estimation's 0.9 docs are at `/dev/`. Nothing here changes for it.
- Built on Julia 1.13.3 only: no 1.12 is installed on dgx01.
