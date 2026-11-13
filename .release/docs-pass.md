docs-pass: b4c5b15
landing: RansacScoringDoneRight/ransacscoringdoneright.github.io a44d59e (no hand edit: the site is the Docs workflow's build of docs/src; the install line there changes when main's Docs job deploys this tree)

# 0.8.0 docs pass (julia-docs-pass), the wave-10 stack

[x] API diff since 0.7.2: none in this package (no public name added, removed or re-signatured; src/ changes are comments and
    the five loss-width scorer structs dropping a written-out default constructor, so `S{DG, T}(threshold)` is unchanged).
[x] Docstrings: unchanged. The suite's `docs` item (Core's test_docs) passes in the wave-10 gate.
[x] Terminology (owner ruling 2026-11-13): docs/src/conventions.md names Core's `pixel_correspondences`. CLAUDE.md says the pool
    record holds correspondences. "measurement" left in prose meant a timing or an error table and now says so. No
    keypoint or sighting appears. check_naming finds 0 in the gate.
[x] Install lines: README, docs/src/index.md, docs/src/guide/experiments.md, neurips2026/README.md and REPRODUCE.md add
    no registry (the whole stack is in General). REPRODUCE.md keeps a one-line note for a depot that holds another registry.
    The CI and docs workflows install from General alone; the Downgrade workflow is deleted.
[x] neurips2026: REPRODUCE.md names the artifact `eth3d_splg_correspondences` and points to its `published_name` (the Zenodo
    file of record), without spelling the published token in prose.
[x] Docs build: docs/make.jl (strict, doctests, linkcheck) green in the wave-10 gate against the upstream snapshot.
[x] README example: the `readme` item runs the example in a fresh environment (on the -tmp branch it develops the snapshot's
    Core, since the wave's Core is not registered yet).

# 0.7.2 docs pass (julia-docs-pass) — Julia 1.13

RANSACScoringDoneRight 0.7.2 is a patch whose only change is the Julia requirement: `julia = "1.13"` in the `[compat]` of every
Project.toml in the repository, the CI workflows on Julia 1.13 only, and the CHANGELOG section.
- API diff since the last release: none (no source file under src/ or ext/ changed).
- Docstrings: unchanged; the suite's docs item runs in the fresh-clone gate.
- README / landing / guides: every statement of the required Julia version says 1.13 now: README.md, docs/src/index.md and neurips2026/REPRODUCE.md.
  Mentions of Julia 1.12 left in test comments record why a bound or a workaround was set (LLVM 18 vs 20 rounding,
  Julia 1.12's docstring rules, which 1.13 keeps); they are history, not instructions.
- Re-recorded after the Downgrade workflow became on-demand only (CI and CHANGELOG only; no docs change).
- Docs build: run by the fresh-clone gate on Julia 1.13.3 against the registered stack (General from GitHub).

---

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
