# RANSACScoringDoneRight: duplicate removals of the 0.9 leg

Owner decision 12 (2026-11-11) and the owner's ruling "all duplicates must be gone; all common code must be hoisted
without exception". RSDR has no section in cascade-duplicate-removals.md; the items are the SIMFIT++ move, everything
the duplication audit (duplication-audit.md, 2026-11-10) attributes to RSDR, and every hit of Core's
`test_duplication` (monorepo dde0b5ca) on this repository. Each item is ticked with the commit that closes it.

## Removals

- [x] SIMFIT++ copy (src/simfitpp.jl, test/simfitpp_tests.jl, the train-split row sum `_loss_width_sum!`) → VisualGeometryToolkitEstimation's `SimFitPP`/`simfitpp_select` (e8f2f60)
- [x] test/docs_gates.jl (audit 1, the docs-gate copy) → VisualGeometryToolkitCore's `test_docs` (e8f2f60)
- [x] the private device-upload helper of src/marg_sweep_ka.jl (audit 17; a retired name now) → VisualGeometryToolkitCore's `ka_upload` (e8f2f60)
- [x] `SCORING_TAGS`, a name VisualGeometryToolkitCore owns (test_duplication `[name]`) → renamed `ALL_SCORING_TAGS`, the merge of Core's `SCORING_TAGS` with this package's families that Core's docstring prescribes (e8f2f60)
- [x] neurips2026 `prose_macros`, a `\newcommand` writer (audit 11; the paper-project item "its TeX writer → Eval's number-macro writer"; test_duplication `[pattern] tex_macro_writer`) → VisualGeometryToolkitEval's `PaperNumbers`/`add_number!`/`write_numbers` (0f91df8)
- [x] the suite runs VisualGeometryToolkitCore's `test_duplication` (test/quality_naming_tests.jl) (e8f2f60)
- [x] the scalar-constraint test problem restated in four items (`constraint_with_jacobians` ×3, `solver_covariance` ×2; test_duplication `[similar]`) → one `LineProblems` test setup in test/fixtures_testsetups.jl (86ec6b2)
- [x] neurips2026's `verify()`/`deploy()` restated in three drivers (test_duplication `[similar]`) → `verify_paper_assets`/`deploy_paper_assets` in neurips2026/src/assets.jl (1f14ef2)

## Hoists into the stack

Common code no stack package has yet. Not kept as a private copy by choice: each waits for its owner to land it, and
the switch here is the line noted.

- [x] hoist `.githooks/sync-manifest.jl` (and the shell hooks `sync-manifest`, `post-checkout`, `post-merge`, `post-rewrite`, byte-identical to BlobBoards'; audit 20; test_duplication `[file]` against BlobBoards:.githooks/sync-manifest.jl) into VisualGeometryToolkitCore's quality module as `sync_manifest(mode::AbstractString = "update"; io = stdout) -> NamedTuple{(:created, :moved, :seconds)}` (re-resolve the active project — `Pkg.update` for an untracked Manifest, `Pkg.instantiate` for a tracked one — and report the direct dependencies that moved). Callers here: `.githooks/sync-manifest` (runs the .jl after checkout, merge and rewrite). The switch: `.githooks/sync-manifest.jl` becomes `using VisualGeometryToolkitCore: sync_manifest; sync_manifest(get(ARGS, 1, "update"))`; the shell hooks belong with it (Core's `quality/githooks/`, installed by a `install_githooks(repo)`), so no repository carries a copy. — CLOSED: VisualGeometryToolkitCore `sync_manifest`/`install_githooks`/`githooks_dir` (monorepo 1555c4ab; the hooks are Core's `githooks/`, the Julia side src/quality/sync_manifest.jl, included by path so the hook never loads a package from an illegal Manifest) and .githooks/ deleted here (382b9c1)
- [x] hoist neurips2026 `paper_figdir()` (src/assets.jl:22, `projectdir("paper", "figures")`; audit 20, also RadiometricCamera:cvpr2027/src/assets.jl:24) into VisualGeometryToolkitEval's `PaperProjects` as `paper_figures_dir(project_root = DrWatson.projectdir()) -> String` (the directory the paper reads generated assets from; Eval's template already spells it `PAPER_FIGURES = projectdir({{FIGDIR_ARGS}})`). Callers here: neurips2026/src/assets.jl (`verify_paper_assets`, `deploy_paper_assets`), scripts/fig_profile.jl, scripts/fig_sensitivity.jl, scripts/paper_numbers.jl, test/test_figures.jl. The switch: delete the definition and import `paper_figures_dir` in src/NeurIPS2026.jl under the name used. — CLOSED: VisualGeometryToolkitEval `PaperProjects.paper_figures_dir(paper = "paper/main.tex"; root = project_root())` (monorepo 5108b97f) and the definition deleted here (382b9c1)
- [x] hoist neurips2026 `artifact_catalog()` (src/data_artifacts.jl:21, `load_catalog(<project>/configs/data_artifacts.toml)`; audit 20, also LocalOptimizationDoneRight:accv2026/src/data_artifacts.jl:30) into VisualGeometryToolkitEval's `DataArtifacts` as `project_catalog(project_root = DrWatson.projectdir()) -> DataCatalog` (the project's `configs/data_artifacts.toml`, bound to its `Artifacts.toml` or the env override). Callers here: neurips2026/scripts/fetch_data.jl (and configs/data_artifacts.toml's comments). The switch: delete the definition and import `project_catalog`. — CLOSED: VisualGeometryToolkitEval `DataArtifacts.project_catalog(root = project_root())` (monorepo 4b9b95d4) and the definition deleted here (382b9c1)
- [x] hoist the `_partition_score` of src/sweep.jl (audit 7: BlobBoards:src/adapt_ka.jl:316 `_sweep_score` restates it) — RSDR owns the score, so the item here is to make it public and device-callable (`partition_score(...)`, plain Float64 arithmetic already) so BlobBoards' leg can delete its copy. Not done in this leg: it needs BlobBoards' kernel signature agreed with its leg; RSDR's own code has no copy. — CLOSED here: `partition_score(prior, lg_half_dof, half_dof, RSS, outlier_cost)` is public and precision-generic (the formula that was `_sweep_score`; the size-indexed form is `_table_partition_score`) (6ddd808); BlobBoards' deletion of its `_sweep_score` is its own leg's switch

- [x] the twin audit's V1 (cpu-gpu-twin-audit.md, owner decision 25: one KernelAbstractions kernel on every backend): `exact_pm_score_kernel!` and `argmax_models_kernel!` of src/loss_width_sweep_ka.jl, the device form of VisualGeometryToolkitCore's host selection (`select_on_grid!` → `best_pool_model` on a threaded loop) → VisualGeometryToolkitCore's `exact_pm_score_kernel!`/`argmax_models_kernel!` (monorepo 6accc512), which Core's `select_on_grid!` now launches on the CPU backend; `_support_sq` → Core's `support_sq_loss_width` (this package's methods for MAGSAC++ and smooth TLS), `has_residual_quality` for its five families, and `precompute_pair`'s sorted columns → Core's `sorted_squared_residuals`. The copies are deleted here (this commit).

## Not a duplicate

- The five loss-width scorer structs (`MlesacScoring`, `MagsacScoring`, `WelschCostScoring`, `SmoothTLSScoring`,
  `GemanScoring`; audit 20 "RSDR scorer boilerplate"): extension families of Core's threshold-scoring contract, not
  copies (the audit says so); test_duplication does not flag them. No item.

## test_duplication

See the leg report (/home/jpr/logs/dgx-agent/cascade-09-RANSACScoringDoneRight.md) for the run and its output.
