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

- [x] the twin audit's V2 (owner ruling 2026-11-11, Option A: one kernel, the slower CPU accepted): `sweep!`'s host prefix-RSS scan (src/sweep.jl) beside `ig_sweep_kernel!` → the one kernel, which now carries the per-point penalties, the prior table and the best k; `sweep!` is the driver (sort, gather, parametric path) and launches it as one CPU work item; (S*, k*) bit for bit the scan's (this commit)

- [x] the owner's ruling (2026-11-11): the Student-t + uniform scorer hoisted into the stack with no duplication →
  VisualGeometryToolkitCore's `StudentTUniformScoring` with its sweep, prefix kernel, workspace, label priors and
  supplied half-width (monorepo c7b8106e); the defaults of its priors in VisualGeometryToolkitEstimation's
  default.toml and its TOML table read there (74db20d2). src/sweep.jl and the scorer's half of src/scoring.jl,
  extreme_value.jl and ig_sweep_ka.jl deleted here; the research outlier models extend Core's hooks (this commit)

## Not a duplicate

- The five loss-width scorer structs (`MlesacScoring`, `MagsacScoring`, `WelschCostScoring`, `SmoothTLSScoring`,
  `GemanScoring`; audit 20 "RSDR scorer boilerplate"): extension families of Core's threshold-scoring contract, not
  copies (the audit says so); test_duplication does not flag them. No item.

## test_duplication

See the leg report (/home/jpr/logs/dgx-agent/cascade-09-RANSACScoringDoneRight.md) for the run and its output.

## Robust schema

The owner's ruling (2026-11-11): "There's no such thing as a threshold for the Student-t. There's only the outlier
half-width." Every name that called a Student-t's half-width a threshold, or used one name for both meanings,
before (e9545f8) → after (this commit). Pure renames: no value moved (the timing protocol's defaults stay 1.64e-3 each).

- [x] neurips2026/scripts/studies/score_timing.jl:68–69 `timed_scorers(pairs, loss_width)` with
  `outlier_halfwidth = loss_width` (the loss width reinterpreted as the Student-t's half-width) →
  :68 `timed_scorers(pairs; loss_width, outlier_halfwidth)`, two values, neither read as the other
- [x] neurips2026/scripts/studies/score_timing.jl:139 the CLI `--loss-width` feeding both → :138–139 `--loss-width`
  (the loss-width families') and `--outlier-halfwidth` (the Student-t's), each default 0.00164; :152 the call, :155
  the saved record (`outlier_halfwidth` saved beside `loss_width`); :88 the docstring's call
- [x] neurips2026/scripts/studies/score_timing.jl:46–51 the comment stating the reuse → :46–51 the two options
- [x] neurips2026/scripts/slurm/score_timing.sbatch:43,45,50 `LOSS_WIDTH` only → :43–44 `LOSS_WIDTH` and
  `OUTLIER_HALFWIDTH`, :46 logged, :51 handed on as `--outlier-halfwidth`
- [x] neurips2026/notes/score_timing.md:29–31 "loss width = outlier half-width … so every score sits at the same
  decision boundary" → :29–33 two quantities, two options, not one boundary
- [x] docs/src/index.md:109 "the sweep's optimal boundary acts as a threshold" → :109–111 "the Student-t has no
  threshold, only the outlier half-width"
- [x] test/native_widths_tests.jl: a `threshold` key or keyword with the Student-t is refused (never read as a
  half-width) and `outlier_halfwidth` reaches the scorer exactly (the refusals already held through
  VisualGeometryToolkitEstimation's key check and the constructor's keywords; the test pins them)

Reviewed, not a hit: `TailRules.pot_threshold_quantile` (renamed from `threshold_quantile` by the owner's later ruling: the POT quantile level of the extreme-value fit, not a
width); `has_decision_boundary`/`decision_boundary` (VisualGeometryToolkitCore's contract: the residual at which the MAP
rule flips, `decision_boundary`, not a half-width); the loss-width families' `threshold` (their loss width, kept);
docs/src/index.md:129 "threshold miscalibration" (the paper's abstract, verbatim; for the owner); the table and
config files of neurips2026 (no Student-t key named a threshold: `snn_threshold` is the SIFT ratio test).

## The paper-structure review (paper-09-rsdr, 2026-11-12)

Every private keying, layout, completion, root and launcher copy of neurips2026 removed in favour of
VisualGeometryToolkitEval's one implementation:

- [x] `stored_dir`/`stored_table` and the configs' `[stored] dir` / `[runs]` (stored runs read from the results root) → each config's `[inputs.<id>]` (tier `neurips2026_paper_runs`, path under `data_root()`, sha256), read by `input_path`/`input_table`, the composition of Eval's `data_root(; tiers)` and `require_input` (723d791)
- [x] the studies' command-line parameters and their output directories under `data_root()/accv2026/results` (score_timing, sample_budget_miscal) and the probe's undefined `DATA_ROOT` → one config of record each, `keyed_cell_dir`, `write_run_records`, `run_complete`, `run_stages` (ec77439)
- [x] scripts/slurm/score_timing.sbatch, sample_budget_miscal.sbatch (the launchers, their POOL/OUT/… environment switches) → configs/reproduce.toml stages and scripts/reproduce/reproduce_results_{dgx,desktop}.sh on Eval's runner (309a69f)
- [x] the deferred-violation allowlist of test/test_paper_project.jl → none: the check runs with no allowance (ec77439, b14bdf7)
