# CLAUDE.md

neurips2026: the paper project of RansacScoringDoneRight ("RANSAC Scoring Done
Right", NeurIPS 2026). It follows the stack's paper-project conventions —
VisualGeometryToolkitEval README, "Paper project conventions"; LocalOptimizationDoneRight's
accv2026 is the worked example — and `test/runtests.jl` enforces them
(`VisualGeometryToolkitEval.PaperProjects.paper_project_violations`). In short:

- every result and paper asset comes from a driver under `scripts/`
  (`run_stages`: `--plan`, `--stage=NAME`, `--force`) reading a TOML config of
  record under `configs/`; no data constant (seed, scene or frame list, data
  path, numeric table) lives in a driver;
- outputs go under `results_root()` in a directory keyed by `run_key(cfg)`,
  with `write_run_records` / `write_provenance` beside them; inputs under
  `data_root()` are read-only; results are add-only (move a superseded run
  aside, never delete it); no machine path, no `artifact"…"` in code;
- every figure the paper reads is a row of `REPRODUCE.md`;
- a regenerated figure never replaces the submitted one silently: a driver
  renders into its keyed results directory, `verify` compares it with
  `paper/figures` (bytes and coordinates), and `deploy` writes only what is
  identical unless the config's `[deploy] replace_submitted` records the
  author's sign-off;
- published data are Julia artifacts on Zenodo, catalogued in
  `configs/data_artifacts.toml` (the raw inputs are shared with accv2026).

The manuscript's prose and numbers are the author's: never edit them to match
a regenerated result; report the difference (`scripts/paper_numbers.jl`
checks the prose numbers against their data).

The package
itself is a library and never depends on this project. Run the tests with
`julia --project=. test/runtests.jl`.
