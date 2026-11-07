# CLAUDE.md

cvpr2027: the paper project of RANSACScoringDoneRight for "RANSAC Scoring
Done Extremely Right" (CVPR 2027), the sibling of `../neurips2026`. It follows
the stack's paper-project conventions — VisualGeometryToolkitEval README,
"Paper project conventions"; LocalOptimizationDoneRight's accv2026 is the
worked example, neurips2026 the sibling — and `test/runtests.jl` enforces them
(`VisualGeometryToolkitEval.PaperProjects.paper_project_violations`). In short:

- every result and paper asset comes from a driver under `scripts/`
  (`run_stages`: `--plan`, `--stage=NAME`, `--force`) reading a TOML config of
  record under `configs/`; no data constant (seed, scene or frame list, data
  path, numeric table) lives in a driver;
- outputs go under `results_root()` in a directory keyed by `run_key(cfg)`,
  with `write_run_records` / `write_provenance` beside them; inputs under
  `data_root()` are read-only; results are add-only; no machine path, no
  `artifact"…"` in code;
- every table, figure and macro the paper reads is a row of `REPRODUCE.md`;
- published data are Julia artifacts on Zenodo, catalogued in
  `configs/data_artifacts.toml`; the real-data pools are neurips2026's.

The manuscript's prose and numbers are the author's: never edit them to match
a regenerated result; report the difference.

Julia only: the experiments are Julia drivers run through kaimon. The score —
the outlier models, the label priors, the generalized Pareto fit, the sweep —
is the package's (`../src`, StudentTUniformScoring); `src/simulation.jl` holds
only the synthetic generative model, the block baselines, the soft-label
formulation and the metrics, and never re-implements the package.

The package itself is a library and never depends on this project. Run the
tests with `julia --project=. test/runtests.jl`.
