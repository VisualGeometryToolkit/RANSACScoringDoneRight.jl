# REPRODUCE — NeurIPS 2026 "RANSAC Scoring Done Right"

Every figure the paper reads is produced by a driver under `scripts/` from its
config of record under `configs/`, and indexed below. The suite
(`test/runtests.jl`, `paper_project_violations`) checks the index against what
`paper/main.tex` reads, and checks that the drivers reproduce every submitted
fragment byte for byte.

```
# Julia 1.12 (juliaup). The stack is registered in CauRegistry (and submitted to
# General); until it is in General, add both registries once (General first: Pkg
# adds it on its own only to an empty depot):
julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl")'   # once per depot
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/fetch_data.jl            # only on a machine without the data (data/README.md)
julia --project=. scripts/<driver>.jl --plan       # what is done, what is pending
julia --project=. scripts/<driver>.jl              # run the pending stages
make -C paper                                      # the PDF (18 pages)
```

## What is reproduced, and from what

The paper's two figures are drawn from **six evaluated runs** (PT/E, ETH3D/F,
HEB/H × Profile, Sensitivity) produced on 2026-05-06 by the evaluation
pipeline of the time (RobustVisualGeometry `eval/neurips2026`, see "Not
ported"). Each run's evaluated table (`source.jld2`) is kept, with the run's
parameters and the renders of the time, as the stored runs of record:
`<results_root()>/stored/paper-runs-2026-05-06/` (`PROVENANCE.toml`,
`SHA256SUMS`; published as the `neurips2026_paper_runs` artifact).

From those tables the drivers here regenerate every figure fragment of the
paper, with the renderer ported from the one that drew them (`src/panels.jl`).
With the submitted settings **every regenerated fragment was byte-identical to
the submitted one** (stage `verify`, 2026-10-09). Three corrections were then
signed off by the author and regenerated through the drivers (new run keys)
and deployed: the ETH3D Profile cell now draws its confidence band, the HEB
titles read SIFT, and one prose range (the baselines' std at
$|\mathcal{V}| = 2$) is generated from its data ("Changes from the
submitted version" below). `test/test_figures.jl` checks that every fragment
in `paper/figures` is what the drivers render. The computation that produced the
tables — the pools, the coarse probes, the per-pair grid evaluation and the
bootstrap — is not ported; it is described, with the reruns it would take,
under "Not ported".

## Data

Inputs are read from `data_root()` (read-only: `CAU_DATA_ROOT`, else
`data/exp_raw`, else the artifact view of the published data); results are
written under `results_root()` (`CAU_RESULTS_ROOT`, else `data/exp_pro`), one
directory per run key, each with its `_config.toml` and `_provenance.toml`.
The stored runs are a results-side input: the figure configs name them by
`[stored] dir`, relative to `results_root()`. The published data
(`configs/data_artifacts.toml`, `Artifacts.toml`): the stored runs
(`neurips2026_paper_runs`, required; Zenodo record not yet deposited) and the
raw inputs of the paper's pools, shared with LocalOptimizationDoneRight's
accv2026 (`pt_raw_validation`, `pt_raw_test`, `heb_raw`,
`eth3d_splg_csponds`: the same records and hashes).

A regenerated figure never replaces the submitted one silently: a driver
renders into its keyed results directory, `verify` writes `verify.tsv` (byte
identity, coordinate identity, largest coordinate difference), and `deploy`
writes into `paper/figures` only what is identical, unless the config's
`[deploy] replace_submitted = true` records the author's sign-off.

## Asset index

| label | asset | kind | generator | config | data | provenance |
|---|---|---|---|---|---|---|
| `fig:profile-grid` | `paper/figures/PT-E-SIFT-Profile.tex` | generated — ported, verified byte-identical | `scripts/fig_profile.jl` | `configs/fig_profile.toml` | stored run `phototourism/sift/E/Profile/2026-05-06T130222` (`profile`, `profile_oracle`) | `_provenance.toml` of `figures/fig_profile-<key>`; the run's `config.toml` (job 845) |
| `fig:profile-grid` | `paper/figures/ETH3D-F-SPLG-Profile.tex` | generated — ported; REGENERATED with bands, changed from the submitted file (signed off 2026-10-09) | `scripts/fig_profile.jl` | `configs/fig_profile.toml` | stored run `eth3d/splg/F/Profile/2026-05-06T140718` | as above (job 855) |
| `fig:profile-grid` | `paper/figures/HEB-H-SIFT-Profile.tex` | generated — ported; REGENERATED with the title "HEB / SIFT", numbers identical to the submitted file (signed off 2026-10-09) | `scripts/fig_profile.jl` | `configs/fig_profile.toml` | stored run `heb/sift/H/Profile/2026-05-06T130202` | as above (job 843) |
| `fig:profile-grid` | `paper/figures/profile_styles.tex` `paper/figures/profile_styles_shared.tex` | generated — ported, verified byte-identical | `scripts/fig_profile.jl` | `configs/fig_profile.toml` | the HEB Profile run's method set | as above |
| `fig:sensitivity-grid` | `paper/figures/PT-E-SIFT-Sensitivity-eRmean.tex` `paper/figures/PT-E-SIFT-Sensitivity-eRstd.tex` | generated — ported, verified byte-identical | `scripts/fig_sensitivity.jl` | `configs/fig_sensitivity.toml` | stored run `phototourism/sift/E/Sensitivity/2026-05-06T102716` (`summary`) | `_provenance.toml` of `figures/fig_sensitivity-<key>`; the run's `config.toml` (job 844) |
| `fig:sensitivity-grid` | `paper/figures/ETH3D-F-SPLG-Sensitivity-eRmean.tex` `paper/figures/ETH3D-F-SPLG-Sensitivity-eRstd.tex` | generated — ported, verified byte-identical | `scripts/fig_sensitivity.jl` | `configs/fig_sensitivity.toml` | stored run `eth3d/splg/F/Sensitivity/2026-05-06T140300` | as above (job 854) |
| `fig:sensitivity-grid` | `paper/figures/HEB-H-SIFT-Sensitivity-eRmean.tex` `paper/figures/HEB-H-SIFT-Sensitivity-eRstd.tex` | generated — ported; REGENERATED with the title "HEB / SIFT", numbers identical to the submitted file (signed off 2026-10-09) | `scripts/fig_sensitivity.jl` | `configs/fig_sensitivity.toml` | stored run `heb/sift/H/Sensitivity/2026-05-06T084659` | as above (job 842) |
| `fig:sensitivity-grid` | `paper/figures/sensitivity_styles.tex` `paper/figures/sensitivity_styles_shared.tex` | generated — ported, verified byte-identical | `scripts/fig_sensitivity.jl` | `configs/fig_sensitivity.toml` | the HEB Sensitivity run's method set and validation sizes | as above |
| — | `paper/figures/paper_numbers.tex` | generated — new (no submitted copy): `\BaselinesStdVtwoLo`, `\BaselinesStdVtwoHi`, `\StdGapVtwoLo`, `\StdGapVtwoHi` (Sec. 4.4) | `scripts/paper_numbers.jl` | `configs/paper_numbers.toml` | stored runs `phototourism/sift/E/Sensitivity/…`, `eth3d/splg/F/Sensitivity/…` (`summary`, `std_med` at n = 2) | `_provenance.toml` of `numbers/paper_numbers-<key>` |

The Sensitivity fragments are written in the `coordinates {…}` form the paper
carries since its arXiv preparation (paper repository 98714cd converted the
renderer's inline `table` form, same numbers); the renderer emits that form
directly.

## Coverage

Everything the paper states that comes from data, with its status.

| item | where | status | generator |
|---|---|---|---|
| Fig. 1, Profile (3 fragments, 2 style sheets) | Sec. 4.3 | ported + verified (byte-identical, from the stored runs); ETH3D and HEB cells then regenerated with the signed-off corrections | `scripts/fig_profile.jl` |
| Fig. 2, Sensitivity (6 fragments, 2 style sheets) | Sec. 4.4 | ported + verified (byte-identical, from the stored runs); HEB panels then regenerated with the corrected title | `scripts/fig_sensitivity.jl` |
| The evaluated tables behind both figures (`source.jld2`) | — | stored; the computation is NOT ported ("Not ported") | RobustVisualGeometry 6bc825d `eval/neurips2026/evaluate_val.jl` (history) |
| Prose numbers of Sec. 4.3–4.4 (17 checked) | Sec. 4 | all 17 match their data; 13 are literals the driver checks; the baselines' std range is generated and read by `main.tex`; the std gap (`\StdGapVtwoLo`–`\StdGapVtwoHi` = 2× ETH3D – 29× PT) is generated and read by `main.tex` (replacing the unsupported "30–100×"; approved 2026-10-09); the convergence ratios (`\ConvRatioPT` = 128, `\ConvRatioETH` = 8) replace the five "~100×" (approved 2026-10-09) | `scripts/paper_numbers.jl` |
| Tab. `tab:datasets` scene and pair counts (21 / 44 211, 13 / 3 039, 10 / 21 999) and "nearly 70 000 pairs" | Sec. 4.2, App. | literal in `main.tex`; the counts of the pools of record; no generator in the history; not checked (the pools have since been migrated) | — |
| "$n_p \approx 140$" (PT), "median $n_p \approx 10$, ${\sim}4\%$ ratio" (HEB) | Sec. 4.3 | literal; per-pair GT inlier statistics of the pools; no generator found | — |
| Tab. `tab:model-dims` | App. A | static: model dimensions (mathematics, no data) | — |
| Algorithm 1, equations | Sec. 3, App. | static: mathematics | — |

## Changes from the submitted version

Signed off by the author on 2026-10-09; each regenerated through its driver
(a new run key, so a new keyed run directory beside the one that reproduced
the submitted file) and deployed with `[deploy] replace_submitted = true`.

| asset | submitted | now |
|---|---|---|
| `paper/figures/ETH3D-F-SPLG-Profile.tex` | 22 coordinate blocks: mean curves, per-pair-best envelopes, argmin markers, Oracle; no confidence band (the caption states bands for every panel) | 36 blocks: the same 22, unchanged, plus the ±1.96σ bootstrap band of each of the 7 methods (half-widths 0.028°–0.157°, median 0.035°) and its two-line header |
| `paper/figures/HEB-H-SIFT-Profile.tex`, `HEB-H-SIFT-Sensitivity-eRmean.tex`, `HEB-H-SIFT-Sensitivity-eRstd.tex` | title `HEB / RootSIFT / \(\bH\)` | title `HEB / SIFT / \(\bH\)` (HEB ships SIFT matches, Sec. 4.2); every coordinate identical |
| `paper/main.tex` Sec. 4.4, `paper/figures/paper_numbers.tex` | "threshold-based baselines disperse over $0.05^\circ$–$0.35^\circ$" (literal) | `$\BaselinesStdVtwoLo^\circ$--$\BaselinesStdVtwoHi^\circ$` = 0.04°–0.38°, generated: the min / max over RANSAC, MSAC, GaU, MAGSAC++ of the cross-trial std of the median test error at $|\mathcal{V}| = 2$, PT and ETH3D (0.0398° ETH3D GaU, 0.3762° PT RANSAC) |
| `paper/main.tex` Sec. 4.4, `paper/figures/paper_numbers.tex` | "The proposed methods are also far more reproducible … ---a 30--100$\times$ gap." (literal) | "… also more reproducible … ---every baseline at least $\StdGapVtwoLo\times$ (ETH3D) to $\StdGapVtwoHi\times$ (PhotoTourism) more than any proposed variant." = 2× / 29×: per run, the least-dispersed baseline's std over the most-dispersed proposed variant's std at $|\mathcal{V}| = 2$ (PT 0.1377/0.0047, ETH3D 0.0398/0.0187) |
| `paper/main.tex` abstract, Sec. 1, Fig. 2 caption, Sec. 4.4, Sec. 5; `paper/figures/paper_numbers.tex` | "baselines need ${\sim}100\times$ more" (validation data), five times, literal, unqualified | "$\ConvRatioPT\times$ more on PhotoTourism" = 128× (abstract, Sec. 1, Sec. 5); caption and Sec. 4.4 add "($\ConvRatioETH\times$ on ETH3D at a 5\% tolerance)" = 8×. Convergence size: the smallest $|\mathcal{V}|$ at which E[e_R] is within the tolerance of the method's own asymptote; ratio: the baselines' smallest over the proposed methods' largest. PT at 1 % (the paper's "near-optimal"): 256 (MSAC, GaU, MAGSAC++; RANSAC 512) / 2. ETH3D at 1 % shows no advantage (ratio 1), at 5 %: 16 / 2. HEB is not claimed |

The PT titles keep "RootSIFT" (PT ships RootSIFT matches, Sec. 4.2).

## Not ported

The tables the figures are drawn from came out of RobustVisualGeometry's
`eval/` tree at **67d76e2** with uncommitted changes, committed as **6bc825d**
on 2026-05-09 ("Pending neurips2026 eval work"); the tree was later moved to
a since-retired evaluation package (`eval/neurips2026/`, deleted there on
2026-07-18; the bootstrap tuner it used left
RansacScoringDoneRight with it). No Manifest was tracked at that commit. The
chain, per cell (dataset × matcher × problem):

1. **pools** — `eval/generate_pt_scoring_data.jl`, `generate_heb_scoring_data.jl`,
   `generate_roma_scoring_data.jl --problem=fundmat` (1 000 accepted minimal
   samples per pair; configs of record here: `configs/pt_e_pools.toml`,
   `heb_h_pools.toml`, `eth_f_pools.toml`, whose `[source]` also derives the
   published raw-input artifacts). Those pool trees were migrated to later
   pool formats in 2026-09; the May trees are not kept.
2. **coarse probes** — `eval/neurips2026/run_coarse_probes.jl`
   (`coarse_probes.sbatch`): each method's anchor κ⋆ on a coarse log grid,
   written as `coarse_ranges_<protocol>.jld2` beside the pools.
3. **evaluation** — `eval/neurips2026/run_pipeline.jl` → `evaluate_val.jl`
   (8 GPUs, Float32 KernelAbstractions scoring): every method at 200 grid
   points of its two-decade window, per pair; Profile = pair-level bootstrap
   (1 000) of the median error per grid point; Sensitivity = 1 000
   cross-validation trials per validation size with BCa intervals. Output
   `eval_<dataset>_<matcher>_<problem>.jld2` under the dataset's `derived/`
   tree (those files were overwritten after 2026-05-06; the stored runs here
   keep their tables).
4. **render** — `plot_profile.jl` / `plot_val.jl`: PORTED (`src/panels.jl`).

### Reruns to regenerate the tables (planned, not launched)

A rerun replaces the stored runs only through a new keyed run and the
`verify` → sign-off → `deploy` path above. Two routes:

- **Port, then rerun (the route that fits the conventions).** Port steps 1–3
  onto the current stack — the pool producer of VisualGeometryEval (as
  accv2026's `scripts/scoring_pools.jl` does) with the `*_pools.toml` configs
  here, the grid evaluators of this package (`grid_evaluate`,
  `select_on_grid!`) and VisualGeometryEval's `Tuning` for the
  cross-validation — as `scripts/profile.jl` and `scripts/sensitivity.jl`
  writing keyed runs with the same `source.jld2` schema, so `fig_*.jl` reads
  them through a changed `[stored]`/run entry. Expected runtime once ported,
  from the 2026-05-06 jobs (one 8-GPU node): HEB Sensitivity 32 min, HEB
  Profile 7 min, PT Sensitivity 93 min, PT Profile 24 min, ETH3D Sensitivity
  38 min, ETH3D Profile 4 min (≈ 3.3 h), plus the coarse probes (~5–15 min
  per cell) and the pools (hours per dataset; PhotoTourism the longest).
  Bit-identity with the stored tables is not expected (different pools,
  Float32 kernels, RNG streams); the figures would be verified by
  coordinates and the prose numbers by `scripts/paper_numbers.jl`.
- **Historical rerun (a provenance check of the stored tables).** In a
  worktree of RobustVisualGeometry at 6bc825d with an environment resolved
  to that time's stack (no Manifest was tracked), regenerate the pools, then

  ```
  sbatch eval/neurips2026/coarse_probes.sbatch phototourism   # likewise heb, eth3d
  sbatch -J pipe-heb eval/neurips2026/run_pipeline.sbatch heb               # Sensitivity
  sbatch -J prof-heb eval/neurips2026/run_pipeline.sbatch heb --profile     # Profile
  # … phototourism, eth3d (--matcher=splg --problem=F)
  ```

  Same runtimes. It writes into the dataset trees (the data root) and its
  launchers name machine paths, so it is a check, not a route of record.

## Numbers in the prose

`scripts/paper_numbers.jl` (`configs/paper_numbers.toml`) recomputes each
number Sec. 4 states about Fig. 1 and Fig. 2 from the same stored runs,
rounds it as stated, and reports it against the stated value
(`<results>/numbers/paper_numbers-<key>/numbers.tsv`). A number whose entry
names `macros` is generated: written to `paper/figures/paper_numbers.tex`
(through the same verify/deploy stages as the figures), which `main.tex`
reads; the others are literals in the prose. The suite requires every number
to match its data exactly, and the macro file to be what the driver writes
(`test/test_figures.jl`).

## Scoring studies (not paper assets)

`scripts/studies/` holds the package's scoring studies (sample budget under
scale miscalibration, per-hypothesis scoring cost, a Float32 radix-sort
probe), launched by `scripts/slurm/*.sbatch`, with write-ups in `notes/`. The
paper reads none of their outputs. They predate the conventions (command-line
parameters, unkeyed outputs); their deferred violations are listed, with the
fix, in `test/test_paper_project.jl`.
