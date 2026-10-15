# Per-hypothesis scoring cost on CPU

Cost of scoring a single model hypothesis, in microseconds, for our
profile-marginal sort-and-sweep and four baseline scoring rules. Every scorer is
an implementation shipped in this package; the driver
(`neurips2026/scripts/studies/score_timing.jl`) contains no scoring code of its own.

## Results

| Method | Form | typical (n̄ = 715) | dense (n̄ = 1580) |
|---|---|---:|---:|
| RANSAC | exact | 0.65 | 1.43 |
| MSAC | exact | 0.84 | 1.88 |
| **MAGSAC++** | **lookup table** | **1.83** | **5.00** |
| **Ours (sort-and-sweep)** | exact | **9.62** | **21.59** |
| GaU | exact | 12.38 | 28.49 |
| MAGSAC++ | closed form | 29.73 | 92.99 |

**Our score is mid-pack, and it is not the cost leader.** It is 3.1× cheaper
than MAGSAC++ evaluated in closed form and 1.3× cheaper than GaU, but 4.3×
*more expensive* than MAGSAC++ evaluated through its lookup table, and 15×
RANSAC. We report this plainly because the tabulated form is what MAGSAC++
actually deploys (see below), so it is the honest comparison.

## Protocol

Single-threaded (`julia -t 1`). Pool: PT-E-SIFT-4K, scene `british_museum`,
2200 image pairs of stored minimal-sample hypotheses with precomputed pool-frame
Sampson residuals. All methods share one operating point, loss width = outlier
half-width = 1.64 × 10⁻³ on
the residual-unit axis, so every score sits at the same decision boundary. One
warm-up run per scorer precedes the bracket, so compilation is excluded; the
reported figure is the **minimum of three repetitions**, each a bracket over 20
pairs divided by the hypotheses scored, making it budget-independent.

Two density regimes, because per-hypothesis cost scales with the correspondence
count: a lower-density sample (n̄ = 715 points/pair, 342 625 hypotheses) and the
twenty densest pairs (n̄ = 1580, 335 213 hypotheses).

Workspaces are sized **per problem**. This matters: our sweep takes an O(n)
radix sort only when the permutation buffer matches n exactly, and otherwise
falls back to a comparison `sortperm!`. `RobustVisualGeometry`'s RANSAC resizes
every sortable buffer to n for precisely this reason, so per-problem sizing is
what a deployment pays.

## MAGSAC++ verification

Our MAGSAC++ was cross-checked against the paper (Barath, Noskova, Ivashechkin &
Matas, §2.2) and against the reference implementation at
`github.com/danini/magsac` (`src/pymagsac/include/magsac.h`,
`getModelQualityPlusPlus`). It agrees with both on every substantive point:

| quantity | paper | `danini/magsac` | ours |
|---|---|---|---|
| ρ second term | r²/4 | `squared_residual / 4.0` | r²/4 |
| σ_max from the loss width | τ* = k·σ_max | `threshold_to_sigma_multiplier = 1/k` | `loss_width / _MAGSAC_SCALE_MULTIPLE` |
| truncation | k·σ_max | `residual > maximum_threshold` | `d < loss_width` |
| quality | Q = 1/L | `score_ = 1.0 / total_loss` | equivalent arg-best |

Both forms in the table compute this same ρ. The closed form evaluates the
incomplete gammas per residual; the lookup form reads them from a 10 000-entry
table spanning R ∈ [0, k²/2), indexed by `round` and clamped, with the lower and
upper gammas interleaved so one residual touches a single cache line. On the
five densest pairs the two select **identical models** (0 disagreements), so the
table is a drop-in, not an accuracy trade.

**The reference implementation ships the lookup table.** The paper prescribes it
— *"the precise calculation can be done very fast by storing the values of the
complete and incomplete gamma functions in a lookup table … merely a few
operations per point"* — and `danini/magsac` reads
`stored_lower_incomplete_gamma_values[x]` and `stored_complete_gamma_values[x]`.
Reporting only the closed form would overstate MAGSAC++'s cost by ~18×.

We also examined SupeRANSAC's `magsac_scoring.h`, which deviates from both the
paper and the reference on three points: it uses a constant σ_max²/4 where the
paper has per-residual r²/4; it sets σ_max = threshold rather than threshold/k;
and it truncates at σ_max instead of k·σ_max, severing the weight at ~80% of its
peak instead of where it reaches zero. We follow the paper and the reference.

## Reading the table

Cost is governed by **what each score evaluates per residual**, not by
asymptotic complexity. RANSAC (a comparison) and MSAC (a multiply–add) are
cheapest. Tabulated MAGSAC++ joins them, because a table read plus a few
multiplies is comparable work. The expensive scorers are those evaluating a
special function per point: a softplus for GaU, two incomplete gammas for
closed-form MAGSAC++.

Our score is the only one that **sorts** — the sweep needs the order statistics
of the residuals — yet it still costs less than either closed-form special
function. The sort is an LSD radix over monotone unsigned keys: one fused
histogram scan plus at most four byte iterations at the default `UInt32` key width,
with trivial digits skipped. It is Θ(n), not Θ(n log n).

Scaling confirms this. Over 2.21× more points (715 → 1580) our cost grows 2.24×,
i.e. linearly, since both the radix sort and the sweep are Θ(n). Closed-form
MAGSAC++ grows 3.13×, superlinearly, because its truncation guard means only
inliers pay for the incomplete gammas and denser pairs hold proportionally more
of them.

## Qualifications

**These are scoring costs given residuals.** A deployed RANSAC trial also
computes each hypothesis's residuals from the model; this measurement reads
stored residuals so every scorer sees identical data. That cost is
scorer-independent and cancels in all ratios above, but the absolute figures
understate a full trial.

**Only our score currently runs inside our RANSAC.** The framework dispatches on
the marginal scoring type; the baselines are evaluated over the same stored
hypotheses. The comparison is like-for-like as a measurement of scoring cost,
but it is not a comparison of complete estimators.

**The lower-density regime is the first twenty pairs in pool storage order**,
not a median-density sample. Read it as an example of a less dense regime, not a
claim about the pool's central tendency.

---
Source: `neurips2026/scripts/studies/score_timing.jl` via `neurips2026/scripts/slurm/score_timing.sbatch` (job 4841).
Data: `$CAU_DATA_ROOT/accv2026/results/score_timing/score_timing_exact_british_museum.jld2` (not committed).
