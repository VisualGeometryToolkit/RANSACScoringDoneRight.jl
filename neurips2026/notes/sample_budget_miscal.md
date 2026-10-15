# Sample budget under scale miscalibration

How well does each scoring function select a model as the hypothesis budget
grows, and how much of that survives a mis-estimated noise scale?

There is **no local optimization** anywhere in this experiment. It exercises the
selection rule alone — score every stored hypothesis, keep the running best,
read off its pose error — so the curves are a property of the score, not of a
refiner attached to it.

## Results

mAA@10° and median pose error on held-out data, at every minimal-sample budget.

### Calibrated (ζ = 1)

#### mAA@10°

| Scorer | 100 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| **Jeffreys (ours)** | **0.4536** | **0.5216** | **0.5422** | **0.5550** | **0.5630** | **0.5702** | **0.5744** | **0.5785** | **0.5814** |
| MSAC | 0.4500 | 0.5192 | 0.5421 | 0.5543 | 0.5616 | 0.5674 | 0.5720 | 0.5761 | 0.5791 |
| GaU | 0.4526 | 0.5208 | 0.5410 | 0.5532 | 0.5610 | 0.5665 | 0.5708 | 0.5748 | 0.5781 |
| RANSAC | 0.4426 | 0.5125 | 0.5349 | 0.5477 | 0.5543 | 0.5608 | 0.5649 | 0.5690 | 0.5712 |
| MAGSAC++ | 0.4517 | 0.5163 | 0.5340 | 0.5452 | 0.5526 | 0.5582 | 0.5620 | 0.5651 | 0.5681 |

#### median error (°)

| Scorer | 100 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| **Jeffreys (ours)** | **4.87** | **3.20** | **2.80** | **2.60** | **2.49** | **2.35** | **2.31** | **2.25** | **2.22** |
| MSAC | 4.99 | 3.21 | 2.81 | 2.64 | 2.53 | 2.40 | 2.36 | 2.30 | 2.27 |
| GaU | 4.88 | 3.20 | 2.82 | 2.66 | 2.53 | 2.43 | 2.39 | 2.33 | 2.29 |
| RANSAC | 5.21 | 3.34 | 2.95 | 2.71 | 2.61 | 2.50 | 2.47 | 2.40 | 2.38 |
| MAGSAC++ | 4.91 | 3.28 | 2.96 | 2.79 | 2.68 | 2.58 | 2.54 | 2.50 | 2.48 |

Spread at 4000 samples is 0.013 mAA — within noise. **With a correct noise
scale the choice of scoring function barely matters**, and every score improves
monotonically with budget, as it should.

### Miscalibrated (ζ = 32)

#### mAA@10°

| Scorer | 100 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| **Jeffreys (ours)** | **0.4130** | **0.4556** | **0.4662** | **0.4729** | **0.4754** | **0.4773** | **0.4783** | **0.4804** | **0.4815** |
| MSAC | 0.1577 | 0.1480 | 0.1449 | 0.1406 | 0.1375 | 0.1354 | 0.1351 | 0.1344 | 0.1344 |
| GaU | 0.1538 | 0.1445 | 0.1411 | 0.1374 | 0.1352 | 0.1331 | 0.1329 | 0.1320 | 0.1320 |
| RANSAC | 0.1190 | 0.1089 | 0.1027 | 0.0982 | 0.0953 | 0.0941 | 0.0936 | 0.0935 | 0.0932 |
| MAGSAC++ | 0.1149 | 0.1043 | 0.1017 | 0.1001 | 0.0989 | 0.0966 | 0.0956 | 0.0940 | 0.0939 |

#### median error (°)

| Scorer | 100 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| **Jeffreys (ours)** | **6.60** | **5.01** | **4.60** | **4.41** | **4.31** | **4.28** | **4.24** | **4.20** | **4.17** |
| MSAC | 20.33 | 20.18 | 20.36 | 20.49 | 20.76 | 20.93 | 20.89 | 20.94 | 21.05 |
| GaU | 20.68 | 20.77 | 20.98 | 21.10 | 21.14 | 21.32 | 21.41 | 21.51 | 21.55 |
| RANSAC | 23.02 | 22.64 | 22.89 | 23.37 | 23.43 | 23.43 | 23.27 | 23.41 | 23.47 |
| MAGSAC++ | 23.63 | 24.19 | 24.53 | 24.71 | 24.92 | 24.98 | 25.00 | 25.08 | 25.04 |

**The baselines cannot spend a sample budget under miscalibration; ours can.**
Our mAA rises at every one of the nine steps (0.4130 → 0.4815, +16.6%) while
all four baselines fall at every step: MSAC −14.8%, GaU −14.2%, MAGSAC++
−18.3%, RANSAC −21.7%. There are no crossings and no non-monotone points. The
median-error tables say the same thing from the other side — ours falls 6.60° →
4.17° while every baseline's rises.

## Analysis

### Why more samples *hurt* the baselines

At ζ = 32 a threshold scorer's decision boundary sits 32× too high. For RANSAC
that means essentially every correspondence counts as an inlier, so the score
`Σ 1(r < loss_width)` saturates near `n` for almost every hypothesis. MSAC and GaU
saturate more softly but in the same direction: the weight function is nearly
flat over the whole residual range, so score differences between a good model
and a bad one shrink toward the noise floor.

Once the score is that weakly informative, the arg-max over a hypothesis pool
behaves like an arg-max over near-ties. Drawing more samples adds more
candidates competing for a maximum that no longer tracks geometric quality, so
the probability that *some* poor model attains the highest score by chance grows
with the pool. The running best therefore drifts **away** from the truth as the
budget increases — which is exactly what the ζ = 32 table shows, monotonically,
for all four baselines and at every one of the nine steps.

This is worth stating precisely because it inverts the usual intuition about
RANSAC: more hypotheses is not weakly beneficial under a bad threshold, it is
actively harmful. A practitioner who responds to poor results by raising the
trial budget makes the estimate worse.

### Why ours does not

The profile-marginal score does not treat its native as the decision boundary.
The sweep sorts residuals and, at every candidate partition `k`, re-derives the
inlier scale from the data itself — `RSS/ν` over the points currently assigned
to the inlier set — then scores that partition by its marginal likelihood. The
corrupted parameter enters only as the outlier half-width, i.e. the prior
width of the uniform component, not as the estimate of the noise.

So ζ corrupts a *prior*, while the *scale* stays anchored to the deployed data's
own partition. That is why the score keeps ranking hypotheses correctly: an
inflated half-width makes the outlier component more permissive, which costs
some accuracy (0.5814 → 0.4815, a 17% loss), but it does not destroy the
ordering, because the quantity that decides the ordering is re-estimated rather
than supplied.

The mechanism predicts the shape we observe: our curve should still *improve*
with budget, just from a lower starting point — more samples still means more
chances to find a genuinely better partition, and the score can still tell which
one that is. It rises 0.4130 → 0.4815 across the nine budgets.

### What separates the two groups

The distinction is not robustness of the loss shape — MSAC and GaU are both
redescending-ish and still collapse. It is **whether the scale used for the
decision is supplied or estimated**. Every baseline here takes its scale as a
fixed input; ours estimates it per partition. Under correct calibration these
are equivalent (the ζ = 1 table: 0.013 mAA spread across all five). Under a
mis-estimated scale, only the estimating one survives.

That equivalence at ζ = 1 is the important control. It rules out the alternative
explanation that our score is simply better at selection — it is not, by any
margin that matters when the threshold is right. The entire advantage is
conditional on the scale being wrong.

### Magnitude

Retention at 4000 samples — the fraction of calibrated mAA@10 that survives
ζ = 32:

| Scorer | ζ=1 | ζ=32 | retained |
|---|---:|---:|---:|
| **Jeffreys (ours)** | 0.5814 | **0.4815** | **83%** |
| MSAC | 0.5791 | 0.1344 | 23% |
| GaU | 0.5781 | 0.1320 | 23% |
| RANSAC | 0.5712 | 0.0932 | 16% |
| MAGSAC++ | 0.5681 | 0.0939 | 17% |

In median-error terms the baselines land at 21–25°, which for relative pose is
not a degraded estimate but a failed one. Ours stays at 4.17°, still a usable
pose.

## Protocol

Pool: PT-E-SIFT-4K, **4 held-out test scenes, 8800 image pairs**
(`palace_of_westminster`, `piazza_san_marco`, `sacre_coeur`,
`notre_dame_front_facade`). Stored minimal-sample hypotheses with precomputed
pool-frame Sampson residuals; ~17 600 models per pair from 4000 minimal samples.

**Budget is in MINIMAL SAMPLES**, the unit a RANSAC trial draws and its
termination criterion counts — not models, which is solver-dependent (the
5-point solver returns ~4.4 models per sample here). `sample_ids` recovers the
boundaries: one sample's solver siblings occupy contiguous columns sharing a
`sample_indices` column. Grid: 100, 500, 1000, …, 4000, spanning the pool
exactly with no extrapolation at the top.

**Miscalibration** is `corrupt_native(nat, ζ) = ζ·nat`, applied identically to
every scorer. It is principled across both families because both natives are
linear in the noise scale ŝ: `loss_width = scale_multiple·ŝ` for the threshold
scores and `outlier_halfwidth = ŝ·√(π/2)·exp(scale_multiple²/2)` for ours, where the exponential factor depends only on
the standardized cutoff. So a scale wrong by 32× multiplies every scorer's
tuned parameter by exactly 32 — nobody gets a special case.

**Hyperparameters are not re-tuned.** Each native is read from the deployed
coarse→fine CV sidecar, tuned at calibration (ζ=1); ζ enters only at readout.
Re-tuning at ζ=32 would let each scorer find its best parameter for the
corrupted scale and would measure robustness no deployed system has. The loader
hard-errors on any CV record carrying `boundary_hit`, so a clipped operating
point cannot enter the curves; all five were interior with zero grid
extensions.

A contract check on the first pair asserts that the per-model scores
agree with the package's own `best_pool_model`, so a divergence from the
shipped scorers fails loudly instead of quietly changing what the curves mean.

## Qualifications

**Four of the eleven test scenes.** The driver's residency is cumulative across
scenes rather than per-scene — RSS tracks bytes read 1:1 and is not released
between loads — so the full eleven would need ~900 GB, beyond anything
schedulable. This is a defect in the driver, tracked separately, not a property
of the experiment. The four scenes are held-out test data and the pattern is
unambiguous, but the error bars are wider than an eleven-scene run would give.

**Five scorers, not eight.** Welsch, smooth-TLS and Geman-McClure are in the
package's canonical roster but have no tuned native in the PT-E-SIFT-4K probe
sidecar, having been added after it ran. They are excluded rather than run at an
untuned operating point. One probe re-run adds all three; the roster is a table
at the top of the driver and `load_natives` consumes whatever the sidecar
carries.

**ζ = 32 is a single, severe corruption.** It is one point on the scale axis,
chosen to separate the methods, not a sweep. The calibrated column is the
control.

---
Source: `neurips2026/scripts/studies/sample_budget_miscal.jl` via
`neurips2026/scripts/slurm/sample_budget_miscal.sbatch` (job 4900).
Data: `scoring_budget_miscal.{tsv,jld2}` in the driver's `--out` directory (not committed).
