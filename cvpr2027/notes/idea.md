# Eliminating the outlier half-width a

## Idea
Replace the fixed outlier half-width `a` in the profile-marginal score by an
estimate from the complement of the current prefix (the points it calls
outliers). With Jeffreys on σ, the score then has no length parameter:
rescaling every residual by c shifts each partition's score by
−(N d_g − n_θ) log c, so the ranking is unit-invariant.

## Estimators tried (simulation, d_g = 1, n_θ = 0, 100-200 trials)
Setup: 300 inliers |N(0,1)|, 700 outliers Uniform[0,100]; argmax k vs true 300.

| estimator of a | result |
|---|---|
| fixed true a = A (reference) | 320 ± 5 |
| block MINIMA of a random binning of the complement | collapses to k_min ≈ 3 in every setting, including B = 5..80 and a 3ŝ margin |
| block MAXIMA (mean bin max · (m+1)/m) | 321 ± 5; insensitive to B |
| POT, threshold = q-quantile of the complement, GPD by PWM | 322 ± 5; insensitive to q ∈ [0.8, 0.95]; deterministic |

- Minima collapse: at short prefixes the complement holds the true inliers, so
  a-hat → 0 and −n_O log(2 a-hat) rewards calling everything an outlier.
- Unit invariance holds exactly for all variants (30/30, 50/50 trials).
- The fitted GPD shape ξ is a diagnostic: ≈ −1 for uniform outliers, −0.47
  for a linearly decaying outlier density (theory −1/2).
- The hard cases (3% inliers; dense outliers A = 10σ; decaying density)
  overshoot identically with the fixed true a: failures of the score's
  Gaussian–uniform model, not of the estimator.

## Next
1. Cost: compute the tail from the top fraction of ALL distances (one fit per
   model) so the sweep stays O(N); check it matches the per-prefix fit.
2. A density-matched a (outlier density near the model, not the endpoint)
   for non-uniform outliers.
3. Real pools (PhotoTourism E, ETH3D F, HEB H): Jeffreys + POT vs Jeffreys +
   CV-tuned a, vs a fixed at the image extent, vs AC-RANSAC, MAGSAC++,
   SIMFIT++. One setting, no retuning.
4. Venue: check the CVPR 2027 deadline; ECCV/BMVC as alternatives.
