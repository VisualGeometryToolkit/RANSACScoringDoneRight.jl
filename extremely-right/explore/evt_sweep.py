"""Quick exploration: profile-marginal (Jeffreys) score with the outlier
half-width a replaced by an EVT (block-minima) estimate from the complement.

d_g = 1, n_theta = 0 for simplicity (nu = k).  One candidate model:
inlier distances |N(0, sigma^2)|, outlier distances Uniform[0, A].
For a prefix k: outliers = sorted distances k..N-1.  Randomly permute them
(fixed seed per pair), split into B bins, take each bin's minimum D_b.
Locally uniform outliers => D_b ~ Exp(rate = m / A_hat) with m = bin size,
so A_hat = m * mean(D_b)  (MLE).  Score uses a = A_hat (plug-in), or the
Jeffreys marginal over the rate (closed form).
"""
import math, sys
import numpy as np

def jeffreys_fit(k, rss):
    # log Gamma(nu/2) - nu/2 log(pi RSS)   (d_g = 1, nu = k)
    nu = k
    return math.lgamma(nu / 2) - nu / 2 * math.log(math.pi * rss)

def sweep(d, a_fixed=None, B=None, rng=None, kmin=3, margin=0.0):
    """Return argmax k for the fixed-a score and the EVT score."""
    d = np.sort(d); N = len(d); cs = np.cumsum(d ** 2)
    best_fix = (-np.inf, None); best_evt = (-np.inf, None)
    perm_key = rng.random(N)          # fixed random key per point -> bin label
    for k in range(kmin, N - 1):
        rss = cs[k - 1]
        fit = jeffreys_fit(k, rss)
        nO = N - k
        if a_fixed is not None:
            s = fit - nO * math.log(2 * a_fixed)
            if s > best_fix[0]: best_fix = (s, k)
        # EVT: complement, optionally only beyond a margin of the boundary
        comp = d[k:]
        keys = perm_key[k:]
        if margin > 0:
            shat = math.sqrt(rss / k)
            sel = comp > d[k - 1] + margin * shat
            comp, keys = comp[sel], keys[sel]
        nb = min(B, len(comp))
        if nb < 2: continue
        labels = np.floor(np.argsort(np.argsort(keys)) * nb / len(comp)).astype(int)
        mins = np.full(nb, np.inf)
        np.minimum.at(mins, labels, comp)
        m = len(comp) / nb
        A_hat = m * mins.mean()           # MLE of the outlier range from Exp minima
        s = fit - nO * math.log(2 * A_hat)
        if s > best_evt[0]: best_evt = (s, k)
    return best_fix[1], best_evt[1]

def trial(n_in, n_out, sigma, A, a_fixed, B, rng, scale=1.0, margin=0.0):
    din = np.abs(rng.normal(0, sigma, n_in))
    dout = rng.uniform(0, A, n_out)
    d = np.concatenate([din, dout]) * scale
    return sweep(d, a_fixed=a_fixed * scale if a_fixed else None, B=B, rng=rng, margin=margin)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 200
    print("setting                         | fixed-a argmax k (mean/sd) | EVT argmax k (mean/sd)")
    for (n_in, n_out, A, a_fix, B, margin, label) in [
        (300, 700, 100, 50, 20, 0.0, "base, a=A/2 (calibrated), B=20"),
        (300, 700, 100, 5,  20, 0.0, "fixed a 10x too small"),
        (300, 700, 100, 500, 20, 0.0, "fixed a 10x too large"),
        (300, 700, 100, 50, 5,  0.0, "B=5"),
        (300, 700, 100, 50, 80, 0.0, "B=80"),
        (300, 700, 100, 50, 20, 3.0, "EVT with margin 3 s-hat"),
        (30,  970, 100, 50, 20, 0.0, "few inliers (3%)"),
        (300, 700, 10,  5,  20, 0.0, "dense outliers A=10 sigma"),
    ]:
        rf, re = [], []
        for _ in range(T):
            f, e = trial(n_in, n_out, 1.0, A, a_fix, B, rng, margin=margin)
            rf.append(f); re.append(e)
        rf, re = np.array(rf, float), np.array(re, float)
        print(f"{label:32s}| {rf.mean():7.1f} / {rf.std():5.1f}  (true {n_in}) | {re.mean():7.1f} / {re.std():5.1f}")
    # unit invariance: same draws, residuals scaled by 1000
    rng1 = np.random.default_rng(7); rng2 = np.random.default_rng(7)
    same = 0
    for _ in range(50):
        _, e1 = trial(300, 700, 1.0, 100, 50, 20, rng1, scale=1.0)
        _, e2 = trial(300, 700, 1.0, 100, 50, 20, rng2, scale=1000.0)
        same += (e1 == e2)
    print(f"unit invariance (EVT argmax identical after x1000 rescale): {same}/50")
