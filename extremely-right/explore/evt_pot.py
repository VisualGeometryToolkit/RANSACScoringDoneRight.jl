"""POT (peaks over threshold) estimate of the outlier half-width a, threshold =
a quantile q of the complement, GPD fitted by probability-weighted moments
(Hosking & Wallis 1987).  Compared with block maxima and the fixed true a."""
import math, numpy as np
from evt_sweep import jeffreys_fit

def gpd_pwm(y):
    """Hosking-Wallis PWM fit of a GPD to excesses y >= 0.
    Returns (xi, beta) in the F(y) = 1 - (1 + xi y/beta)^(-1/xi) convention."""
    y = np.sort(y); n = len(y)
    p = (np.arange(1, n + 1) - 0.35) / n
    a0 = y.mean(); a1 = np.mean((1 - p) * y)
    k = a0 / (a0 - 2 * a1) - 2.0           # Hosking's k = -xi
    beta = 2 * a0 * a1 / (a0 - 2 * a1)
    return -k, beta

def a_pot(comp, q):
    u = np.quantile(comp, q); y = comp[comp > u] - u
    if len(y) < 5: return comp.max(), np.nan
    xi, beta = gpd_pwm(y)
    end = u + beta / (-xi) if xi < -1e-3 else comp.max()   # bounded tail -> endpoint
    return max(end, comp.max()), xi

def a_bmax(comp, B, keys):
    nb = min(B, len(comp))
    lab = np.floor(np.argsort(np.argsort(keys)) * nb / len(comp)).astype(int)
    mx = np.full(nb, -np.inf); np.maximum.at(mx, lab, comp); m = len(comp) / nb
    return mx.mean() * (m + 1) / m

def sweep(d, est, rng, kmin=3, step=1):
    d = np.sort(d); N = len(d); cs = np.cumsum(d ** 2); keys = rng.random(N)
    best = (-np.inf, None); xis = []
    for k in range(kmin, N - 10, step):
        comp = d[k:]
        if est[0] == "fixed": a = est[1]
        elif est[0] == "pot": a, xi = a_pot(comp, est[1]); xis.append(xi)
        else: a = a_bmax(comp, est[1], keys[k:])
        s = jeffreys_fit(k, cs[k - 1]) - (N - k) * math.log(2 * a)
        if s > best[0]: best = (s, k)
    return best[1], (np.nanmedian(xis) if xis else np.nan)

def draw(rng, kind, n_in, n_out, A):
    din = np.abs(rng.normal(0, 1, n_in))
    if kind == "uniform": dout = rng.uniform(0, A, n_out)
    else:  # decaying density: half-triangular on [0, A] (density 2(A-x)/A^2), denser near the model
        dout = A * (1 - np.sqrt(rng.uniform(0, 1, n_out)))
    return np.concatenate([din, dout])

if __name__ == "__main__":
    rng = np.random.default_rng(1); T = 100
    cases = [("uniform", 300, 700, 100, "uniform, 30% inliers"),
             ("uniform", 30, 970, 100, "uniform, 3% inliers"),
             ("uniform", 300, 700, 10, "uniform, dense (A=10 sigma)"),
             ("decay", 300, 700, 100, "decaying outlier density")]
    ests = [("fixed", 100.0), ("bmax", 20), ("pot", 0.80), ("pot", 0.90), ("pot", 0.95)]
    print(f"{'case':30s}| true | " + " | ".join(f"{e[0]}:{e[1]}".ljust(16) for e in ests))
    for kind, n_in, n_out, A, lab in cases:
        row = []
        for e in ests:
            e2 = ("fixed", float(A)) if e[0] == "fixed" else e
            ks, xs = [], []
            for _ in range(T):
                k, xi = sweep(draw(rng, kind, n_in, n_out, A), e2, rng)
                ks.append(k); xs.append(xi)
            ks = np.array(ks, float)
            cell = f"{ks.mean():5.0f}±{ks.std():3.0f}"
            if e[0] == "pot": cell += f" ξ{np.nanmedian(xs):+.2f}"
            row.append(cell.ljust(16))
        print(f"{lab:30s}| {n_in:4d} | " + " | ".join(row))
    # unit invariance for POT
    r1 = np.random.default_rng(5); r2 = np.random.default_rng(5); same = 0
    for _ in range(30):
        d1 = draw(r1, "uniform", 300, 700, 100); d2 = draw(r2, "uniform", 300, 700, 100) * 1000
        same += sweep(d1, ("pot", 0.9), r1)[0] == sweep(d2, ("pot", 0.9), r2)[0]
    print(f"POT q=0.9 unit invariance (x1000): {same}/30")
