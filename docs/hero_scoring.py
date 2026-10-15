# Renders docs/src/assets/landing/hero_scoring_{light,dark}.png (the landing hero, one
# variant per page theme), cropped to the non-transparent extent. Needs matplotlib + numpy + scipy (e.g. a pixi env); not part
# of the build. Run from docs/:  python hero_scoring.py
"""Hero image: a line hypothesis scored by the scale-free marginal score.

Top: noisy points around a line model, with gross outliers. Bottom: the same
points' residuals sorted ascending (the sort-and-sweep), with the Jeffreys
marginal score S(k) of every candidate partition drawn over them; its maximum
selects the inlier set. Points are coloured from red (outlier) to green (inlier)
by their posterior inlier probability at the selected partition's scale
estimate, with size and opacity as a second cue, so no threshold and no inlier
scale enters anywhere. Transparent background for light/dark themes.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gammaln

rng = np.random.default_rng(11)

# --- data: a line y = m x + c, Student-t-ish inliers, uniform outliers -------------
m, c = 0.42, 0.05
n_in, n_out = 150, 90
x_in = rng.uniform(-1.0, 1.0, n_in)
d_in = 0.035 * rng.standard_t(5, n_in)                 # signed perpendicular offsets
nrm = np.array([-m, 1.0]) / np.hypot(m, 1.0)            # unit normal of the line
p_in = np.stack([x_in, m * x_in + c], -1) + d_in[:, None] * nrm
p_out = np.stack([rng.uniform(-1.0, 1.0, n_out), rng.uniform(-0.75, 0.85, n_out)], -1)
P = np.vstack([p_in, p_out])
r = (P - np.array([0.0, c])) @ nrm                      # perpendicular residual (d_g = 1)

# --- the scale-free marginal score of every sorted partition (Jeffreys prior) ------
N, dg, n_theta = len(r), 1, 2                          # a line: 2 parameters, codim 1
a = 1.0                                                 # outlier half-width
order = np.argsort(r ** 2)
rss = np.cumsum(r[order] ** 2)
k = np.arange(1, N + 1)
nu = k * dg - n_theta
S = np.full(N, -np.inf)
ok = nu > 0
S[ok] = (gammaln(nu[ok] / 2) - (nu[ok] / 2) * np.log(np.pi * rss[ok])
         - (N - k[ok]) * dg * np.log(2 * a))
k_star = int(np.argmax(S)) + 1
s_hat = np.sqrt(rss[k_star - 1] / (k_star * dg - n_theta))

# posterior inlier probability at the partition's scale (Gaussian inlier, uniform outlier)
pi_in = k_star / N
lik_in = np.exp(-0.5 * (r / s_hat) ** 2) / (np.sqrt(2 * np.pi) * s_hat)
post = pi_in * lik_in / (pi_in * lik_in + (1 - pi_in) / (2 * a))

# red (outlier) -> desaturated grey -> green (inlier); marker size and opacity
# also grow with the probability, a second cue for red-green colour blindness.
# Two variants on a transparent background, one per page theme: the dark one
# lightens the model colour and the extremes of the point colormap, and
# raises the band opacity, so they read on a dark page.
VARIANTS = {
    "light": dict(red="#d23c4b", grey="#a3a8ad", green="#2f9e44", deep="#2b8a3e",
                  model="#2b5d7d", band=1.0, out_alpha=0.6),
    "dark":  dict(red="#f06672", grey="#8a9096", green="#51cf66", deep="#69db7c",
                  model="#8ec5e8", band=1.4, out_alpha=0.8),
}


def render(name, red, grey, green, deep, model, band, out_alpha):
    cmap = LinearSegmentedColormap.from_list("post", [red, grey, green])
    fig = plt.figure(figsize=(8.0, 7.2), dpi=170)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.03, 0.34, 0.94, 0.64])
    bx = fig.add_axes([0.03, 0.03, 0.94, 0.30])
    for a_ in (ax, bx):
        a_.set_facecolor((0, 0, 0, 0))
        a_.set_axis_off()

    # top: the model, a soft band at the estimated scale, the points
    xs = np.linspace(-1.08, 1.08, 2)
    for w, al in ((3.0, 0.10), (2.0, 0.14), (1.0, 0.20)):
        off = w * s_hat * np.hypot(m, 1.0)
        ax.fill_between(xs, m * xs + c - off, m * xs + c + off, color=model,
                        alpha=al * band, lw=0)
    ax.plot(xs, m * xs + c, color=model, lw=2.2, solid_capstyle="round")
    o = np.argsort(post)
    ax.scatter(P[o, 0], P[o, 1], c=post[o], cmap=cmap, vmin=0, vmax=1,
               s=10 + 28 * post[o], alpha=out_alpha + (0.95 - out_alpha) * post[o],
               edgecolors="none")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.72, 0.88)
    ax.set_aspect("equal", adjustable="box")

    # bottom: sorted |r| as bars, and the score curve over the partitions
    ab = np.abs(r[order])
    cap = 0.35
    bx.bar(k, np.minimum(ab, cap), width=0.9, color=cmap(post[order]), lw=0, alpha=0.9)
    Sk = S[ok]
    Sn = (Sk - Sk.min()) / (Sk.max() - Sk.min())
    bx.plot(k[ok], 0.04 + 0.34 * Sn, color=model, lw=2.0)
    bx.axvline(k_star + 0.5, color=deep, lw=1.4, ls=(0, (3, 2)))
    bx.plot([k_star], [0.04 + 0.34], "o", color=model, ms=6)
    bx.axhline(0, color=model, lw=0.8, alpha=0.5)
    bx.set_xlim(0, N + 1); bx.set_ylim(-0.01, 0.40)

    fig.savefig(f"src/assets/landing/hero_scoring_{name}.png", transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


for name, kw in VARIANTS.items():
    render(name, **kw)
print("k* =", k_star, "of", N, "(true inliers:", n_in, ") s_hat =", round(s_hat, 4))
