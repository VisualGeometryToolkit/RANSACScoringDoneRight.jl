import math, numpy as np
from evt_sweep import jeffreys_fit
def sweep_max(d, B, rng, kmin=3):
    d = np.sort(d); N = len(d); cs = np.cumsum(d**2); key = rng.random(N)
    best = (-np.inf, None)
    for k in range(kmin, N-1):
        comp = d[k:]; keys = key[k:]; nb = min(B, len(comp))
        if nb < 2: continue
        lab = np.floor(np.argsort(np.argsort(keys))*nb/len(comp)).astype(int)
        mx = np.full(nb, -np.inf); np.maximum.at(mx, lab, comp)
        m = len(comp)/nb
        a_hat = mx.mean()*(m+1)/m          # unbiased range of a uniform from block maxima
        s = jeffreys_fit(k, cs[k-1]) - (N-k)*math.log(2*a_hat)
        if s > best[0]: best = (s, k)
    return best[1]
rng = np.random.default_rng(0); T = 200
print("setting                    | EVT block-max argmax k (mean/sd)")
for n_in, n_out, A, B, lab in [(300,700,100,20,"base B=20 (true 300)"),(300,700,100,5,"B=5"),(300,700,100,80,"B=80"),
                               (30,970,100,20,"few inliers 3% (true 30)"),(300,700,10,20,"dense outliers A=10sigma")]:
    r=[]
    for _ in range(T):
        d=np.concatenate([np.abs(rng.normal(0,1,n_in)), rng.uniform(0,A,n_out)])
        r.append(sweep_max(d,B,rng))
    r=np.array(r,float); print(f"{lab:27s}| {r.mean():7.1f} / {r.std():5.1f}")
# reference: fixed a = A (the true outlier half-width)
from evt_sweep import sweep
r=[]
for _ in range(T):
    d=np.concatenate([np.abs(rng.normal(0,1,300)), rng.uniform(0,100,700)]); r.append(sweep(d,a_fixed=100,B=20,rng=rng)[0])
print(f"{'reference: fixed a = A':27s}| {np.mean(r):7.1f} / {np.std(r):5.1f}")
