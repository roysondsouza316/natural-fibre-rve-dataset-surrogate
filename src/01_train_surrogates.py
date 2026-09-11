"""
Surrogate training and global sensitivity analysis
--------------------------------------------------
The sensitivity question is posed over CONTINUOUS uniform ranges of the
design window (log-uniform for fibre length), per length-distribution
type, so the learned surrogate is required to answer it. Sobol indices
are estimated with Saltelli/Jansen estimators on quasi-random Sobol'
samples with bootstrap confidence intervals.

Verification chain
  1. Exact 5-factor ANOVA per distribution on the factorial grid
     (computed inline; discrete uniform over the design levels).
  2. The same Saltelli estimator restricted to the grid levels must
     reproduce (1) within estimator noise -> the surrogate and the
     estimator are faithful where the truth is known exactly.
  3. The continuous-range indices are then the headline numbers.

Waviness is parameterised by the mean deviation inclination tdeg
(0, 18.2, 25.8, 36.9 deg), which is monotone in physical waviness;
the raw kappa_theta feature is NOT (0 means straight). Length enters
the feature vector as log10(L).

02_sobol_indices.py then repeats the continuous block with the centred
first-order estimator and a larger sample; those outputs are the
numbers reported in the manuscript.

Outputs (out/sensitivity/)
  surrogates_tdeg.joblib      HGB + GP per distribution, tdeg feature
  sobol_continuous.csv        per-dist S1/ST (+CI) over continuous ranges
  sobol_table6.csv            6-factor headline table (Dist as group)
  sobol_grid_check.csv        exact grid ANOVA vs Saltelli-on-grid
  sobol_conditional.csv       3-factor indices at the anchor compositions
  sobol_gp_check.csv          GP cross-check of the continuous indices
  sobol_s2.csv                second-order (closed) pair indices
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import qmc
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(11)
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
DATA = HERE / 'out' / 'dataset.csv'

TARGETS = ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']
FACTORS = ['Vf', 'EfEm', 'alpha', 'logL', 'tdeg']
NBOOT = 300
N_CONT = 2 ** 13          # base sample, continuous run
N_GRID = 2 ** 13          # base sample, grid-restricted validation
N_GP = 2 ** 10            # base sample, GP cross-check
ANCHORS = [(0.10, 12.0), (0.20, 28.0), (0.30, 48.0)]


def theta_deg(kappa):
    if kappa == 0:
        return 0.0
    return float(np.degrees(np.arccos(1 / np.tanh(kappa) - 1 / kappa)))


TDEG = {0: 0.0, 5: theta_deg(5), 10: theta_deg(10), 20: theta_deg(20)}

# ----------------------------------------------------------------------
# data and surrogates (tdeg + log10 L features)
# ----------------------------------------------------------------------
df = pd.read_csv(DATA)
df['tdeg'] = df['theta'].map(lambda k: TDEG[int(k)])
df['logL'] = np.log10(df['length_value'])
FEAT_SRC = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio',
            'logL', 'tdeg']

BOUNDS = {          # continuous ranges of the design window
    'Vf': (0.10, 0.30),
    'EfEm': (4.0, 48.0),
    'alpha': (1.0, 3.0),
    'logL': (np.log10(50.0), np.log10(200.0)),   # log-uniform in L
    'tdeg': (0.0, TDEG[5]),
}
LEVELS = {
    'Vf': np.array([0.10, 0.15, 0.20, 0.25, 0.30]),
    'EfEm': np.array([4., 12., 20., 28., 32., 40., 48.]),
    'alpha': np.array([1., 2., 3.]),
    'logL': np.log10(np.array([50., 100., 200.])),
    'tdeg': np.array([TDEG[0], TDEG[20], TDEG[10], TDEG[5]]),
}

print('training surrogates (tdeg + log10 L features) ...')
bundle = {}
for dist in ('constant', 'exponential'):
    sub = df[df.length_distribution_type == dist]
    X = sub[FEAT_SRC].values
    hgb = {}
    gp = {}
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    for t in TARGETS:
        y = sub[t].values
        m = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, min_samples_leaf=5,
            random_state=0).fit(X, y)
        hgb[t] = m
        kern = (ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(length_scale=np.ones(5),
                      length_scale_bounds=(0.3, 5.0))
                + WhiteKernel(1e-6, (1e-9, 1e-2)))
        g = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                     random_state=0).fit(Xs, y)
        gp[t] = g
    bundle[dist] = {'hgb': hgb, 'gp': gp, 'scaler': scaler}
joblib.dump(bundle, OUT / 'surrogates_tdeg.joblib')
print('  saved', OUT / 'surrogates_tdeg.joblib')
# light copy with the HGB models only, used by predict.py
light = {d: {'hgb': bundle[d]['hgb'], 'scaler': bundle[d]['scaler']}
         for d in bundle}
joblib.dump(light, OUT.parent / 'surrogate_hgb.joblib', compress=3)
print('  saved', OUT.parent / 'surrogate_hgb.joblib')


# ----------------------------------------------------------------------
# Saltelli machinery (hand-rolled; Sobol' sequence from scipy.stats.qmc)
# ----------------------------------------------------------------------
def saltelli_matrices(n, k, seed):
    s = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    U = s.random(n)
    return U[:, :k], U[:, k:]


def map_continuous(U):
    X = np.empty_like(U)
    for j, f in enumerate(FACTORS):
        lo, hi = BOUNDS[f]
        X[:, j] = lo + U[:, j] * (hi - lo)
    return X


def map_grid(U):
    X = np.empty_like(U)
    for j, f in enumerate(FACTORS):
        lv = LEVELS[f]
        idx = np.minimum((U[:, j] * len(lv)).astype(int), len(lv) - 1)
        X[:, j] = lv[idx]
    return X


def predict(model, X, scaler=None):
    if scaler is not None:
        return model.predict(scaler.transform(X))
    return model.predict(X)


def sobol_run(model, n, k, mapper, seed, scaler=None, with_s2=False):
    """Return per-factor estimator ingredients fA, fB, fAB (n,k), fBA."""
    A_u, B_u = saltelli_matrices(n, k, seed)
    # swap columns in U-space so conditional mappers (which embed the
    # varying factors into a larger feature vector) stay correct
    A, B = mapper(A_u), mapper(B_u)
    fA = predict(model, A, scaler)
    fB = predict(model, B, scaler)
    fAB = np.empty((n, k))
    fBA = np.empty((n, k)) if with_s2 else None
    for i in range(k):
        ABi_u = A_u.copy()
        ABi_u[:, i] = B_u[:, i]
        fAB[:, i] = predict(model, mapper(ABi_u), scaler)
        if with_s2:
            BAi_u = B_u.copy()
            BAi_u[:, i] = A_u[:, i]
            fBA[:, i] = predict(model, mapper(BAi_u), scaler)
    return fA, fB, fAB, fBA


def indices_from(fA, fB, fAB, idx=None):
    """Saltelli-2010 S1 and Jansen ST from stored evaluations."""
    if idx is not None:
        fA, fB, fAB = fA[idx], fB[idx], fAB[idx]
    y = np.concatenate([fA, fB])
    V = y.var()
    f0 = y.mean()
    # centred Saltelli-2010 estimator: same expectation, lower variance
    S1 = ((fB - f0)[:, None] * (fAB - fA[:, None])).mean(axis=0) / V
    ST = 0.5 * ((fA[:, None] - fAB) ** 2).mean(axis=0) / V
    return S1, ST


def s2_closed(fA, fB, fAB, fBA):
    """Closed second-order indices S2_ij (Saltelli 2002)."""
    n, k = fAB.shape
    f0sq = fA.mean() * fB.mean()
    V = np.concatenate([fA, fB]).var()
    S1, _ = indices_from(fA, fB, fAB)
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            Vij = (fBA[:, i] * fAB[:, j]).mean() - f0sq
            out[(i, j)] = Vij / V - S1[i] - S1[j]
    return out


def bootstrap_ci(fA, fB, fAB, nboot=NBOOT):
    n = len(fA)
    S1b = np.empty((nboot, fAB.shape[1]))
    STb = np.empty((nboot, fAB.shape[1]))
    for b in range(nboot):
        idx = RNG.integers(0, n, n)
        S1b[b], STb[b] = indices_from(fA, fB, fAB, idx)
    return (np.percentile(S1b, [2.5, 97.5], axis=0),
            np.percentile(STb, [2.5, 97.5], axis=0))


# ----------------------------------------------------------------------
# exact 5-factor ANOVA on the factorial grid (per distribution)
# ----------------------------------------------------------------------
def exact_grid_anova(sub, target):
    """S1 and ST from the complete 5-factor factorial (discrete uniform)."""
    cols = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio',
            'logL', 'tdeg']
    y = sub[target]
    V = y.var(ddof=0)
    S1, ST = [], []
    for c in cols:
        m = sub.groupby(c)[target].mean()
        S1.append(m.var(ddof=0) / V)
        others = [o for o in cols if o != c]
        mo = sub.groupby(others)[target].mean()
        ST.append(1.0 - mo.var(ddof=0) / V)
    return np.array(S1), np.array(ST)


# ----------------------------------------------------------------------
# run everything
# ----------------------------------------------------------------------
rows_cont, rows_grid, rows_s2, rows_gp = [], [], [], []
store = {}
for dist in ('constant', 'exponential'):
    sub = df[df.length_distribution_type == dist]
    for t in TARGETS:
        m = bundle[dist]['hgb'][t]
        # continuous headline run (with S2 matrices)
        fA, fB, fAB, fBA = sobol_run(m, N_CONT, 5, map_continuous,
                                     seed=101, with_s2=True)
        S1, ST = indices_from(fA, fB, fAB)
        (S1lo, S1hi), (STlo, SThi) = bootstrap_ci(fA, fB, fAB)
        Vw = np.concatenate([fA, fB]).var()
        mu = np.concatenate([fA, fB]).mean()
        store[(dist, t)] = dict(S1=S1, ST=ST, V=Vw, mu=mu)
        for j, f in enumerate(FACTORS):
            rows_cont.append(dict(dist=dist, target=t, factor=f,
                                  S1=S1[j], S1_lo=S1lo[j], S1_hi=S1hi[j],
                                  ST=ST[j], ST_lo=STlo[j], ST_hi=SThi[j]))
        for (i, j), v in s2_closed(fA, fB, fAB, fBA).items():
            rows_s2.append(dict(dist=dist, target=t,
                                pair=f'{FACTORS[i]}x{FACTORS[j]}', S2=v))
        # grid-restricted validation vs exact ANOVA
        gA, gB, gAB, _ = sobol_run(m, N_GRID, 5, map_grid, seed=202)
        gS1, gST = indices_from(gA, gB, gAB)
        eS1, eST = exact_grid_anova(sub, t)
        for j, f in enumerate(FACTORS):
            rows_grid.append(dict(dist=dist, target=t, factor=f,
                                  S1_exact=eS1[j], S1_saltelli=gS1[j],
                                  ST_exact=eST[j], ST_saltelli=gST[j]))
        # GP cross-check (continuous)
        g = bundle[dist]['gp'][t]
        sc = bundle[dist]['scaler']
        pA, pB, pAB, _ = sobol_run(g, N_GP, 5, map_continuous,
                                   seed=101, scaler=sc)
        pS1, pST = indices_from(pA, pB, pAB)
        for j, f in enumerate(FACTORS):
            rows_gp.append(dict(dist=dist, target=t, factor=f,
                                S1_hgb=S1[j], S1_gp=pS1[j],
                                ST_hgb=ST[j], ST_gp=pST[j]))
    print(f'  {dist}: continuous + grid check + GP check done')

pd.DataFrame(rows_cont).to_csv(OUT / 'sobol_continuous.csv', index=False)
pd.DataFrame(rows_grid).to_csv(OUT / 'sobol_grid_check.csv', index=False)
pd.DataFrame(rows_s2).to_csv(OUT / 'sobol_s2.csv', index=False)
pd.DataFrame(rows_gp).to_csv(OUT / 'sobol_gp_check.csv', index=False)

# ----------------------------------------------------------------------
# 6-factor headline table: Dist as an equal-weight group factor
# ----------------------------------------------------------------------
rows6 = []
for t in TARGETS:
    c, e = store[('constant', t)], store[('exponential', t)]
    Vtot = 0.5 * c['V'] + 0.5 * e['V'] + (c['mu'] - e['mu']) ** 2 / 4
    Sdist = (c['mu'] - e['mu']) ** 2 / 4 / Vtot
    row = dict(target=t, S_Dist=Sdist)
    for j, f in enumerate(FACTORS):
        row[f'S1_{f}'] = (0.5 * c['V'] * c['S1'][j]
                          + 0.5 * e['V'] * e['S1'][j]) / Vtot
        row[f'ST_{f}'] = (0.5 * c['V'] * c['ST'][j]
                          + 0.5 * e['V'] * e['ST'][j]) / Vtot
    rows6.append(row)
t6 = pd.DataFrame(rows6)
t6.to_csv(OUT / 'sobol_table6.csv', index=False)

# ----------------------------------------------------------------------
# conditional (fixed-composition) indices at the anchors
# ----------------------------------------------------------------------
MFACT = ['alpha', 'logL', 'tdeg']
rows_cond = []
for vf, ef in ANCHORS:
    for t in TARGETS:
        s1_acc, st_acc, v_acc = [], [], []
        for dist in ('constant', 'exponential'):
            m = bundle[dist]['hgb'][t]

            def mapper(U, vf=vf, ef=ef):
                X = np.empty((len(U), 5))
                X[:, 0] = vf
                X[:, 1] = ef
                for jj, f in enumerate(MFACT):
                    lo, hi = BOUNDS[f]
                    X[:, 2 + jj] = lo + U[:, jj] * (hi - lo)
                return X

            fA, fB, fAB, _ = sobol_run(m, N_CONT, 3, mapper, seed=303)
            S1, ST = indices_from(fA, fB, fAB)
            s1_acc.append(S1)
            st_acc.append(ST)
            v_acc.append(np.concatenate([fA, fB]).var())
        w = np.array(v_acc) / np.sum(v_acc)
        S1m = w[0] * s1_acc[0] + w[1] * s1_acc[1]
        STm = w[0] * st_acc[0] + w[1] * st_acc[1]
        for jj, f in enumerate(MFACT):
            rows_cond.append(dict(Vf=vf, EfEm=ef, target=t, factor=f,
                                  S1=S1m[jj], ST=STm[jj]))
pd.DataFrame(rows_cond).to_csv(OUT / 'sobol_conditional.csv', index=False)

# ----------------------------------------------------------------------
# console summary
# ----------------------------------------------------------------------
print('\n=== 6-factor headline table (S1, %) ===')
show = t6.set_index('target')
for t in ['E1_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'v12', 'v13']:
    r = show.loc[t]
    print(f"{t:8s} Vf {100*r.S1_Vf:5.1f}({100*r.ST_Vf:5.1f})  "
          f"EfEm {100*r.S1_EfEm:5.1f}({100*r.ST_EfEm:5.1f})  "
          f"a {100*r.S1_alpha:4.1f}({100*r.ST_alpha:4.1f})  "
          f"L {100*r.S1_logL:4.1f}({100*r.ST_logL:4.1f})  "
          f"th {100*r.S1_tdeg:4.1f}({100*r.ST_tdeg:4.1f})  "
          f"Dist {100*r.S_Dist:4.1f}")

gc = pd.DataFrame(rows_grid)
d1 = (gc.S1_exact - gc.S1_saltelli).abs()
dt = (gc.ST_exact - gc.ST_saltelli).abs()
print(f'\n=== grid check: |exact - Saltelli| ===')
print(f'S1: median {d1.median():.4f}  max {d1.max():.4f}')
print(f'ST: median {dt.median():.4f}  max {dt.max():.4f}')

gp = pd.DataFrame(rows_gp)
dg = (gp.S1_hgb - gp.S1_gp).abs()
print(f'GP vs HGB (continuous S1): median {dg.median():.4f}  max {dg.max():.4f}')

cond = pd.DataFrame(rows_cond)
print('\n=== conditional S1 for E3_Em ===')
print(cond[cond.target == 'E3_Em'].round(3).to_string(index=False))

s2 = pd.DataFrame(rows_s2)
top = (s2.groupby('pair').S2.mean().sort_values(ascending=False).head(4))
print('\n=== largest mean closed S2 pairs ===')
print(top.round(3).to_string())
print('\ndone.')
