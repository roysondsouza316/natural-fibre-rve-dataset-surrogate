"""Final Sobol computation reported in the manuscript.

Centred first-order estimator, N = 2^14 base samples. Writes the
continuous-range indices, the second-order pairs, the 6-factor headline
table and the conditional indices at fixed compositions to
out/sensitivity/. Requires the surrogate bundle saved by
01_train_surrogates.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import qmc

RNG = np.random.default_rng(11)
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'sensitivity'
bundle = joblib.load(OUT / 'surrogates_tdeg.joblib')

TARGETS = ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']
FACTORS = ['Vf', 'EfEm', 'alpha', 'logL', 'tdeg']
TD5 = 36.863
BOUNDS = {'Vf': (0.10, 0.30), 'EfEm': (4.0, 48.0), 'alpha': (1.0, 3.0),
          'logL': (np.log10(50.0), np.log10(200.0)), 'tdeg': (0.0, TD5)}
N = 2 ** 14
NBOOT = 300
ANCHORS = [(0.10, 12.0), (0.20, 28.0), (0.30, 48.0)]
MFACT = ['alpha', 'logL', 'tdeg']


def saltelli_matrices(n, k, seed):
    s = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    U = s.random(n)
    return U[:, :k], U[:, k:]


def indices_from(fA, fB, fAB, idx=None):
    if idx is not None:
        fA, fB, fAB = fA[idx], fB[idx], fAB[idx]
    y = np.concatenate([fA, fB])
    V = y.var()
    f0 = y.mean()
    S1 = ((fB - f0)[:, None] * (fAB - fA[:, None])).mean(axis=0) / V
    ST = 0.5 * ((fA[:, None] - fAB) ** 2).mean(axis=0) / V
    return S1, ST


def bootstrap_ci(fA, fB, fAB, nboot=NBOOT):
    n = len(fA)
    S1b = np.empty((nboot, fAB.shape[1]))
    STb = np.empty((nboot, fAB.shape[1]))
    for b in range(nboot):
        idx = RNG.integers(0, n, n)
        S1b[b], STb[b] = indices_from(fA, fB, fAB, idx)
    return (np.percentile(S1b, [2.5, 97.5], axis=0),
            np.percentile(STb, [2.5, 97.5], axis=0))


def map_continuous(U):
    X = np.empty_like(U)
    for j, f in enumerate(FACTORS):
        lo, hi = BOUNDS[f]
        X[:, j] = lo + U[:, j] * (hi - lo)
    return X


rows_cont, rows_s2 = [], []
store = {}
for dist in ('constant', 'exponential'):
    for t in TARGETS:
        m = bundle[dist]['gp'][t]
        A_u, B_u = saltelli_matrices(N, 5, seed=101)
        A, B = map_continuous(A_u), map_continuous(B_u)
        fA, fB = m.predict(A), m.predict(B)
        fAB = np.empty((N, 5))
        fBA = np.empty((N, 5))
        for i in range(5):
            ABi_u = A_u.copy()
            ABi_u[:, i] = B_u[:, i]
            fAB[:, i] = m.predict(map_continuous(ABi_u))
            BAi_u = B_u.copy()
            BAi_u[:, i] = A_u[:, i]
            fBA[:, i] = m.predict(map_continuous(BAi_u))
        S1, ST = indices_from(fA, fB, fAB)
        (S1lo, S1hi), (STlo, SThi) = bootstrap_ci(fA, fB, fAB)
        y = np.concatenate([fA, fB])
        store[(dist, t)] = dict(S1=S1, ST=ST, V=y.var(), mu=y.mean())
        for j, f in enumerate(FACTORS):
            rows_cont.append(dict(dist=dist, target=t, factor=f,
                                  S1=S1[j], S1_lo=S1lo[j], S1_hi=S1hi[j],
                                  ST=ST[j], ST_lo=STlo[j], ST_hi=SThi[j]))
        f0sq = fA.mean() * fB.mean()
        V = y.var()
        for i in range(5):
            for j in range(i + 1, 5):
                Vij = (fBA[:, i] * fAB[:, j]).mean() - f0sq
                rows_s2.append(dict(
                    dist=dist, target=t,
                    pair=f'{FACTORS[i]}x{FACTORS[j]}',
                    S2=Vij / V - S1[i] - S1[j]))
    print(dist, 'done')

pd.DataFrame(rows_cont).to_csv(OUT / 'sobol_continuous.csv', index=False)
pd.DataFrame(rows_s2).to_csv(OUT / 'sobol_s2.csv', index=False)

# 6-factor headline table
rows6 = []
for t in TARGETS:
    c, e = store[('constant', t)], store[('exponential', t)]
    Vtot = 0.5 * c['V'] + 0.5 * e['V'] + (c['mu'] - e['mu']) ** 2 / 4
    row = dict(target=t, S_Dist=(c['mu'] - e['mu']) ** 2 / 4 / Vtot)
    for j, f in enumerate(FACTORS):
        row[f'S1_{f}'] = (0.5 * c['V'] * c['S1'][j]
                          + 0.5 * e['V'] * e['S1'][j]) / Vtot
        row[f'ST_{f}'] = (0.5 * c['V'] * c['ST'][j]
                          + 0.5 * e['V'] * e['ST'][j]) / Vtot
    rows6.append(row)
t6 = pd.DataFrame(rows6)
t6.to_csv(OUT / 'sobol_table6.csv', index=False)

# conditional (centred)
rows_cond = []
for vf, ef in ANCHORS:
    for t in TARGETS:
        s1_acc, st_acc, v_acc = [], [], []
        for dist in ('constant', 'exponential'):
            m = bundle[dist]['gp'][t]

            def mapper(U):
                X = np.empty((len(U), 5))
                X[:, 0] = vf
                X[:, 1] = ef
                for jj, f in enumerate(MFACT):
                    lo, hi = BOUNDS[f]
                    X[:, 2 + jj] = lo + U[:, jj] * (hi - lo)
                return X

            A_u, B_u = saltelli_matrices(N, 3, seed=303)
            fA, fB = m.predict(mapper(A_u)), m.predict(mapper(B_u))
            fAB = np.empty((N, 3))
            for i in range(3):
                ABi_u = A_u.copy()
                ABi_u[:, i] = B_u[:, i]
                fAB[:, i] = m.predict(mapper(ABi_u))
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

# ---- summaries -------------------------------------------------------
print('\n=== 6-factor headline table S1(ST) % ===')
show = t6.set_index('target')
for t in ['E1_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'v12', 'v13']:
    r = show.loc[t]
    print(f"{t:8s} Vf {100*r.S1_Vf:5.1f}({100*r.ST_Vf:5.1f})  "
          f"EfEm {100*r.S1_EfEm:5.1f}({100*r.ST_EfEm:5.1f})  "
          f"a {100*r.S1_alpha:4.1f}({100*r.ST_alpha:4.1f})  "
          f"L {100*r.S1_logL:4.1f}({100*r.ST_logL:4.1f})  "
          f"th {100*r.S1_tdeg:4.1f}({100*r.ST_tdeg:4.1f})  "
          f"Dist {100*r.S_Dist:4.1f}")

cc = pd.DataFrame(rows_cont)
print(f"\nS1 95% CI width: median {(cc.S1_hi-cc.S1_lo).median():.4f}  "
      f"max {(cc.S1_hi-cc.S1_lo).max():.4f}")
print(f"ST 95% CI width: median {(cc.ST_hi-cc.ST_lo).median():.4f}  "
      f"max {(cc.ST_hi-cc.ST_lo).max():.4f}")

cond = pd.DataFrame(rows_cond)
print('\n=== conditional S1(ST) % ===')
for vf, ef in ANCHORS:
    print(f'--- Vf={vf:.2f}, Ef/Em={ef:.0f} ---')
    for t in ['E1_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'v12', 'v13']:
        c = cond[(cond.Vf == vf) & (cond.EfEm == ef) & (cond.target == t)]
        v = {r.factor: (100 * r.S1, 100 * r.ST) for r in c.itertuples()}
        print(f"  {t:8s} a {v['alpha'][0]:5.1f}({v['alpha'][1]:5.1f})  "
              f"L {v['logL'][0]:5.1f}({v['logL'][1]:5.1f})  "
              f"th {v['tdeg'][0]:5.1f}({v['tdeg'][1]:5.1f})")

s2 = pd.DataFrame(rows_s2)
e3 = s2[s2.target == 'E3_Em'].pivot_table(index='pair', columns='dist',
                                          values='S2')
print('\n=== closed S2 for E3_Em (%) ===')
print((100 * e3).round(1).sort_values('constant', ascending=False)
      .head(4).to_string())
