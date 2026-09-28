"""
Surrogate comparison: Gaussian process against polynomial regressions
--------------------------------------------------------------------
Every model is scored per length-distribution type and per elastic constant
with the five inputs of the surrogate of record (v_f, Ef/Em, alpha,
log10 L_p, mean inclination tdeg):
  * fit on the design grid (R^2, MAPE)
  * 10-fold cross-validation on the grid (KFold, shuffled, seed 42)
  * the 60 independent FE test points between the design levels
    (30 RVEs by Latin hypercube sampling inside the design window, two
    modulus ratios each)

Models: the GP of the manuscript (surrogate_gp.py, fitted here without
optimiser restarts) and polynomial regressions of total degree 2 to 5,
each with all monomials and with only the monomials the design determines
(surrogate_poly.py).

Outputs (out/benchmark/)
  model_benchmark.csv          one row per model, variant, distribution
                               and constant
  model_benchmark_summary.csv  pooled over the constants and both
                               distributions
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from surrogate_gp import gp_surrogate, theta_deg
from surrogate_poly import poly_surrogate

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'out'
BENCHDIR = OUTDIR / 'benchmark'
BENCHDIR.mkdir(parents=True, exist_ok=True)
DATA = OUTDIR / 'dataset_clean.csv' if (OUTDIR / 'dataset_clean.csv').exists() else OUTDIR / 'dataset.csv'
TEST_DIR = OUTDIR / 'test_set' if (OUTDIR / 'test_set').exists() else HERE.parent / 'test_data'

FEATURES = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio', 'logL', 'tdeg']
TARGETS = ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']
REPORTED = ['E1_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'v12', 'v13']
CV = KFold(n_splits=10, shuffle=True, random_state=42)

MODELS = [('Gaussian process', '', lambda: gp_surrogate(restarts=0))]
for deg in (2, 3, 4, 5):
    MODELS.append((f'Polynomial deg {deg}', 'all terms', lambda d=deg: poly_surrogate(d, drop_undetermined=False)))
    MODELS.append((f'Polynomial deg {deg}', 'determined terms', lambda d=deg: poly_surrogate(d, drop_undetermined=True)))


def mape(y, p):
    return 100.0 * np.mean(np.abs(p - y) / np.abs(y))


def load_test_set():
    """FE results of the 60 test points with their surrogate inputs."""
    design = pd.read_csv(TEST_DIR / 'test_set_design.csv')
    fe = pd.read_csv(TEST_DIR / 'test_set_fe_results.csv')
    fe = fe.rename(columns={'E1': 'E1_Em', 'E2': 'E2_Em', 'E3': 'E3_Em', 'G12': 'G12_Em',
                            'G13': 'G13_Em', 'G23': 'G23_Em'})
    if 'matrix_modulus' in fe.columns:
        for t in TARGETS[:6]:
            fe[t] = fe[t] / fe['matrix_modulus']
    fe['Ef_Em'] = fe['fiber_modulus']
    fe = fe.drop(columns=[c for c in ('dist', 'vf', 'alpha', 'Lp', 'tdeg') if c in fe.columns])
    fe = fe.merge(design[['sim_name', 'dist', 'vf', 'alpha', 'Lp', 'tdeg']],
                  left_on='sim', right_on='sim_name')
    fe['fiber_fraction'] = fe['vf']
    fe['width_to_thickness_ratio'] = fe['alpha']
    fe['logL'] = np.log10(fe['Lp'])
    return fe


def n_terms(model):
    step = getattr(model, 'named_steps', {}).get('polynomialterms')
    return step.n_terms_ if step is not None else np.nan


def main():
    df = pd.read_csv(DATA)
    df['tdeg'] = df['theta'].map(lambda k: theta_deg(int(k)))
    df['logL'] = np.log10(df['length_value'])
    test = load_test_set()
    rows = []
    for label, variant, factory in MODELS:
        t0 = time.time()
        for dist in ('constant', 'exponential'):
            sub = df[df['length_distribution_type'] == dist].reset_index(drop=True)
            X = sub[FEATURES].values
            te = test[test['dist'] == dist]
            Xt = te[FEATURES].values
            for t in TARGETS:
                y, yt = sub[t].values, te[t].values
                m = factory().fit(X, y)
                p = m.predict(X)
                oof = np.empty(len(y))
                for a, b in CV.split(X):
                    oof[b] = factory().fit(X[a], y[a]).predict(X[b])
                pt = m.predict(Xt)
                ape_t = 100.0 * np.abs(pt - yt) / np.abs(yt)
                rows.append(dict(model=label, variant=variant, distribution=dist, target=t,
                                 n_terms=n_terms(m),
                                 train_r2=r2_score(y, p), train_mape=mape(y, p),
                                 cv_r2=r2_score(y, oof), cv_mape=mape(y, oof),
                                 test_mape=ape_t.mean(), test_max=ape_t.max()))
        r = pd.DataFrame([x for x in rows if x['model'] == label and x['variant'] == variant])
        print(f"{label:18s} {variant:17s} terms {r.n_terms.iloc[0]!s:>5s} | train R2 {r.train_r2.mean():.4f} "
              f"MAPE {r.train_mape.mean():6.2f} | CV R2 {r.cv_r2.mean():.4f} MAPE {r.cv_mape.mean():6.2f} | "
              f"test MAPE {r[r.target.isin(REPORTED)].test_mape.mean():10.3g} max {r[r.target.isin(REPORTED)].test_max.max():10.3g} "
              f"[{time.time() - t0:.0f} s]", flush=True)

    long = pd.DataFrame(rows)
    long.to_csv(BENCHDIR / 'model_benchmark.csv', index=False)
    rep = long[long.target.isin(REPORTED)]
    summary = (long.groupby(['model', 'variant'], sort=False)
               .agg(n_terms=('n_terms', 'first'),
                    train_r2=('train_r2', 'mean'), train_mape=('train_mape', 'mean'),
                    cv_r2=('cv_r2', 'mean'), cv_min_r2=('cv_r2', 'min'), cv_mape=('cv_mape', 'mean'))
               .reset_index())
    summary['test_mape'] = rep.groupby(['model', 'variant'], sort=False).test_mape.mean().values
    summary['test_max'] = rep.groupby(['model', 'variant'], sort=False).test_max.max().values
    summary.to_csv(BENCHDIR / 'model_benchmark_summary.csv', index=False)
    print(f"\nWrote {BENCHDIR / 'model_benchmark.csv'} and model_benchmark_summary.csv")
    print("train and CV pooled over the nine constants; test set over the six reported constants")
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
