"""Cross-validation benchmark: Gaussian process vs polynomial surrogate.

Both models are scored under the same 10-fold cross-validation (shuffled,
seed 42), per length-distribution type and per elastic constant, with the
five inputs of the surrogate of record (v_f, Ef/Em, alpha, log10 L_p,
mean inclination). Model definitions: surrogate_gp.py, surrogate_poly.py.

Writes out/benchmark/model_benchmark.csv and model_benchmark_summary.csv.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from surrogate_gp import gp_surrogate, theta_deg
from surrogate_poly import poly_surrogate

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'out'
BENCHDIR = OUTDIR / 'benchmark'
BENCHDIR.mkdir(parents=True, exist_ok=True)
DATA = OUTDIR / 'dataset_clean.csv' if (OUTDIR / 'dataset_clean.csv').exists() else OUTDIR / 'dataset.csv'

FEATURES = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio', 'logL', 'tdeg']
TARGETS = ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']
CV = KFold(n_splits=10, shuffle=True, random_state=42)
MODELS = {
    'Gaussian process': lambda: gp_surrogate(restarts=0),
    'Polynomial (deg 4)': poly_surrogate,
}


def main():
    df = pd.read_csv(DATA)
    df['tdeg'] = df['theta'].map(lambda k: theta_deg(int(k)))
    df['logL'] = np.log10(df['length_value'])
    rows = []
    for dist in ['constant', 'exponential']:
        sub = df[df['length_distribution_type'] == dist].reset_index(drop=True)
        X = sub[FEATURES].values
        print(f"\n=== Distribution = {dist} ({len(sub)} rows) ===")
        print(f"{'model':<22}{'mean R2':>10}{'min R2':>10}{'MAPE %':>10}{'t [s]':>9}")
        for label, factory in MODELS.items():
            t0 = time.time()
            r2s, mapes = [], []
            for t in TARGETS:
                cv = cross_validate(factory(), X, sub[t].values, cv=CV,
                                    scoring=('r2', 'neg_mean_absolute_percentage_error'))
                r2 = cv['test_r2'].mean()
                mape = -cv['test_neg_mean_absolute_percentage_error'].mean() * 100.0
                r2s.append(r2)
                mapes.append(mape)
                rows.append(dict(distribution=dist, model=label, target=t,
                                 cv_r2=r2, cv_mape=mape))
            print(f"{label:<22}{np.mean(r2s):>10.4f}{np.min(r2s):>10.4f}"
                  f"{np.mean(mapes):>10.2f}{time.time() - t0:>9.1f}", flush=True)

    long = pd.DataFrame(rows)
    long.to_csv(BENCHDIR / 'model_benchmark.csv', index=False)
    summ = (long.groupby(['distribution', 'model'])
            .agg(mean_r2=('cv_r2', 'mean'), min_r2=('cv_r2', 'min'),
                 mean_mape=('cv_mape', 'mean')).reset_index())
    pooled = (long.groupby('model')
              .agg(mean_r2=('cv_r2', 'mean'), min_r2=('cv_r2', 'min'),
                   mean_mape=('cv_mape', 'mean')).reset_index())
    pooled.insert(0, 'distribution', 'pooled')
    summary = pd.concat([summ, pooled], ignore_index=True)
    summary.to_csv(BENCHDIR / 'model_benchmark_summary.csv', index=False)
    print(f"\nWrote {BENCHDIR / 'model_benchmark.csv'} and model_benchmark_summary.csv")
    print("\n=== Pooled (mean over 9 outputs, both distributions) ===")
    print(pooled[['model', 'mean_r2', 'min_r2', 'mean_mape']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
