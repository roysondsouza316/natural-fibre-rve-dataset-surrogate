"""
Evaluate the Gaussian-process surrogate for one parameter combination.

    python predict.py --vf 0.20 --efem 20 --alpha 2 --lp 100 --kappa 10 --dist exponential

Inputs
  --vf     fibre volume fraction (0.10 to 0.30)
  --efem   fibre-to-matrix modulus ratio (4 to 48)
  --alpha  cross-section aspect ratio, width over thickness (1 to 3)
  --lp     projected fibre length in micrometres (50 to 200); for the
           exponential distribution this is the nominal mean
  --kappa  backbone waviness kappa_theta (0 = straight, 20, 10, 5), or
           give --tdeg directly (mean deviation inclination in degrees,
           0 to 36.9)
  --dist   length distribution, constant or exponential

Output: the nine homogenised constants normalised by the matrix modulus,
direction 3 being the fibre direction. The fibre thickness of the dataset
is 10 um; for other thicknesses scale --lp by the thickness ratio.

The GP models (Matern-5/2 kernel, log targets; see surrogate_gp.py) are
rebuilt on first use from out/dataset.csv with the fitted kernel
hyperparameters stored in out/gp_hyperparameters.json, without any
optimisation, and cached in out/surrogate_gp.joblib for later calls.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from surrogate_gp import gp_surrogate, theta_deg, FEATURES

HERE = Path(__file__).resolve().parent
DATA = HERE / 'out' / 'dataset.csv'
HYPER = HERE / 'out' / 'gp_hyperparameters.json'
CACHE = HERE / 'out' / 'surrogate_gp.joblib'
RANGES = {'vf': (0.10, 0.30), 'efem': (4.0, 48.0), 'alpha': (1.0, 3.0),
          'lp': (50.0, 200.0)}
TARGETS = ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']


def build_models():
    """Refit the GP pipelines with the stored kernel hyperparameters."""
    df = pd.read_csv(DATA)
    df['tdeg'] = df['theta'].map(lambda k: theta_deg(int(k)))
    df['logL'] = np.log10(df['length_value'])
    hyper = json.load(open(HYPER))
    models = {}
    for dist in ('constant', 'exponential'):
        sub = df[df.length_distribution_type == dist]
        X = sub[FEATURES].values
        models[dist] = {}
        for t in TARGETS:
            pipe = gp_surrogate(restarts=0)
            gpr = pipe[-1].regressor
            gpr.kernel = gpr.kernel.clone_with_theta(
                np.array(hyper[dist][t]['theta']))
            gpr.optimizer = None
            models[dist][t] = pipe.fit(X, sub[t].values)
    joblib.dump(models, CACHE, compress=3)
    return models


def load_models():
    if CACHE.exists():
        return joblib.load(CACHE)
    print('building the GP models from the dataset (first use only) ...',
          flush=True)
    return build_models()


def predict(vf, efem, alpha, lp, tdeg, dist, models=None):
    """Return a dict of the nine constants for one parameter set."""
    models = models or load_models()
    x = np.array([[vf, efem, alpha, np.log10(lp), tdeg]])
    return {t: float(models[dist][t].predict(x)[0]) for t in TARGETS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--vf', type=float, required=True)
    ap.add_argument('--efem', type=float, required=True)
    ap.add_argument('--alpha', type=float, default=2.0)
    ap.add_argument('--lp', type=float, default=100.0)
    ap.add_argument('--kappa', type=float, default=10.0)
    ap.add_argument('--tdeg', type=float, default=None)
    ap.add_argument('--dist', choices=['constant', 'exponential'],
                    default='exponential')
    a = ap.parse_args()

    for name, (lo, hi) in RANGES.items():
        v = getattr(a, name)
        if not lo <= v <= hi:
            print(f'warning: {name} = {v} is outside the design window '
                  f'[{lo}, {hi}]; the surrogate extrapolates')
    tdeg = a.tdeg if a.tdeg is not None else theta_deg(a.kappa)
    if not 0.0 <= tdeg <= theta_deg(5) + 1e-9:
        print(f'warning: tdeg = {tdeg:.1f} is outside the design window '
              f'[0, {theta_deg(5):.1f}]; the surrogate extrapolates')

    out = predict(a.vf, a.efem, a.alpha, a.lp, tdeg, a.dist)
    print(f'inputs: vf={a.vf}, Ef/Em={a.efem}, alpha={a.alpha}, '
          f'Lp={a.lp} um, tdeg={tdeg:.1f} deg, {a.dist} distribution')
    for t, v in out.items():
        print(f'  {t:7s} {v:8.4f}')


if __name__ == '__main__':
    main()
