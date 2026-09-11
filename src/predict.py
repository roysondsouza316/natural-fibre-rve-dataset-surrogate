"""
Evaluate the trained surrogate for one parameter combination.

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

The models are the histogram gradient-boosting surrogates of the
manuscript, stored in out/surrogate_hgb.joblib (written by
01_train_surrogates.py).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'out' / 'surrogate_hgb.joblib'
RANGES = {'vf': (0.10, 0.30), 'efem': (4.0, 48.0), 'alpha': (1.0, 3.0),
          'lp': (50.0, 200.0)}
TARGETS = ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']


def theta_deg(kappa):
    """Mean deviation inclination of the von Mises-Fisher backbone."""
    if kappa == 0:
        return 0.0
    return float(np.degrees(np.arccos(1 / np.tanh(kappa) - 1 / kappa)))


def predict(vf, efem, alpha, lp, tdeg, dist, bundle=None):
    """Return a dict of the nine constants for one parameter set."""
    bundle = bundle or joblib.load(BUNDLE)
    models = bundle[dist]['hgb']
    x = np.array([[vf, efem, alpha, np.log10(lp), tdeg]])
    return {t: float(models[t].predict(x)[0]) for t in TARGETS}


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
