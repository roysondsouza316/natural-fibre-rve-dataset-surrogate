"""
Surrogate model benchmark
-------------------------
Why this script exists
======================
The variance-based sensitivity analysis uses a fast regression
surrogate as the inner loop of the Saltelli estimator.  The manuscript uses a
histogram gradient-boosting regressor (HGB).  A reasonable question from a
reviewer (and a colleague) is: *why HGB and not one of the surrogates that are
standard in the global-sensitivity / micromechanics literature*, such as
Gaussian process / kriging, polynomial chaos expansion (PCE), random forest,
kernel methods, or a small neural network?

This script answers that question with evidence.  It benchmarks a panel of
candidate surrogates under the *same* 10-fold cross-validation protocol,
separately for each length-distribution type, and reports the mean
coefficient of determination (R^2) and the mean absolute percentage error
(MAPE) averaged over the nine elastic constants.

Models compared
===============
  * Polynomial (deg 2) ridge   - cheap analytic baseline
  * Polynomial chaos (deg 4)   - Legendre PCE, the GSA-community standard
                                 (implemented here in pure numpy so the run
                                 has no fragile binary dependency)
  * Kernel ridge (RBF)
  * Support-vector regression (RBF)
  * Random forest
  * Extra trees
  * Hist. gradient boosting    - the model used in the paper
  * Gaussian process (kriging) - the other GSA-community standard
  * Multilayer perceptron      - a small neural network

Outputs (written to out/benchmark/)
  * model_benchmark.csv          long format: model, distribution, target,
                                  cv_r2, cv_mape
  * model_benchmark_summary.csv  model x (mean/min R^2, mean MAPE) across the
                                  nine outputs, per distribution and pooled
  * model_benchmark.png          grouped bar chart of mean CV R^2 and MAPE
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor, ExtraTreesRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'out'
BENCHDIR = OUTDIR / 'benchmark'
BENCHDIR.mkdir(parents=True, exist_ok=True)
DATA = OUTDIR / 'dataset.csv'

FEATURES = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio',
            'length_value', 'theta']
TARGETS = ['E1_Em', 'E2_Em', 'E3_Em',
           'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']

CV = KFold(n_splits=10, shuffle=True, random_state=42)


# ----------------------------------------------------------------------
# Pure-numpy Legendre polynomial-chaos surrogate (sklearn-compatible)
# ----------------------------------------------------------------------
class LegendrePCE(BaseEstimator, RegressorMixin):
    """Total-degree Legendre polynomial chaos expansion fitted by ordinary
    least squares.  Inputs are linearly mapped to [-1, 1] using the training
    range, which makes the Legendre basis orthogonal over the design box.
    """

    def __init__(self, degree=4):
        self.degree = degree

    def _multi_indices(self, n_dim):
        from itertools import product
        idx = [c for c in product(range(self.degree + 1), repeat=n_dim)
               if sum(c) <= self.degree]
        return np.array(idx, dtype=int)

    def _design(self, Xs):
        # Xs already scaled to [-1, 1]; build per-dimension Legendre-Vandermonde
        n, d = Xs.shape
        vander = [np.polynomial.legendre.legvander(Xs[:, j], self.degree)
                  for j in range(d)]               # each (n, degree+1)
        phi = np.ones((n, len(self.midx_)))
        for k, alpha in enumerate(self.midx_):
            col = np.ones(n)
            for j in range(d):
                col = col * vander[j][:, alpha[j]]
            phi[:, k] = col
        return phi

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.xmin_ = X.min(axis=0)
        self.xmax_ = X.max(axis=0)
        self.midx_ = self._multi_indices(X.shape[1])
        Xs = self._scale(X)
        phi = self._design(Xs)
        self.coef_, *_ = np.linalg.lstsq(phi, np.asarray(y, dtype=float),
                                         rcond=None)
        return self

    def _scale(self, X):
        rng = np.where(self.xmax_ - self.xmin_ == 0, 1.0,
                       self.xmax_ - self.xmin_)
        Xs = 2.0 * (X - self.xmin_) / rng - 1.0
        return np.clip(Xs, -1.0, 1.0)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        phi = self._design(self._scale(X))
        return phi @ self.coef_


def make_models():
    """Return an ordered dict of (label -> fresh estimator factory)."""
    gp_kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                 * RBF(length_scale=np.ones(len(FEATURES)),
                       length_scale_bounds=(1e-2, 1e2))
                 + WhiteKernel(noise_level=1e-3,
                               noise_level_bounds=(1e-8, 1e1)))
    return {
        'Polynomial (deg 2)': lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-3)),
        'Poly. chaos (deg 4)': lambda: LegendrePCE(degree=4),
        'Kernel ridge (RBF)': lambda: make_pipeline(
            StandardScaler(), KernelRidge(kernel='rbf', alpha=1e-2, gamma=0.1)),
        'SVR (RBF)': lambda: make_pipeline(
            StandardScaler(), SVR(kernel='rbf', C=10.0, epsilon=1e-3)),
        'Random forest': lambda: RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'Extra trees': lambda: ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'Hist. grad. boosting': lambda: HistGradientBoostingRegressor(
            max_iter=500, max_depth=None, min_samples_leaf=5,
            learning_rate=0.05, random_state=42),
        'Gaussian process': lambda: make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True,
                                     n_restarts_optimizer=0, random_state=42)),
        'MLP (neural net)': lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                         alpha=1e-3, max_iter=800, early_stopping=True,
                         n_iter_no_change=15, random_state=42)),
    }


def main():
    df = pd.read_csv(DATA)
    models = make_models()
    rows = []

    for dist in ['constant', 'exponential']:
        sub = df[df['length_distribution_type'] == dist].reset_index(drop=True)
        X = sub[FEATURES].values
        print(f"\n=== Distribution = {dist} ({len(sub)} rows) ===")
        print(f"{'model':<22}{'mean R2':>10}{'min R2':>10}{'MAPE %':>10}{'t [s]':>9}")
        for label, factory in models.items():
            t0 = time.time()
            r2s, mapes = [], []
            for t in TARGETS:
                y = sub[t].values
                r2 = cross_val_score(factory(), X, y, cv=CV, scoring='r2')
                mape = cross_val_score(
                    factory(), X, y, cv=CV,
                    scoring='neg_mean_absolute_percentage_error')
                r2s.append(r2.mean())
                mapes.append(-mape.mean() * 100.0)
                rows.append(dict(distribution=dist, model=label, target=t,
                                 cv_r2=r2.mean(), cv_mape=-mape.mean() * 100.0))
            dt = time.time() - t0
            print(f"{label:<22}{np.mean(r2s):>10.4f}{np.min(r2s):>10.4f}"
                  f"{np.mean(mapes):>10.2f}{dt:>9.1f}")

    long = pd.DataFrame(rows)
    long.to_csv(BENCHDIR / 'model_benchmark.csv', index=False)
    print(f"\nWrote {BENCHDIR / 'model_benchmark.csv'}")

    # ---- summary: per model, per distribution + pooled ----
    summ = (long.groupby(['distribution', 'model'])
            .agg(mean_r2=('cv_r2', 'mean'),
                 min_r2=('cv_r2', 'min'),
                 mean_mape=('cv_mape', 'mean'))
            .reset_index())
    pooled = (long.groupby('model')
              .agg(mean_r2=('cv_r2', 'mean'),
                   min_r2=('cv_r2', 'min'),
                   mean_mape=('cv_mape', 'mean'))
              .reset_index())
    pooled.insert(0, 'distribution', 'pooled')
    summary = pd.concat([summ, pooled], ignore_index=True)
    summary.to_csv(BENCHDIR / 'model_benchmark_summary.csv', index=False)
    print(f"Wrote {BENCHDIR / 'model_benchmark_summary.csv'}")
    print("\n=== Pooled summary (mean across 9 outputs, both distributions) ===")
    print(pooled[['model', 'mean_r2', 'min_r2', 'mean_mape']]
          .sort_values('mean_r2', ascending=False).to_string(index=False))

    plot_benchmark(pooled)


def plot_benchmark(pooled):
    order = pooled.sort_values('mean_r2', ascending=True)
    labels = order['model'].tolist()
    r2 = order['mean_r2'].values
    mape = order['mean_mape'].values
    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    hgb_mask = np.array(['gradient boosting' in m.lower() for m in labels])
    colors = np.where(hgb_mask, '#d62728', '#1f77b4')

    ax1.barh(y, r2, color=colors, edgecolor='white')
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel(r'Mean 10-fold CV $R^2$ (9 outputs)')
    ax1.set_xlim(min(0.9, r2.min() - 0.02), 1.0)
    ax1.grid(axis='x', alpha=0.3)
    for yi, v in zip(y, r2):
        ax1.text(v - 0.002, yi, f'{v:.3f}', va='center', ha='right',
                 color='white', fontsize=9)

    ax2.barh(y, mape, color=colors, edgecolor='white')
    ax2.set_xlabel('Mean CV MAPE (%)')
    ax2.grid(axis='x', alpha=0.3)
    for yi, v in zip(y, mape):
        ax2.text(v + max(mape) * 0.01, yi, f'{v:.2f}', va='center', ha='left',
                 fontsize=9)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#d62728', label='Model used in paper'),
                        Patch(color='#1f77b4', label='Alternatives')],
               loc='lower center', ncol=2, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Surrogate model comparison (higher $R^2$, lower MAPE better)')
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = BENCHDIR / 'model_benchmark.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
