"""
Coefficients of the polynomial comparison surrogate
---------------------------------------------------
Fits the polynomial of surrogate_poly.py (total degree 4, determined terms,
113 monomials) to each of the nine constants of each length distribution
and writes the coefficients so that the constants can be evaluated without
the code:

    y = sum_k c_k * x1^p1k * x2^p2k * x3^p3k * x4^p4k * x5^p5k

with the scaled inputs x_i = 2 (u_i - u_min) / (u_max - u_min) - 1 of
u = (v_f, Ef/Em, alpha, log10 L_p, tdeg); u_min and u_max are the first two
rows of each file.

Reads : out/dataset.csv
Writes: out/sensitivity/poly_coefficients_constant.csv,
        out/sensitivity/poly_coefficients_exponential.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

from surrogate_gp import theta_deg
from surrogate_poly import poly_surrogate

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'sensitivity'
FEAT = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio', 'logL', 'tdeg']
NAMES = ['vf', 'EfEm', 'alpha', 'log10Lp', 'tdeg']
TARGETS = ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em',
           'v12', 'v13', 'v23']

DATA = HERE / 'out' / 'dataset_clean.csv' if (HERE / 'out' / 'dataset_clean.csv').exists() else HERE / 'out' / 'dataset.csv'
df = pd.read_csv(DATA)
df['tdeg'] = df['theta'].map(lambda k: theta_deg(int(k)))
df['logL'] = np.log10(df['length_value'])

for dist in ('constant', 'exponential'):
    sub = df[df.length_distribution_type == dist]
    X = sub[FEAT].values
    rows = None
    models = {}
    for t in TARGETS:
        m = models[t] = poly_surrogate().fit(X, sub[t].values)
        scaler, terms, lin = m.named_steps.values()
        powers = terms.poly_.powers_[terms.keep_]
        if rows is None:
            rows = pd.DataFrame(powers, columns=[f'p_{n}' for n in NAMES])
            rows.insert(0, 'term', ['*'.join(f'{n}^{p}' for n, p in zip(NAMES, pw) if p) or '1' for pw in powers])
        coef = lin.coef_.copy()
        coef[0] += lin.intercept_          # the constant term carries the intercept
        rows[t] = coef
    head = pd.DataFrame({'term': ['u_min', 'u_max'], **{f'p_{n}': [lo, hi] for n, lo, hi in zip(NAMES, scaler.data_min_, scaler.data_max_)},
                         **{t: [np.nan, np.nan] for t in TARGETS}})
    out = pd.concat([head, rows], ignore_index=True)
    path = OUT / f'poly_coefficients_{dist}.csv'
    out.to_csv(path, index=False, float_format='%.10g')
    # check: evaluate the table by hand against the pipeline for the first design point
    u = X[0]
    x = 2 * (u - scaler.data_min_) / (scaler.data_max_ - scaler.data_min_) - 1
    y_table = sum(c * np.prod(x ** pw) for c, pw in zip(rows['E3_Em'], powers))
    print(f'{dist}: {len(rows)} terms written to {path.name}; check E3 at first design point: table {y_table:.6f}, pipeline {models["E3_Em"].predict(X[:1])[0]:.6f}')
