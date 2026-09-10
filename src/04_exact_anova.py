"""
Exact factorial ANOVA for the supplementary material
----------------------------------------------------
Ordinary-least-squares ANOVA (Type-II sums of squares) on the full
factorial, with all six main effects and all two-way interactions; the
residual (three-way and higher interactions plus RVE realisation
noise) serves as the error term, the standard practice for a
single-replicate factorial. Reports df, sum-of-squares share, F and p
per factor for each independent elastic constant.

Output: out/sensitivity/classical_anova.csv (+ console table)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'sensitivity' / 'classical_anova.csv'

df = pd.read_csv(HERE / 'out' / 'dataset.csv')
df = df.rename(columns={'fiber_fraction': 'Vf', 'Ef_Em': 'EfEm',
                        'width_to_thickness_ratio': 'alpha',
                        'length_value': 'L', 'theta': 'kt',
                        'length_distribution_type': 'dist'})
TARGETS = ['E1_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'v12', 'v13']
FACT = ['Vf', 'EfEm', 'alpha', 'L', 'kt', 'dist']

main = ' + '.join(f'C({f})' for f in FACT)
pairs = ' + '.join(f'C({a}):C({b})'
                   for i, a in enumerate(FACT) for b in FACT[i + 1:])
rows = []
for t in TARGETS:
    model = smf.ols(f'{t} ~ {main} + {pairs}', data=df).fit()
    an = sm.stats.anova_lm(model, typ=2)
    ss_tot = an['sum_sq'].sum()
    err_df = int(an.loc['Residual', 'df'])
    for f in FACT:
        r = an.loc[f'C({f})']
        rows.append(dict(target=t, factor=f, df=int(r['df']),
                         SS_pct=100 * r['sum_sq'] / ss_tot,
                         F=r['F'], p=r['PR(>F)'], err_df=err_df))
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

print(f'error df = {res.err_df.iloc[0]}')
for t in TARGETS:
    sub = res[res.target == t]
    line = f'{t:8s}'
    for f in FACT:
        r = sub[sub.factor == f].iloc[0]
        star = ('***' if r.p < 1e-3 else '**' if r.p < 1e-2
                else '*' if r.p < 0.05 else 'ns')
        line += f'  {f}: F={r.F:9.1f} {star}'
    print(line)
print('\nnon-significant (p >= 0.05) entries:')
ns = res[res.p >= 0.05]
print(ns[['target', 'factor', 'F', 'p']].round(3).to_string(index=False)
      if len(ns) else '  none')
print('\nwrote', OUT)
