"""
Literature plug-in validation of the surrogate
----------------------------------------------
Reads hand-curated experimental points (out/sensitivity/
literature_points.csv), maps each to the surrogate inputs and compares
the predicted longitudinal modulus with the measured one.

Mapping rules (stated in the manuscript):
  Vf        taken as reported, or converted from wt% with the phase
            densities via  Vf = (w/rho_f) / (w/rho_f + (1-w)/rho_m)
  Ef/Em     from the reported constituent moduli, clipped to [4, 48]
  L feature the surrogate was trained at D = 10 um, so the physically
            comparable length is matched through the aspect ratio:
            L_eq = 10 um x (L/D)_paper, clipped to [50, 200] um
  waviness  as-processed (tdeg = 25.8 deg); alpha = 2 (mid level);
            exponential length distribution
  E3        prediction = (E3/Em)_surrogate x Em_paper

Any clipped input is flagged in the output table.

Output: out/sensitivity/literature_validation.csv (+ console table)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
SENS = HERE / 'out' / 'sensitivity'
POINTS = SENS / 'literature_points.csv'
OUT = SENS / 'literature_validation.csv'

TDEG_AS = 25.8
ALPHA = 2.0

bundle = joblib.load(SENS / 'surrogates_tdeg.joblib')
hgb = bundle['exponential']['hgb']['E3_Em']

df = pd.read_csv(POINTS, comment='#')
rows = []
for r in df.itertuples():
    flags = []
    # --- volume fraction ---------------------------------------------
    if r.content_type == 'vol':
        vf = r.content_pct / 100.0
    else:
        w = r.content_pct / 100.0
        vf = (w / r.rho_f) / (w / r.rho_f + (1 - w) / r.rho_m)
    vf_c = float(np.clip(vf, 0.10, 0.30))
    if abs(vf_c - vf) > 1e-9:
        flags.append(f'Vf {vf:.2f}->{vf_c:.2f}')
    # --- modulus contrast --------------------------------------------
    efem = r.Ef_GPa / r.Em_GPa
    efem_c = float(np.clip(efem, 4.0, 48.0))
    if abs(efem_c - efem) > 1e-9:
        flags.append(f'Ef/Em {efem:.1f}->{efem_c:.1f}')
    # --- equivalent length through the aspect ratio ------------------
    l_eq = 10.0 * r.L_um / r.D_um
    l_c = float(np.clip(l_eq, 50.0, 200.0))
    if abs(l_c - l_eq) > 1e-9:
        flags.append(f'L_eq {l_eq:.0f}->{l_c:.0f}')
    # --- predict ------------------------------------------------------
    X = np.array([[vf_c, efem_c, ALPHA, np.log10(l_c), TDEG_AS]])
    e3 = hgb.predict(X)[0] * r.Em_GPa
    rows.append(dict(
        source=r.source, system=r.system,
        Vf=round(vf, 3), EfEm=round(efem, 1),
        L_um=r.L_um, D_um=r.D_um, L_eq=round(l_eq, 0),
        E_meas=r.E_meas_GPa, E_pred=round(e3, 2),
        dev_pct=round(100 * (e3 - r.E_meas_GPa) / r.E_meas_GPa, 1),
        clipped='; '.join(flags) if flags else '',
        notes=r.notes))

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print('\nwrote', OUT)
print(f"median |dev| {out.dev_pct.abs().median():.1f}%  "
      f"mean dev {out.dev_pct.mean():+.1f}%")
