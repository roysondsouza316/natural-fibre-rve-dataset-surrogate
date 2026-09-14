"""Shared definition of the GP surrogate of record and the waviness feature."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

FEATURES = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio', 'logL', 'tdeg']


def theta_deg(kappa):
    """Mean deviation inclination (deg) of the vMF backbone; 0 = straight."""
    if kappa == 0:
        return 0.0
    return float(np.degrees(np.arccos(1 / np.tanh(kappa) - 1 / kappa)))


def gp_surrogate(restarts=1):
    """GP of record: Matern-5/2 on standardised inputs, fitted to the log of
    the constant (all targets are positive), with a noise term that absorbs
    the RVE realisation scatter. predict() takes the raw features
    [v_f, Ef/Em, alpha, log10 L_p, tdeg] and returns the constant in its
    original units."""
    kern = (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(5), length_scale_bounds=(0.5, 10.0),
                     nu=2.5)
            + WhiteKernel(1e-3, (1e-9, 1e-1)))
    return make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(
            regressor=GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                               n_restarts_optimizer=restarts,
                                               random_state=0),
            func=np.log, inverse_func=np.exp))
