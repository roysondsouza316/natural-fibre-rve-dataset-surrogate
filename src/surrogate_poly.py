"""Polynomial comparison surrogate.

Ordinary polynomial regression of total degree 4 in the five inputs
[v_f, Ef/Em, alpha, log10 L_p, tdeg], scaled to [-1, 1]. The power of each
variable is limited to its number of design levels minus one (4, 6, 2, 2, 3),
because higher powers cannot be determined from the factorial grid and run
away between the levels. Fitted by least squares to the logarithm of each
constant, as the GP surrogate. `predict(X)` takes the same raw features as
the GP.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

TOTAL_DEGREE = 4
MAX_POWER = (4, 6, 2, 2, 3)   # design levels - 1 for v_f, Ef/Em, alpha, log L_p, tdeg


class CappedPolynomialFeatures(BaseEstimator, TransformerMixin):
    """Monomials up to `total_degree`, each variable at most `max_power`."""

    def __init__(self, total_degree=TOTAL_DEGREE, max_power=MAX_POWER):
        self.total_degree = total_degree
        self.max_power = max_power

    def fit(self, X, y=None):
        self.poly_ = PolynomialFeatures(self.total_degree).fit(X)
        self.keep_ = np.all(self.poly_.powers_ <= np.asarray(self.max_power), axis=1)
        self.n_terms_ = int(self.keep_.sum())
        return self

    def transform(self, X):
        return self.poly_.transform(X)[:, self.keep_]


def poly_surrogate():
    """Polynomial surrogate with the same interface as gp_surrogate()."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(MinMaxScaler(feature_range=(-1, 1)),
                                CappedPolynomialFeatures(),
                                LinearRegression()),
        func=np.log, inverse_func=np.exp)
