"""Polynomial surrogate used for comparison with the GP.

scikit-learn PolynomialFeatures + LinearRegression on inputs scaled to
[-1, 1] (scikit-learn User Guide, 1.1.17 Polynomial regression).

An input with k design levels determines its powers up to k - 1 only
(v_f 5 levels, Ef/Em 7, alpha 3, log10 L_p 3, tdeg 4). With
drop_undetermined=True the monomials with a higher power are left out, so
the model holds every term of total degree <= degree that the factorial
design determines (113 terms for degree 4). With drop_undetermined=False
all monomials are kept: the fit on the design grid is the same, but the
coefficients of the undetermined terms are arbitrary and the prediction
between the levels is not defined. predict(X) takes the raw features
[v_f, Ef/Em, alpha, log10 L_p, tdeg].
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

LEVELS = (5, 7, 3, 3, 4)   # design levels of v_f, Ef/Em, alpha, log10 L_p, tdeg


class PolynomialTerms(BaseEstimator, TransformerMixin):
    """PolynomialFeatures(degree), optionally without the monomials whose
    power in an input exceeds its number of design levels minus one."""

    def __init__(self, degree=4, drop_undetermined=True, levels=LEVELS):
        self.degree = degree
        self.drop_undetermined = drop_undetermined
        self.levels = levels

    def fit(self, X, y=None):
        self.poly_ = PolynomialFeatures(self.degree).fit(X)
        powers = self.poly_.powers_
        if self.drop_undetermined:
            self.keep_ = np.all(powers <= np.asarray(self.levels) - 1, axis=1)
        else:
            self.keep_ = np.ones(len(powers), dtype=bool)
        self.n_terms_ = int(self.keep_.sum())
        return self

    def transform(self, X):
        return self.poly_.transform(X)[:, self.keep_]


def poly_surrogate(degree=4, drop_undetermined=True):
    """Polynomial surrogate with the same interface as gp_surrogate()."""
    return make_pipeline(MinMaxScaler(feature_range=(-1, 1)),
                         PolynomialTerms(degree, drop_undetermined),
                         LinearRegression())
