"""Functions used for data transformations."""
import warnings
import numpy as np
from scipy.stats.mstats import gmean


def prop(x):
    """Transform to proportional data."""
    x_prop = x.divide(x.sum(axis=1), axis=0)
    x_prop.fillna(0.0, inplace=True)  # If a row has all zeros
    return x_prop


def alr(x, denom_ind=None):
    """Perform additive log-ratio transformation."""
    if denom_ind is None:
        denom_ind = (x == 0).sum(axis=0).argmin()
    x_alr = np.log(x).sub(np.log(x.iloc[:, denom_ind]), axis=0)
    x_alr = x_alr.replace(-np.inf, -100)
    return x_alr, denom_ind


def clr(x, pseudo_count=None):
    """Perform centered log-ratio transformation."""
    if pseudo_count is None:
        pseudo_count = x[x != 0].min().min() / 2
        pseudo_count = np.min((pseudo_count, 1.0))
    x = x + pseudo_count
    return np.log(x).sub(np.log(gmean(x, 1)), axis=0)


def data_transformation(X, transformation_type, transformation_opts):
    """Apply data transformation after splitting into train/test."""
    if transformation_type == 'prop':
        X = prop(X)
    elif transformation_type == 'alr':
        if 'denom_ind' in transformation_opts:
            denom_ind = transformation_opts['denom_ind']
            X, _ = alr(X, denom_ind)
        else:
            X, denom_ind = alr(X)
            transformation_opts['denom_ind'] = denom_ind
    elif transformation_type == 'clr':
        if 'pseudo_count' in transformation_opts:
            pseudo_count = transformation_opts['pseudo_count']
        else:
            pseudo_count = None
        X = clr(X, pseudo_count)
    elif transformation_type == 'none':
        return X, transformation_opts
    else:
        warnings.warn('Invalid transformation_type. '
                      + 'No data transformation performed.')
    return X, transformation_opts