"""Starter utilities for the stochastic-gradient-descent portion of Lab 04."""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def data_prep(X, y):
    """Add an intercept column and return arrays ready for matrix operations.

    Scaling and missing-value imputation should already be complete before this
    function is called. The returned design matrix and coefficient vector use
    the shapes needed for matrix multiplication: ``X`` is ``(n, p)`` and
    ``y`` is ``(n, 1)``.
    """
    poly = PolynomialFeatures(degree=1)
    X = poly.fit_transform(X)
    n, p = X.shape
    y = np.asarray(y).reshape(n, 1)
    return X, y, n, p


def sgd(X, y, alpha=0.1, epochs=750, decay=0.99, random_state=None):
    """Estimate linear-regression coefficients with stochastic gradient descent.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Imputed and scaled feature values. Do not add an intercept column here.
    y : array-like of shape (n_samples,)
        Target values corresponding to the rows of ``X``.
    alpha : float, default=0.1
        Initial learning rate.
    epochs : int, default=750
        Number of complete passes through the training observations.
    decay : float, default=0.99
        Multiplier applied to the learning rate after each epoch.
    random_state : int or None, default=None
        Optional seed for reproducible initialization and observation order.

    Returns
    -------
    numpy.ndarray of shape (n_features + 1, 1)
        The intercept followed by the feature coefficients.

    Notes for implementation
    ------------------------
    One SGD update uses **one observation**, unlike the batch-gradient update
    used in class. For each epoch, visit every observation once in a random
    order. For the selected row, calculate its prediction and residual, form
    the squared-error gradient for that one row, and update ``betas``. The
    per-observation gradient must have the same shape as ``betas``. Apply
    learning-rate decay only after the epoch is complete.
    """
    X, y, n, p = data_prep(X, y)
    rng = np.random.default_rng(random_state)
    betas = rng.standard_normal((p, 1))

    for _ in range(epochs):
        # Create a random ordering so that every observation is used once.
        # TODO: replace this placeholder with the ordering you will iterate over.
        observation_order = []

        for index in observation_order:
            # TODO: Select one row from X and its matching target value from y.
            # TODO: Compute the one-observation prediction and residual.
            # TODO: Compute the squared-error gradient for this observation.
            #       Check that its shape matches `betas` before updating betas.
            # TODO: Update betas using the current learning rate, alpha.
            pass

        # Decay after the epoch, not after each individual observation.
        alpha *= decay

    return betas
