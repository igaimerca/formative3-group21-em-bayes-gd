"""
Expectation-Maximization for a 2-component 1D Gaussian Mixture Model,
built from raw arithmetic (no sklearn.mixture, no black-box EM calls).

Only numpy is used, and only for vectorized arithmetic (exp, sum, mean) -
none of numpy's fitting/estimation routines are touched.
"""

import numpy as np


def gaussian_density(x, mu, var):
    """Standard univariate normal density, written out term by term."""
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = -((x - mu) ** 2) / (2.0 * var)
    return coeff * np.exp(exponent)


def e_step(x, mu1, var1, pi1, mu2, var2, pi2):
    """
    Compute responsibilities: gamma_i = P(component 2 | x_i).
    This is the soft assignment step - every point gets a probability
    for BOTH components instead of being forced into one bucket.
    """
    weighted1 = pi1 * gaussian_density(x, mu1, var1)
    weighted2 = pi2 * gaussian_density(x, mu2, var2)
    total = weighted1 + weighted2
    gamma2 = weighted2 / total   # posterior prob of belonging to component 2
    gamma1 = 1.0 - gamma2
    return gamma1, gamma2


def m_step(x, gamma1, gamma2):
    """Re-estimate mu, var, pi for each component from the responsibilities."""
    n = len(x)
    n1 = gamma1.sum()
    n2 = gamma2.sum()

    mu1 = np.sum(gamma1 * x) / n1
    mu2 = np.sum(gamma2 * x) / n2

    var1 = np.sum(gamma1 * (x - mu1) ** 2) / n1
    var2 = np.sum(gamma2 * (x - mu2) ** 2) / n2

    pi1 = n1 / n
    pi2 = n2 / n

    return mu1, var1, pi1, mu2, var2, pi2


def log_likelihood(x, mu1, var1, pi1, mu2, var2, pi2):
    """Sum of log( pi1*N(x|mu1,var1) + pi2*N(x|mu2,var2) ) over all points."""
    mix = pi1 * gaussian_density(x, mu1, var1) + pi2 * gaussian_density(x, mu2, var2)
    return np.sum(np.log(mix))


def run_em(x, mu1_init, var1_init, pi1_init, mu2_init, var2_init, pi2_init,
           n_iter=2, track_history=True):
    """
    Runs EM for n_iter full E/M cycles and returns the parameter history.

    history[0] is always the initialization (iteration 0, before any update),
    matching the assignment's requirement of an "init" row plus iter 1, iter 2, ...
    """
    x = np.asarray(x, dtype=float)
    mu1, var1, pi1 = mu1_init, var1_init, pi1_init
    mu2, var2, pi2 = mu2_init, var2_init, pi2_init

    history = []
    ll0 = log_likelihood(x, mu1, var1, pi1, mu2, var2, pi2)
    history.append({
        "iteration": 0, "mu1": mu1, "var1": var1, "pi1": pi1,
        "mu2": mu2, "var2": var2, "pi2": pi2, "log_likelihood": ll0,
    })

    for it in range(1, n_iter + 1):
        gamma1, gamma2 = e_step(x, mu1, var1, pi1, mu2, var2, pi2)
        mu1, var1, pi1, mu2, var2, pi2 = m_step(x, gamma1, gamma2)
        ll = log_likelihood(x, mu1, var1, pi1, mu2, var2, pi2)
        if track_history:
            history.append({
                "iteration": it, "mu1": mu1, "var1": var1, "pi1": pi1,
                "mu2": mu2, "var2": var2, "pi2": pi2, "log_likelihood": ll,
            })

    final_params = {
        "mu1": mu1, "var1": var1, "pi1": pi1,
        "mu2": mu2, "var2": var2, "pi2": pi2,
    }
    return history, final_params


def posterior_for_point(x_new, params):
    """
    Given final fitted params, return P(component1 | x_new), P(component2 | x_new)
    for a brand-new, unseen value - this is the "live demo" classification step.
    """
    g1, g2 = e_step(
        np.array([x_new]),
        params["mu1"], params["var1"], params["pi1"],
        params["mu2"], params["var2"], params["pi2"],
    )
    return float(g1[0]), float(g2[0])
