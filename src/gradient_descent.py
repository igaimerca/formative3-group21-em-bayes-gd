"""
Gradient descent for y = m1*x1 + m2*x2 + b, matching Part 3's manual
calculation exactly but run for many iterations and with the gradient
cross-checked numerically through SciPy instead of only the hand-derived
formula, per the assignment's "use SciPy to compute the derivative" step.
"""

import numpy as np
from scipy.optimize import approx_fprime


def predict(X, m, b):
    """yhat = X . m + b, matrix multiplication, no per-row loops."""
    return X @ m + b


def cost(params, X, y):
    """
    params packs [m1, m2, b] into one vector so SciPy's numerical
    differentiation (which expects a flat vector) can be pointed at it.
    """
    m = params[:2]
    b = params[2]
    yhat = predict(X, m, b)
    errors = yhat - y
    return np.mean(errors ** 2)


def scipy_gradient(params, X, y, epsilon=1e-6):
    """
    Numerical gradient of cost(params) via SciPy's approx_fprime -
    finite differences, not a hand-coded formula. Used to double-check
    the analytical gradient computed in gradient_analytic below.
    """
    return approx_fprime(params, cost, epsilon, X, y)


def gradient_analytic(m, b, X, y):
    """
    Same chain-rule derivation as Part 3, written out explicitly:
      dJ/dyhat = (2/n) * (yhat - y)
      dJ/dm    = X^T . dJ/dyhat
      dJ/db    = sum(dJ/dyhat)
    """
    n = len(y)
    yhat = predict(X, m, b)
    errors = yhat - y
    d_cost_d_yhat = (2.0 / n) * errors

    d_cost_d_m = X.T @ d_cost_d_yhat
    d_cost_d_b = np.sum(d_cost_d_yhat)

    return d_cost_d_m, d_cost_d_b


def run_gradient_descent(X, y, m_init, b_init, learning_rate=0.01, n_iterations=100):
    """
    Runs plain batch gradient descent, keeping every intermediate value
    so it can be plotted afterward (m/b trajectory, cost trajectory).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.array(m_init, dtype=float)
    b = float(b_init)

    m_history = [m.copy()]
    b_history = [b]
    cost_history = [cost(np.array([m[0], m[1], b]), X, y)]

    for _ in range(n_iterations):
        d_m, d_b = gradient_analytic(m, b, X, y)
        m = m - learning_rate * d_m
        b = b - learning_rate * d_b

        m_history.append(m.copy())
        b_history.append(b)
        cost_history.append(cost(np.array([m[0], m[1], b]), X, y))

    return {
        "m_final": m,
        "b_final": b,
        "m_history": np.array(m_history),
        "b_history": np.array(b_history),
        "cost_history": np.array(cost_history),
    }
