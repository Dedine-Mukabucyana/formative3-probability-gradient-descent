"""Parts 3 & 4 - Gradient descent for  y_hat = X m + b  with MSE cost, in matrix form.

Shapes (n = number of data points, d = number of features):
    X : (n, d)   m : (d,)   b : (n,)  [one bias entry per data point, as given: b = [1, 1]]   y : (n,)
Cost:  J(m, b) = (1/n) * (y_hat - y)^T (y_hat - y)
"""
from __future__ import annotations

import numpy as np
from scipy.differentiate import derivative   # SciPy numerical differentiation


# ----------------------------------------------------------------------------
# Model and cost
# ----------------------------------------------------------------------------
def predict(X, m, b):
    """y_hat = X m + b"""
    return X @ m + b


def mse(X, m, b, y):
    """J = (1/n) (y_hat - y)^T (y_hat - y)"""
    u = predict(X, m, b) - y
    return float(u @ u / len(y))


# ----------------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------------
def analytic_gradients(X, m, b, y):
    """Chain rule with u = y_hat - y :  dJ/du = (2/n) u,  du/dm = X,  du/db = I.

    dJ/dm = (2/n) X^T u        dJ/db = (2/n) u
    """
    n = len(y)
    u = predict(X, m, b) - y
    return (2.0 / n) * X.T @ u, (2.0 / n) * u


def partial_derivative(f, theta: np.ndarray, i: int) -> float:
    """SciPy derivative of the equation ``f(theta)`` with respect to theta[i].

    All other parameters are frozen, which turns f into a one-variable function.
    """
    def f_along_i(t):
        # scipy passes an array of probe points t; evaluate f for each one
        out = np.empty_like(t, dtype=float)
        for idx, ti in np.ndenumerate(t):
            shifted = theta.astype(float).copy()
            shifted[i] = ti
            out[idx] = f(shifted)
        return out

    return float(derivative(f_along_i, theta[i]).df)


def scipy_gradient(f, theta: np.ndarray) -> np.ndarray:
    """Gradient vector of an equation f(theta): one SciPy derivative per parameter."""
    return np.array([partial_derivative(f, theta, i) for i in range(len(theta))])


def pack(m, b):
    """Stack m and b into one parameter vector theta = [m, b]."""
    return np.concatenate([m, b])


def unpack(theta, d):
    """Inverse of pack."""
    return theta[:d], theta[d:]


def cost_of_theta(theta, X, y):
    """The cost written as an equation of ONE vector argument (what SciPy differentiates)."""
    m, b = unpack(theta, X.shape[1])
    return mse(X, m, b, y)


# ----------------------------------------------------------------------------
# One visible update step + a loop that records everything
# ----------------------------------------------------------------------------
def gradient_step(X, y, m, b, lr, use_scipy=True):
    """One update. Every intermediate quantity is returned so it can be shown."""
    y_hat = predict(X, m, b)                        # 1) predictions
    error = y_hat - y                               # 2) residuals
    cost = float(error @ error / len(y))            # 3) MSE
    if use_scipy:                                   # 4) gradients (SciPy numerical derivative)
        grad = scipy_gradient(lambda t: cost_of_theta(t, X, y), pack(m, b))
        dm, db = unpack(grad, X.shape[1])
    else:                                           #    ... or the analytic matrix formula
        dm, db = analytic_gradients(X, m, b, y)
    m_new = m - lr * dm                             # 5) m <- m - lr * dJ/dm
    b_new = b - lr * db                             #    b <- b - lr * dJ/db
    return {"y_hat": y_hat, "error": error, "cost": cost, "dm": dm, "db": db,
            "m_new": m_new, "b_new": b_new}


def run_gradient_descent(X, y, m0, b0, lr, n_iter, use_scipy=True):
    """Repeat gradient_step n_iter times. Returns final (m, b) and a per-iteration history."""
    m, b = np.array(m0, float), np.array(b0, float)
    history = []
    for t in range(n_iter):
        s = gradient_step(X, y, m, b, lr, use_scipy)
        history.append({"iteration": t, "m": m.copy(), "b": b.copy(), **s})
        m, b = s["m_new"], s["b_new"]
    history.append({"iteration": n_iter, "m": m.copy(), "b": b.copy(),
                    "cost": mse(X, m, b, y), "y_hat": predict(X, m, b)})
    return m, b, history
