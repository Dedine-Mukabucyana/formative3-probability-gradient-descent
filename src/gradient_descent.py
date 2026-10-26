"""Parts 3 & 4 - gradient descent for  y_hat = X m + b  with MSE cost, in matrix form.

Shapes (n data points, d features):  X (n, d)   m (d,)   b (n,)   y (n,)
b has one entry per data point, exactly as given in the assignment (b = [1, 1]).
Cost:  J(m, b) = (1/n) (y_hat - y)^T (y_hat - y)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.differentiate import derivative   # SciPy numerical differentiation (SciPy >= 1.15)


# ---------------------------------------------------------------- model and cost
def predict(X, m, b):
    """y_hat = X m + b"""
    return X @ m + b


def mse(X, m, b, y):
    """J = (1/n) u^T u  with  u = y_hat - y"""
    u = predict(X, m, b) - y
    return float(u @ u / len(y))


# ---------------------------------------------------------------- gradients
def analytic_gradients(X, m, b, y):
    """Chain rule with u = y_hat - y:  dJ/du = (2/n) u,  du/dm = X,  du/db = I.

    dJ/dm = (2/n) X^T u        dJ/db = (2/n) u
    """
    u = predict(X, m, b) - y
    n = len(y)
    return (2.0 / n) * X.T @ u, (2.0 / n) * u


def partial_derivative(f, theta: np.ndarray, i: int) -> float:
    """SciPy derivative of the equation f(theta) with respect to theta[i] (others frozen)."""
    def f_along_i(t):
        out = np.empty_like(t, dtype=float)
        for idx, ti in np.ndenumerate(t):          # SciPy passes an array of probe points
            shifted = theta.astype(float).copy()
            shifted[i] = ti
            out[idx] = f(shifted)
        return out

    return float(derivative(f_along_i, theta[i]).df)


def scipy_gradient(f, theta: np.ndarray) -> np.ndarray:
    """Gradient of an equation f(theta): one SciPy derivative per parameter."""
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


# ---------------------------------------------------------------- updates
def gradient_step(X, y, m, b, lr, use_scipy=True):
    """One update. Every intermediate quantity is returned so it can be shown."""
    y_hat = predict(X, m, b)                        # 1) predictions
    error = y_hat - y                               # 2) residuals
    cost = float(error @ error / len(y))            # 3) MSE
    if use_scipy:                                   # 4) gradients
        grad = scipy_gradient(lambda t: cost_of_theta(t, X, y), pack(m, b))
        dm, db = unpack(grad, X.shape[1])
    else:
        dm, db = analytic_gradients(X, m, b, y)
    return {"y_hat": y_hat, "error": error, "cost": cost, "dm": dm, "db": db,
            "m_new": m - lr * dm, "b_new": b - lr * db}   # 5) updates


def run_gradient_descent(X, y, m0, b0, lr, n_iter, use_scipy=True):
    """Repeat gradient_step n_iter times. Returns final (m, b) and a per-iteration history."""
    m, b = np.array(m0, float), np.array(b0, float)
    history = []
    for t in range(n_iter):
        step = gradient_step(X, y, m, b, lr, use_scipy)
        history.append({"iteration": t, "m": m.copy(), "b": b.copy(), **step})
        m, b = step["m_new"], step["b_new"]
    history.append({"iteration": n_iter, "m": m.copy(), "b": b.copy(),
                    "cost": mse(X, m, b, y), "y_hat": predict(X, m, b)})
    return m, b, history


# ---------------------------------------------------------------- reporting helpers
def history_arrays(history):
    """Iteration numbers, m values, b values and costs as arrays (for plotting)."""
    its = np.array([h["iteration"] for h in history])
    M = np.array([h["m"] for h in history])
    B = np.array([h["b"] for h in history])
    J = np.array([h["cost"] for h in history])
    return its, M, B, J


def update_table(history, n_updates, owners=None) -> pd.DataFrame:
    """One row per update, with every intermediate quantity (for the presentation)."""
    rows = []
    for t in range(n_updates):
        h = history[t]
        row = {"Update": t + 1}
        if owners:
            row["Member"] = owners[t]
        row.update({"m before": np.round(h["m"], 4), "b before": np.round(h["b"], 4),
                    "y_hat": np.round(h["y_hat"], 4), "error": np.round(h["error"], 4),
                    "J": round(h["cost"], 4), "dJ/dm": np.round(h["dm"], 4),
                    "dJ/db": np.round(h["db"], 4), "m after": np.round(h["m_new"], 4),
                    "b after": np.round(h["b_new"], 4)})
        rows.append(row)
    return pd.DataFrame(rows)
