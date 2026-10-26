<<<<<<< HEAD
﻿import numpy as np


def gaussian_log_pdf(x, mean, variance):
    """Calculate the log probability density of a Gaussian."""
    variance = max(float(variance), 1e-6)
    return -0.5 * (
        np.log(2 * np.pi * variance)
        + ((x - mean) ** 2) / variance
    )


def gaussian_pdf(x, mean, variance):
    """Calculate the probability density of a Gaussian."""
    return np.exp(gaussian_log_pdf(x, mean, variance))


def calculate_log_likelihood(x, means, variances, proportions):
    """Calculate the total log-likelihood of a two-component mixture."""
    log_components = np.column_stack([
        np.log(proportions[k]) + gaussian_log_pdf(x, means[k], variances[k])
        for k in range(2)
    ])
    maximum = np.max(log_components, axis=1)
    return float(np.sum(
        maximum + np.log(np.sum(np.exp(log_components - maximum[:, None]), axis=1))
    ))


def expectation_step(x, means, variances, proportions):
    """Calculate each observation's responsibility for each component."""
    log_components = np.column_stack([
        np.log(proportions[k]) + gaussian_log_pdf(x, means[k], variances[k])
        for k in range(2)
    ])
    maximum = np.max(log_components, axis=1, keepdims=True)
    probabilities = np.exp(log_components - maximum)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def maximization_step(x, responsibilities):
    """Update means, variances, and mixture proportions."""
    effective_counts = responsibilities.sum(axis=0)
    if np.any(effective_counts <= 1e-12):
        raise ValueError("A mixture component has too few effective observations.")

    means = (responsibilities.T @ x) / effective_counts
    variances = np.array([
        np.sum(responsibilities[:, k] * (x - means[k]) ** 2)
        / effective_counts[k]
        for k in range(2)
    ])
    variances = np.maximum(variances, 1e-6)
    proportions = effective_counts / len(x)
    return means, variances, proportions


def fit_em(x, tolerance=1e-6, max_iterations=1000):
    """Fit a two-component Gaussian mixture and record the first two updates."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Input data must be a non-empty array of finite numbers.")

    means = np.percentile(x, [25, 75]).astype(float)
    initial_variance = np.var(x)
    variances = np.array([initial_variance, initial_variance], dtype=float)
    proportions = np.array([0.5, 0.5], dtype=float)

    def state(iteration):
        return {
            "Iteration": iteration,
            "Mean 1": means[0],
            "Mean 2": means[1],
            "Variance 1": variances[0],
            "Variance 2": variances[1],
            "Mixing proportion 1": proportions[0],
            "Mixing proportion 2": proportions[1],
            "Log-likelihood": calculate_log_likelihood(
                x, means, variances, proportions
            ),
        }

    history = [state(0)]
    previous_log_likelihood = history[0]["Log-likelihood"]

    for iteration in range(1, max_iterations + 1):
        responsibilities = expectation_step(x, means, variances, proportions)
        means, variances, proportions = maximization_step(x, responsibilities)
        current_log_likelihood = calculate_log_likelihood(
            x, means, variances, proportions
        )

        if iteration <= 2:
            history.append(state(iteration))

        if abs(current_log_likelihood - previous_log_likelihood) < tolerance:
            break

        previous_log_likelihood = current_log_likelihood

    return {
        "means": means,
        "variances": variances,
        "proportions": proportions,
        "history": history,
        "iterations": iteration,
        "log_likelihood": calculate_log_likelihood(
            x, means, variances, proportions
        ),
    }
=======
"""Part 1 - Expectation-Maximization for a two-component Gaussian mixture (from scratch).

Model: p(x) = pi_1 N(x | mu_1, var_1) + pi_2 N(x | mu_2, var_2)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_VARIANCE = 1e-6


@dataclass
class EMParams:
    """The six numbers EM learns: means, variances, mixing proportions."""
    means: np.ndarray
    variances: np.ndarray
    proportions: np.ndarray


# ---------------------------------------------------------------- Gaussian pieces
def gaussian_log_pdf(x, mean, variance):
    """Log of the Gaussian density (works element-wise and with broadcasting)."""
    variance = np.maximum(variance, MIN_VARIANCE)
    return -0.5 * (np.log(2 * np.pi * variance) + (x - mean) ** 2 / variance)


def gaussian_pdf(x, mean, variance):
    return np.exp(gaussian_log_pdf(x, mean, variance))


def _weighted_log_densities(x, p: EMParams):
    """log(pi_k) + log N(x_i | mu_k, var_k) for every point i and component k -> (n, K)."""
    x = np.asarray(x, float).reshape(-1, 1)
    return np.log(p.proportions) + gaussian_log_pdf(x, p.means, p.variances)


def _log_sum_exp(a):
    """Row-wise log(sum(exp(a))) computed stably."""
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))).ravel()


def log_likelihood(x, p: EMParams) -> float:
    """Total log-likelihood of the data under the mixture."""
    return float(_log_sum_exp(_weighted_log_densities(x, p)).sum())


# ---------------------------------------------------------------- the algorithm
def initialize(x) -> EMParams:
    """Means at the 25th/75th percentiles, equal variances (data variance), equal weights."""
    return EMParams(
        means=np.percentile(x, [25, 75]).astype(float),
        variances=np.full(2, np.var(x)),
        proportions=np.array([0.5, 0.5]),
    )


def e_step(x, p: EMParams):
    """Responsibilities r[i, k] = P(component k | x_i): soft assignments, rows sum to 1."""
    log_w = _weighted_log_densities(x, p)
    return np.exp(log_w - _log_sum_exp(log_w)[:, None])


def m_step(x, r) -> EMParams:
    """Re-estimate the parameters using the responsibilities as weights."""
    x = np.asarray(x, float)
    n_k = r.sum(axis=0)                                   # effective number of points per component
    means = (r.T @ x) / n_k
    variances = (r * (x[:, None] - means) ** 2).sum(axis=0) / n_k
    return EMParams(means, np.maximum(variances, MIN_VARIANCE), n_k / len(x))


def fit(x, params: EMParams | None = None, tol: float = 1e-6, max_iter: int = 1000):
    """Run EM until the log-likelihood changes by less than tol.

    Returns history = [(params, log_likelihood), ...]; entry 0 is the initial state.
    """
    p = params if params is not None else initialize(x)
    history = [(p, log_likelihood(x, p))]
    for _ in range(max_iter):
        p = m_step(x, e_step(x, p))
        history.append((p, log_likelihood(x, p)))
        if abs(history[-1][1] - history[-2][1]) < tol:
            break
    return history


# ---------------------------------------------------------------- reporting helpers
def tracking_table(history, n_iterations: int = 2) -> pd.DataFrame:
    """The required table: initial state plus the first n_iterations iterations."""
    rows = []
    for i, (p, ll) in enumerate(history[: n_iterations + 1]):
        rows.append({"Iteration": i, "mu1": p.means[0], "mu2": p.means[1],
                     "sigma1^2": p.variances[0], "sigma2^2": p.variances[1],
                     "pi1": p.proportions[0], "pi2": p.proportions[1], "Log-likelihood": ll})
    return pd.DataFrame(rows)


def posterior(height: float, p: EMParams):
    """P(component 1 | height), P(component 2 | height) for one height."""
    return e_step(np.array([float(height)]), p)[0]


def global_mean_split(x):
    """The naive alternative: cut at the global mean and summarise each pile."""
    cut = float(np.mean(x))
    low, high = x[x <= cut], x[x > cut]
    params = EMParams(np.array([low.mean(), high.mean()]), np.array([low.var(), high.var()]),
                      np.array([len(low), len(high)]) / len(x))
    return cut, params


def closest_component(p: EMParams, reference_heights) -> int:
    """Index of the component whose mean is closest to the mean of a labelled reference group."""
    return int(np.argmin(np.abs(p.means - np.mean(reference_heights))))


def evaluate(x, labels, p: EMParams, parent_component: int):
    """Compare hard EM assignments with the true labels (0 = parent, 1 = child)."""
    predicted_component = e_step(x, p).argmax(axis=1)
    predicted = np.where(predicted_component == parent_component, 0, 1)
    confusion = np.zeros((2, 2), int)
    np.add.at(confusion, (labels, predicted), 1)
    recall = confusion.diagonal() / confusion.sum(axis=1)
    return {"confusion": confusion,
            "accuracy": float((predicted == labels).mean()),
            "balanced_accuracy": float(recall.mean()),
            "majority_baseline": float(max(np.mean(labels), 1 - np.mean(labels)))}


def classify(height: float, p: EMParams, parent_component: int):
    """Posterior probabilities (P(parent group), P(child group)) for one height."""
    r = posterior(height, p)
    return float(r[parent_component]), float(r[1 - parent_component])
>>>>>>> origin/main
