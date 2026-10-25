import numpy as np


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
