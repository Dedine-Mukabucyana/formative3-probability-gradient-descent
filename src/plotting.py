<<<<<<< HEAD
﻿import numpy as np
import matplotlib.pyplot as plt

from src.em import gaussian_pdf


def plot_mixture(heights, means, variances, proportions, test_height, output_path):
    """Plot observations and the fitted Gaussian mixture."""
    x_grid = np.linspace(heights.min() - 2, heights.max() + 2, 500)

    component_1 = proportions[0] * gaussian_pdf(
        x_grid, means[0], variances[0]
    )
    component_2 = proportions[1] * gaussian_pdf(
        x_grid, means[1], variances[1]
    )
    mixture_density = component_1 + component_2

    plt.figure(figsize=(10, 6))
    plt.hist(heights, bins=25, density=True, alpha=0.5,
             label="Observed heights")
    plt.plot(x_grid, component_1, linewidth=2,
             label="Gaussian component 1")
    plt.plot(x_grid, component_2, linewidth=2,
             label="Gaussian component 2")
    plt.plot(x_grid, mixture_density, linewidth=2.5,
             label="Combined mixture")
    plt.axvline(test_height, linestyle="--",
                label=f"Test height = {test_height}")
    plt.xlabel("Height (inches)")
    plt.ylabel("Probability density")
    plt.title("Two-Component Gaussian Mixture Fitted Using EM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
=======
"""Plain, reusable plotting helpers shared by every part (DRY)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.em import EMParams, e_step, gaussian_pdf


def use_plain_style():
    """One consistent, minimal look for every figure."""
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
                         "axes.titleweight": "bold", "font.size": 10})


def save(fig, path):
    """Save a figure when a path is given."""
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_lines(x, series: dict, title, xlabel, ylabel, path=None, logy=False, logx=False,
               vlines=None):
    """Several named series against x. vlines: optional {label: x position} markers."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, y in series.items():
        ax.plot(x, y, lw=2, label=name)
    for label, pos in (vlines or {}).items():
        ax.axvline(pos, color="grey", ls=":", lw=1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("symlog", linthresh=3)
    ax.legend()
    save(fig, path)
    return fig


def _mixture_curves(grid, p: EMParams):
    comps = [p.proportions[k] * gaussian_pdf(grid, p.means[k], p.variances[k]) for k in range(2)]
    return comps, comps[0] + comps[1]


def plot_mixture_fit(x, p: EMParams, title, test_height=None, cut=None, path=None):
    """Histogram of pooled heights with the two fitted components and their sum."""
    grid = np.linspace(x.min() - 2, x.max() + 2, 400)
    comps, total = _mixture_curves(grid, p)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(x, bins=28, density=True, alpha=0.35, color="grey", label="pooled heights (no labels)")
    for k, c in enumerate(comps):
        ax.plot(grid, c, lw=2, label=f"component {k + 1}: mu={p.means[k]:.2f}, "
                                      f"sigma={p.variances[k] ** 0.5:.2f}, pi={p.proportions[k]:.2f}")
    ax.plot(grid, total, "k--", lw=1.4, label="mixture density")
    if test_height is not None:
        ax.axvline(test_height, color="black", ls="-.", lw=1, label=f"test height = {test_height}")
    if cut is not None:
        ax.axvline(cut, color="grey", ls=":", lw=2, label=f"global-mean cut = {cut:.2f}")
    ax.set(title=title, xlabel="height (inches)", ylabel="density")
    ax.legend(fontsize=8)
    save(fig, path)
    return fig


def plot_em_snapshots(x, history, steps=(0, 1, 2, None), path=None):
    """Fitted mixture at the initial state, iterations 1-2, and at convergence."""
    grid = np.linspace(x.min() - 2, x.max() + 2, 400)
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 3.4), sharey=True)
    for ax, s in zip(axes, steps):
        i = len(history) - 1 if s is None else s
        p, ll = history[i]
        comps, total = _mixture_curves(grid, p)
        ax.hist(x, bins=28, density=True, color="grey", alpha=0.35)
        ax.plot(grid, comps[0], lw=2)
        ax.plot(grid, comps[1], lw=2)
        ax.plot(grid, total, "k--", lw=1.2)
        name = "Initial state" if i == 0 else (f"Converged (iteration {i})" if s is None else f"Iteration {i}")
        ax.set_title(f"{name}\nlog-likelihood = {ll:,.1f}", fontsize=10)
        ax.set_xlabel("height (inches)")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, path)
    return fig


def plot_loglik(history, path=None):
    """Log-likelihood per iteration: its flattening is the stopping rule."""
    ll = [h[1] for h in history]
    return plot_lines(range(len(ll)), {"log-likelihood": ll}, "EM never decreases the log-likelihood",
                      "iteration (symmetric-log axis)", "log-likelihood", path=path, logx=True)


def plot_soft_vs_hard(x, p: EMParams, cut, path=None):
    """EM's soft probability curves versus a hard cut at the global mean."""
    grid = np.linspace(x.min(), x.max(), 400)
    r = e_step(grid, p)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(grid, r[:, 0], lw=2.2, label="EM: P(component 1 | height)")
    ax.plot(grid, r[:, 1], lw=2.2, label="EM: P(component 2 | height)")
    ax.step(grid, (grid > cut).astype(float), color="black", ls="--", where="post",
            label="global-mean split (only 0 or 1)")
    ax.set(title="Soft assignments keep the uncertainty; a hard cut discards it",
           xlabel="height (inches)", ylabel="probability")
    ax.legend(fontsize=8, loc="center left")
    save(fig, path)
    return fig


def plot_confusion(cm, names, accuracy, path=None):
    """Annotated confusion matrix (rows = actual, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.imshow(cm, cmap="Greys")
    ax.grid(False)
    ax.set_xticks(range(2), [f"predicted {n}" for n in names])
    ax.set_yticks(range(2), [f"actual {n}" for n in names])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix (accuracy {accuracy:.1%})")
    fig.tight_layout()
    save(fig, path)
    return fig


def plot_posteriors(words, posteriors, prior, n_pos, path=None):
    """Bars of P(Positive | keyword); the first n_pos words are the chosen positive keywords."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(words, posteriors, color=["0.35" if i < n_pos else "0.75" for i in range(len(words))])
    ax.axhline(prior, color="black", ls="--", lw=1)
    ax.text(len(words) - 0.5, prior + 0.02, f"prior = {prior:.2f}", ha="right", fontsize=9)
    for bar, value in zip(bars, posteriors):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center")
    ax.set(ylim=(0, 1), ylabel="P(Positive | keyword)",
           title="Posterior per keyword (dark = chosen positive words, light = chosen negative words)")
    save(fig, path)
    return fig


def plot_predictions(y, snapshots, path=None):
    """True targets next to predictions at several iterations."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    n, w = len(y), 0.8 / (len(snapshots) + 1)
    idx = np.arange(n)
    ax.bar(idx, y, width=w, color="black", label="true y")
    for j, (label, y_hat) in enumerate(snapshots):
        ax.bar(idx + (j + 1) * w, y_hat, width=w, label=label)
    ax.set_xticks(idx + 0.4 - w / 2, [f"point {i + 1}" for i in idx])
    ax.set(title="Predictions move toward the targets", ylabel="value")
    ax.legend(fontsize=8)
    save(fig, path)
    return fig
>>>>>>> origin/main
