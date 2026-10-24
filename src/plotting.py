"""Reusable plotting helpers (DRY: every figure in the notebook goes through these)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def save(fig, path: str | None):
    """Save a figure if a path is given."""
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_lines(x, series: dict, title: str, xlabel: str, ylabel: str, path=None, logy=False):
    """Plot several named series against x."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, y in series.items():
        ax.plot(x, y, marker="o" if len(x) <= 15 else None, lw=2, label=name)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path)
    return fig


def plot_mixture(x, labels, p, cut=None, title="", path=None):
    """Histogram of pooled heights with the fitted mixture components."""
    from src.em_gmm import gaussian_pdf
    grid = np.linspace(x.min() - 2, x.max() + 2, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(x, bins=30, density=True, alpha=0.35, color="grey", label="pooled heights (no labels)")
    for k, c in enumerate(["tab:blue", "tab:red"]):
        ax.plot(grid, p.pi[k] * gaussian_pdf(grid, p.mu[k], p.var[k]), c=c, lw=2,
                label=f"component {k + 1}: mu={p.mu[k]:.2f}, sigma={p.var[k] ** 0.5:.2f}, pi={p.pi[k]:.2f}")
    total = sum(p.pi[k] * gaussian_pdf(grid, p.mu[k], p.var[k]) for k in range(2))
    ax.plot(grid, total, "k--", lw=1.5, label="mixture density")
    if cut is not None:
        ax.axvline(cut, c="green", ls=":", lw=2, label=f"global-mean cut = {cut:.2f}")
    ax.set(title=title, xlabel="height (inches)", ylabel="density")
    ax.legend(fontsize=8)
    save(fig, path)
    return fig
