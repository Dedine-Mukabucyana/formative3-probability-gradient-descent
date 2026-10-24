"""Presentation helpers: keep the notebook short by putting plotting / styling code here (DRY)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

BLUE, RED, GREEN, ORANGE, GREY, PURPLE = "#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#6b7280", "#7c3aed"


def set_style():
    """One consistent look for every figure."""
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.frameon": False, "font.size": 10,
    })


def callout(title: str, body: str, color: str = BLUE):
    """Coloured highlight box (works in light and dark themes)."""
    display(HTML(
        f'<div style="border-left:6px solid {color};background:{color}1a;padding:10px 16px;'
        f'border-radius:6px;margin:8px 0"><b style="color:{color}">{title}</b><br>{body}</div>'))


def banner(number: str, title: str, subtitle: str, color: str = BLUE):
    """Big section banner."""
    display(HTML(
        f'<div style="background:linear-gradient(90deg,{color},{color}aa);color:white;padding:18px 24px;'
        f'border-radius:10px;margin:18px 0 8px 0"><span style="font-size:13px;opacity:.85">{number}</span>'
        f'<div style="font-size:26px;font-weight:700">{title}</div>'
        f'<div style="font-size:14px;opacity:.9">{subtitle}</div></div>'))


def normal_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


# --------------------------------------------------------------------------- Part 1
def run_em_history(x, means, variances, proportions, e_step, m_step, loglik, max_iter=1000, tol=1e-6):
    """Re-run EM with the group's own e/m-step functions, recording EVERY iteration (for the visuals)."""
    hist = [(np.array(means, float), np.array(variances, float), np.array(proportions, float),
             loglik(x, means, variances, proportions))]
    prev = hist[0][3]
    for _ in range(max_iter):
        r = e_step(x, *hist[-1][:3])
        mu, var, pi = m_step(x, r)
        ll = loglik(x, mu, var, pi)
        hist.append((mu, var, pi, ll))
        if abs(ll - prev) < tol:
            break
        prev = ll
    return hist


def plot_data_overview(heights, actual_labels, names=("Mother", "Child")):
    """Left: what EM sees (no labels). Right: the hidden truth - the groups overlap heavily."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    bins = np.linspace(heights.min(), heights.max(), 30)
    ax[0].hist(heights, bins=bins, color=GREY, alpha=.8)
    ax[0].set(title="What EM sees: pooled heights, no labels", xlabel="height (inches)", ylabel="count")
    for k, c in enumerate([ORANGE, BLUE]):
        ax[1].hist(heights[actual_labels == k], bins=bins, alpha=.65, color=c, label=names[k])
    ax[1].set(title="The hidden truth: the two groups overlap", xlabel="height (inches)")
    ax[1].legend()
    plt.tight_layout(); plt.show()


def plot_em_evolution(x, hist, steps=(0, 1, 2, None)):
    """Fitted mixture at the initial state, iterations 1 and 2, and at convergence."""
    grid = np.linspace(x.min() - 2, x.max() + 2, 400)
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 3.6), sharey=True)
    for ax, s in zip(axes, steps):
        i = len(hist) - 1 if s is None else s
        mu, var, pi, ll = hist[i]
        ax.hist(x, bins=28, density=True, color=GREY, alpha=.35)
        c1, c2 = pi[0] * normal_pdf(grid, mu[0], var[0]), pi[1] * normal_pdf(grid, mu[1], var[1])
        ax.plot(grid, c1, color=ORANGE, lw=2); ax.plot(grid, c2, color=BLUE, lw=2)
        ax.plot(grid, c1 + c2, "k--", lw=1.2)
        ax.set_title(("Initial state" if i == 0 else f"Iteration {i}" if s is not None else f"Converged (it. {i})")
                     + f"\nlog-lik = {ll:,.1f}")
        ax.set_xlabel("height (inches)")
    axes[0].set_ylabel("density")
    plt.tight_layout(); plt.show()


def plot_loglik(hist):
    """Log-likelihood climbs then flattens: that flattening is the stopping rule."""
    ll = [h[3] for h in hist]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ll, color=PURPLE, lw=2.5)
    ax.set(title="EM never decreases the log-likelihood", xlabel="iteration", ylabel="log-likelihood")
    ax.set_xscale("symlog", linthresh=3)
    ax.annotate("big jump in the first steps", xy=(1, ll[1]), xytext=(2.5, ll[1] - 25),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    plt.tight_layout(); plt.show()


def plot_soft_vs_hard(x, means, variances, proportions, cut, e_step):
    """EM's soft probability curve vs the global-mean hard cut."""
    grid = np.linspace(x.min(), x.max(), 400)
    r = e_step(grid, means, variances, proportions)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(grid, r[:, 0], color=ORANGE, lw=2.5, label="EM: P(component 1 | height)")
    ax.plot(grid, r[:, 1], color=BLUE, lw=2.5, label="EM: P(component 2 | height)")
    ax.step(grid, (grid > cut).astype(float), color=GREEN, lw=2, ls="--", where="post", label="global-mean split (0 or 1 only)")
    ax.axvline(cut, color=GREEN, ls=":", lw=1)
    ax.fill_between(grid, 0, 1, where=(r[:, 0] > .1) & (r[:, 0] < .9), color=GREY, alpha=.15, label="EM says: uncertain")
    ax.set(title="Soft assignments keep the uncertainty; a hard cut throws it away",
           xlabel="height (inches)", ylabel="probability"); ax.legend(fontsize=8, loc="center left")
    plt.tight_layout(); plt.show()


def plot_confusion(cm, row_names, col_names, accuracy):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(cm, cmap="Blues"); ax.grid(False)
    ax.set_xticks(range(2), [f"pred. {n}" for n in col_names]); ax.set_yticks(range(2), [f"actual {n}" for n in row_names])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix (accuracy {accuracy:.1%})")
    plt.tight_layout(); plt.show()


def plot_test_height(x, means, variances, proportions, height, p_mother, p_child):
    """Mixture density with the test height marked and the two posteriors written on it."""
    grid = np.linspace(x.min() - 2, x.max() + 2, 400)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(x, bins=28, density=True, color=GREY, alpha=.3)
    for k, c in enumerate([ORANGE, BLUE]):
        ax.plot(grid, proportions[k] * normal_pdf(grid, means[k], variances[k]), color=c, lw=2)
    ax.axvline(height, color="k", ls="--")
    ax.text(height + .3, ax.get_ylim()[1] * .85, f"test height {height}\nP(Mother) = {p_mother:.1%}\nP(Child) = {p_child:.1%}",
            fontsize=10, bbox=dict(boxstyle="round", fc="white", ec=GREY))
    ax.set(title="Classifying a new height with the fitted mixture", xlabel="height (inches)", ylabel="density")
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- Part 2
def plot_posteriors(words, posteriors, prior=0.5, n_pos=4):
    """Bar chart of P(Positive | word): green = chosen positive words, red = chosen negative words."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(words, posteriors, color=[GREEN if i < n_pos else RED for i in range(len(words))], alpha=.85)
    ax.axhline(prior, color="k", ls="--", lw=1.2); ax.text(len(words) - .5, prior + .02, "prior = 0.5 (no information)", ha="right", fontsize=9)
    for b_, p in zip(bars, posteriors):
        ax.text(b_.get_x() + b_.get_width() / 2, p + .015, f"{p:.2f}", ha="center", fontweight="bold")
    ax.set(ylim=(0, 1), ylabel="P(Positive | word)", title="How much does each keyword shift our belief?")
    plt.tight_layout(); plt.show()


def bayes_card(word, prior, likelihood, marginal, posterior):
    """Prior x Likelihood / Marginal = Posterior, drawn as four boxes."""
    box = lambda t, v, c: (f'<div style="flex:1;border:2px solid {c};border-radius:10px;padding:10px;text-align:center">'
                           f'<div style="font-size:12px;color:{c}">{t}</div><div style="font-size:24px;font-weight:700">{v:.4f}</div></div>')
    op = lambda s: f'<div style="font-size:26px;align-self:center">{s}</div>'
    display(HTML(f'<div style="margin:8px 0"><b>Worked example for the word “{word}”</b>'
                 f'<div style="display:flex;gap:8px;margin-top:6px">{box("Prior P(Positive)", prior, GREY)}{op("×")}'
                 f'{box(f"Likelihood P({word} | Positive)", likelihood, BLUE)}{op("÷")}{box(f"Marginal P({word})", marginal, ORANGE)}'
                 f'{op("=")}{box("Posterior P(Positive | " + word + ")", posterior, RED if posterior < .5 else GREEN)}</div></div>'))


# --------------------------------------------------------------------------- Parts 3 & 4
def plot_predictions(y, snapshots):
    """True targets vs predictions at several iterations (shows the model improving)."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    n = len(y); w = 0.8 / (len(snapshots) + 1); idx = np.arange(n)
    ax.bar(idx, y, width=w, color="k", alpha=.8, label="true y")
    for j, (label, yh) in enumerate(snapshots):
        ax.bar(idx + (j + 1) * w, yh, width=w, label=label, alpha=.85)
    ax.set_xticks(idx + 0.4 - w / 2, [f"point {i + 1}" for i in idx])
    ax.set(title="Predictions move toward the targets", ylabel="value"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
