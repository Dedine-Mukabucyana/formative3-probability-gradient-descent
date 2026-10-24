# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent

**Group 11.** 

## Run it
```bash
pip install -r requirements.txt

jupyter notebook Formative3_Final.ipynb
```

## Part 1 — EM on a two-Gaussian mixture (mothers vs children, Galton)
From scratch with NumPy: E-step (responsibilities), M-step (μ, σ², π), stop when the log-likelihood change is < 1e-6. One mother per family (205) + 934 children; labels used only for grading.
Tracking table (iterations 0, 1, 2) is in the notebook. Result: μ ≈ 64.30 / 69.79, σ² ≈ 5.79 / 5.47, π ≈ 0.64 / 0.36, log-likelihood −3035.7 (the global-mean split's model scores −3061.7). Heights barely separate mothers from children (accuracy 51 %), which we discuss openly. Test height: `classify(68)` → P(Mother) 0.419, P(Child) 0.581.

## Part 2 — Bayes' theorem on IMDb (basic Python only)
P(Positive | keyword) only. Keywords: excellent, wonderful, amazing, great / terrible, awful, boring, worst. Prior, likelihood, marginal and posterior for each are in the notebook.

## Part 3 — Gradient descent by hand
`docs/Part3_Manual_Calculations.pdf` (typed answer sheet) and `docs/Part3_Handwriting_Guide.pdf` (what to write, page by page) — the handwritten submission is scanned from the guide: chain-rule derivation and one update per member (3). Assumptions: b is a 2-vector, learning rate 0.01.

## Part 4 — Gradient descent with SciPy
`src/gradient_descent.py`: `partial_derivative` differentiates any equation with `scipy.differentiate.derivative`, checked against the analytic gradient. Plots of m, b and the MSE per iteration are in `figures/`.
