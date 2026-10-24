# Formative 3: Probability Distributions, Bayesian Probability and Gradient Descent

Mathematics for Machine Learning, African Leadership University. Group 11.

| Member | Leads | Code |
|---|---|---|
| Milliam Mukamukiza | Part 1: EM algorithm | `src/em.py`, `src/data.py` |
| Dedine Mukabucyana | Part 2: Bayesian probability | `src/bayes.py` |
| Cedric Bigwi Hindura | Parts 3 and 4: gradient descent | `src/gradient_descent.py`, `src/plotting.py` |

All members present every part. See `docs/CONTRIBUTIONS.md`.

## Run it

```bash
pip install -r requirements.txt
# download "IMDB Dataset.csv" from the assignment link into data/ (see data/README.md)
jupyter notebook Formative3_Presentation.ipynb      # Restart and Run All
python -m pytest -q                                  # 11 tests
```

`Formative3_Presentation.ipynb` is the single notebook used for the presentation and the contribution check.

## Repository layout

```
Formative3_Presentation.ipynb   the one notebook (all four parts, tables, plots, Q&A prep)
src/
  data.py                       load the Galton heights (mother or father vs children)
  em.py                         EM for a two-Gaussian mixture, tracking table, evaluation
  bayes.py                      Bayes' theorem on IMDb reviews, standard library only
  gradient_descent.py           matrix-form gradient descent, SciPy derivative function
  plotting.py                   shared plotting helpers
tests/                          unit tests for EM, Bayes and gradient descent
data/                           GaltonFamilies.csv and download notes for the IMDb file
figures/                        plots produced by the notebook
docs/                           handwritten Part 3 PDF and contribution proof
```

## Part 1: EM on a two-Gaussian mixture (mothers vs children, Galton)

EM from scratch with NumPy: E-step (responsibilities), M-step (means, variances, mixing proportions), stop when the log-likelihood changes by less than 1e-6. One mother per family (205) plus all 934 children. Labels are used only for evaluation.

- Converges after 386 iterations: means 64.30 and 69.79, variances 5.79 and 5.47, proportions 0.64 and 0.36, log-likelihood -3035.75 (the global-mean split scores -3061.68).
- EM mainly separates shorter from taller people (mothers and daughters low, sons high), so accuracy against mother/child labels is only 51% (the always-guess-child baseline is 82%). The notebook shows this openly.
- The notebook classifies any test height: `TEST_HEIGHT = 68.0` gives P(mother) 0.4192 and P(child) 0.5808.

## Part 2: Bayes' theorem on IMDb (basic Python only)

P(Positive | keyword) only. Positive keywords: excellent, wonderful, amazing, great. Negative keywords: terrible, awful, boring, worst. The notebook prints prior, likelihood, marginal and posterior for each.

## Part 3: Gradient descent by hand

Model y_hat = Xm + b with the given X, y, m0 = [-1, 2], b0 = [1, 1], MSE cost, learning rate 0.01 (stated assumption), b as a 2-vector. Three updates, one per member. The notebook derives both gradients with the chain rule and prints an answer key for the handwritten sheet in `docs/Part3_Manual_Calculations.pdf`.

| Update | y_hat | J | dJ/dm | new m | new b |
|---|---|---|---|---|---|
| 1 | [6, 17] | 61 | [45, 113] | [-1.45, 0.87] | [0.99, 0.89] |
| 2 | [2.15, 3.79] | 6.5033 | [-11.69, -30.65] | [-1.3331, 1.1765] | [1.0185, 0.9121] |
| 3 | [3.2149, 7.3447] | 2.4974 | [3.5937, 8.0917] | [-1.3690, 1.0956] | [1.0364, 0.8987] |

## Part 4: Gradient descent with SciPy

`gd.partial_derivative` differentiates any equation of a parameter vector with `scipy.differentiate.derivative` (SciPy 1.15 or newer) and is checked against the analytic gradient. The notebook shows every step of the update loop and plots m, b and the MSE per iteration.
