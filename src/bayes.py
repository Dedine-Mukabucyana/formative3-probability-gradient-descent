"""Part 2 - Bayes' theorem on IMDb reviews, basic Python only (standard library)."""
from __future__ import annotations

import csv
import os
import re

WORD = re.compile(r"[a-z']+")


def load_reviews(path: str):
    """Read the IMDb csv into a list of (set_of_words, label). Label is 'positive' or 'negative'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. See data/README.md for the download link.")
    reviews = []
    with open(path, encoding="utf-8", newline="") as f:
        rows = csv.reader(f)
        next(rows)                                   # skip header
        for text, label in rows:
            reviews.append((set(WORD.findall(text.lower())), label))
    return reviews


def count_keywords(reviews, keywords):
    """One pass over the data: total reviews, positive reviews, and per-word counts."""
    total = len(reviews)
    positive = 0
    has = {w: 0 for w in keywords}
    has_positive = {w: 0 for w in keywords}
    for words, label in reviews:
        is_pos = label == "positive"
        positive += is_pos
        for w in keywords:
            if w in words:
                has[w] += 1
                has_positive[w] += is_pos
    return total, positive, has, has_positive


def bayes_theorem(total, positive, has, has_positive):
    """Return prior, likelihood, marginal, posterior for one keyword."""
    if has == 0 or positive == 0:
        raise ValueError("keyword never appears (or no positive reviews): posterior undefined")
    prior = positive / total                     # P(Positive)
    likelihood = has_positive / positive         # P(keyword | Positive)
    marginal = has / total                       # P(keyword)
    posterior = likelihood * prior / marginal    # P(Positive | keyword)
    return prior, likelihood, marginal, posterior


def bayes_table(reviews, keywords):
    """Rows of (keyword, prior, likelihood, marginal, posterior), one per keyword."""
    total, positive, has, has_positive = count_keywords(reviews, keywords)
    return [(w, *bayes_theorem(total, positive, has[w], has_positive[w])) for w in keywords]


def format_table(rows) -> str:
    """Plain-text table for printing."""
    lines = [f"{'keyword':<11}{'prior':>8}{'likelihood':>12}{'marginal':>10}{'posterior':>11}"]
    for w, prior, lik, marg, post in rows:
        lines.append(f"{w:<11}{prior:>8.4f}{lik:>12.4f}{marg:>10.4f}{post:>11.4f}")
    return "\n".join(lines)
