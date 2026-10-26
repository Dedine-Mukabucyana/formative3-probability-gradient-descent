import numpy as np

from src import em
from src.data import load_galton


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(60, 2, 400), rng.normal(72, 3, 600)])


def test_responsibilities_sum_to_one():
    x = _synthetic()
    r = em.e_step(x, em.initialize(x))
    assert np.allclose(r.sum(axis=1), 1.0)


def test_log_likelihood_never_decreases():
    history = em.fit(_synthetic())
    lls = [ll for _, ll in history]
    assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:]))


def test_recovers_known_mixture():
    p, _ = em.fit(_synthetic())[-1]
    assert np.allclose(sorted(p.means), [60, 72], atol=0.6)
    assert np.allclose(sorted(p.proportions), [0.4, 0.6], atol=0.05)


def test_galton_reference_values():
    x, labels, _ = load_galton("data/GaltonFamilies.csv")
    assert len(x) == 1139 and labels.sum() == 934
    history = em.fit(x)
    assert abs(history[0][1] - (-3115.4014)) < 1e-3
    assert abs(history[-1][1] - (-3035.7484)) < 1e-3


def test_global_split_scores_lower_than_em():
    x, _, _ = load_galton("data/GaltonFamilies.csv")
    _, split = em.global_mean_split(x)
    assert em.log_likelihood(x, split) < em.fit(x)[-1][1]
