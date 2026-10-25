import pytest

from src import bayes

REVIEWS = [({"great", "film"}, "positive"), ({"great", "plot"}, "negative"),
           ({"awful", "film"}, "negative"), ({"fun", "film"}, "positive")]


def test_posterior_matches_direct_ratio():
    for word in ("great", "awful", "film"):
        total, pos, has, has_pos = bayes.count_keywords(REVIEWS, [word])
        _, _, _, posterior = bayes.bayes_theorem(total, pos, has[word], has_pos[word])
        assert posterior == pytest.approx(has_pos[word] / has[word])


def test_known_values():
    rows = bayes.bayes_table(REVIEWS, ["great"])
    _, prior, likelihood, marginal, posterior = rows[0]
    assert (prior, likelihood, marginal, posterior) == pytest.approx((0.5, 0.5, 0.5, 0.5))


def test_missing_keyword_raises():
    with pytest.raises(ValueError):
        bayes.bayes_table(REVIEWS, ["absent"])
