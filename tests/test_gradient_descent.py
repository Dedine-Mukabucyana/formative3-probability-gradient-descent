import numpy as np

from src import gradient_descent as gd

X = np.array([[1., 3.], [4., 10.]])
y = np.array([5., 6.])
M0, B0 = np.array([-1., 2.]), np.array([1., 1.])


def test_scipy_matches_analytic_gradient():
    g = gd.scipy_gradient(lambda t: gd.cost_of_theta(t, X, y), gd.pack(M0, B0))
    dm, db = gd.analytic_gradients(X, M0, B0, y)
    assert np.allclose(g, np.concatenate([dm, db]), atol=1e-6)


def test_hand_calculation_values():
    _, _, history = gd.run_gradient_descent(X, y, M0, B0, 0.01, 3, use_scipy=False)
    assert np.allclose(history[0]["dm"], [45, 113]) and np.allclose(history[0]["db"], [1, 11])
    assert np.isclose(history[0]["cost"], 61.0)
    assert np.allclose(history[1]["m"], [-1.45, 0.87]) and np.allclose(history[1]["b"], [0.99, 0.89])
    assert np.allclose([h["cost"] for h in history], [61.0, 6.5033, 2.4974, 2.1645], atol=1e-4)
    assert np.allclose(history[3]["m"], [-1.369037, 1.095583], atol=1e-5)


def test_cost_decreases():
    _, _, history = gd.run_gradient_descent(X, y, M0, B0, 0.01, 50)
    costs = [h["cost"] for h in history]
    assert all(b < a for a, b in zip(costs, costs[1:]))
