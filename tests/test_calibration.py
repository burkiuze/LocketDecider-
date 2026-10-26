import numpy as np

from locket_decider.calibration import ece, fit_temperature, macro_f1, selective, softmax, threshold_for_accuracy
from locket_decider.scoring import common_prefix_len


def synthetic(n=4000, k=5, true_t=1.0, scale=3.0, seed=0):
    """Scores whose softmax at `true_t` is the real answer distribution, then inflated by `scale`."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, (n, k))
    answers = [rng.choice(k, p=softmax(row, true_t)) for row in logits]
    return [row * scale for row in logits], answers


def test_softmax_is_stable_and_normalised():
    p = softmax([1000.0, 1000.0, -1000.0])
    assert np.allclose(p, [0.5, 0.5, 0.0])


def test_fit_temperature_undoes_overconfidence():
    scores, answers = synthetic(scale=3.0)
    t = fit_temperature(scores, answers)
    assert abs(t - 3.0) < 0.3


def test_calibration_lowers_ece():
    scores, answers = synthetic(scale=3.0, seed=1)

    def err(t):
        probs = [softmax(s, t) for s in scores]
        return ece([p.max() for p in probs], [p.argmax() == a for p, a in zip(probs, answers)])

    assert err(fit_temperature(scores, answers)) < err(1.0) / 2


def test_ece_of_perfect_and_wrong_predictions():
    assert ece([1.0, 1.0], [True, True]) == 0.0
    assert abs(ece([0.9, 0.9], [False, False]) - 0.9) < 1e-9


def test_macro_f1():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert abs(macro_f1(["a", "a"], ["a", "b"]) - (2 / 3) / 2) < 1e-9


def test_threshold_reaches_target_accuracy():
    conf = [0.99, 0.95, 0.9, 0.8, 0.6, 0.5]
    hit = [True, True, True, False, True, False]
    t = threshold_for_accuracy(conf, hit, 0.75)
    sel = selective(conf, hit, t)
    assert sel["accuracy"] >= 0.75 and sel["coverage"] == 5 / 6
    assert threshold_for_accuracy(conf, [False] * 6, 0.5) is None


def test_common_prefix_len():
    assert common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert common_prefix_len([[5, 6], [7]]) == 0
