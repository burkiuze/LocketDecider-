"""Turning option scores into calibrated probabilities, and the metrics used to judge them.

Pure numpy so it can be tested without a model.
"""

from __future__ import annotations

import numpy as np


def softmax(scores, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(score_rows: list, answers, temperature: float) -> float:
    """Mean negative log-likelihood of the right answers. Rows may have different option counts."""
    total = 0.0
    for scores, a in zip(score_rows, answers):
        z = np.asarray(scores, dtype=np.float64) / temperature
        m = z.max()
        total -= z[a] - (m + np.log(np.exp(z - m).sum()))
    return total / max(len(answers), 1)


def fit_temperature(score_rows: list, answers, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature that minimises validation NLL (Guo et al., 2017). NLL is unimodal in log T."""
    a, b = np.log(lo), np.log(hi)
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(score_rows, answers, np.exp(c)), nll(score_rows, answers, np.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(score_rows, answers, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(score_rows, answers, np.exp(d))
    return float(np.exp((a + b) / 2))


def ece(confidences, correct, bins: int = 15) -> float:
    """Expected calibration error: how far stated confidence is from observed accuracy."""
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    err = 0.0
    for i in range(bins):
        in_bin = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if in_bin.any():
            err += in_bin.mean() * abs(hit[in_bin].mean() - conf[in_bin].mean())
    return float(err)


def macro_f1(pred_labels, gold_labels) -> float:
    """Macro F1 over the label names seen in gold (options are shuffled, so compare names, not keys)."""
    pred = list(pred_labels)
    gold = list(gold_labels)
    scores = []
    for lab in sorted(set(gold)):
        tp = sum(p == lab and g == lab for p, g in zip(pred, gold))
        fp = sum(p == lab and g != lab for p, g in zip(pred, gold))
        fn = sum(p != lab and g == lab for p, g in zip(pred, gold))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def selective(confidences, correct, threshold: float) -> dict:
    """Accuracy on the decisions confident enough to automate, and how many that is."""
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=bool)
    keep = conf >= threshold
    return {
        "threshold": threshold,
        "coverage": float(keep.mean()) if len(conf) else 0.0,
        "accuracy": float(hit[keep].mean()) if keep.any() else 0.0,
    }


def threshold_for_accuracy(confidences, correct, target: float) -> float | None:
    """Lowest confidence threshold whose automated decisions reach the target accuracy."""
    order = np.argsort(-np.asarray(confidences, dtype=np.float64))
    conf = np.asarray(confidences, dtype=np.float64)[order]
    hit = np.asarray(correct, dtype=np.float64)[order]
    acc = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    ok = np.nonzero(acc >= target)[0]
    return float(conf[ok[-1]]) if len(ok) else None
