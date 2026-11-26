"""Accumulated Accuracy Gap Control (AGC): 조기 정지로 인한 정확도 손실을 통계적으로 통제하는 정지 기준 보정.

원 논문: Ringel et al., "Early Time Classification with Accumulated Accuracy Gap Control", ICML 2024
  https://proceedings.mlr.press/v235/ringel24a.html

이 파일은 Transformer 담당 팀원이 논문의 조건부 Algorithm 1, 2를 독립 구현한 것을 그대로 가져온 것이다
(논문 공식 구현 https://github.com/liranringel/etc 과 60회 무작위 비교에서 threshold가 일치했다고 보고됨).
모든 모델에 같은 구현을 쓰기 위해 수정하지 않았다. 아래는 원 구현의 설명이다.

Independent implementation from the published algorithms. Binary loss is
full-correct AND early-wrong, not the signed difference in accuracies.
Stage 2 is an intersection-union test inside a fixed sequence; there is no
Bonferroni division over accumulated times. A family division can be applied
by the caller when certifying several independently trained backbones.

사용법
  calibrate(screen, holdout, alpha, delta, grid_step)
    screen / holdout: {"confidence": 정지 점수 [N, K] (클수록 멈춰도 안전), "predictions": [N, K], "labels": [N]}
    screen으로 시점별 threshold 후보를 찾고(Algorithm 1), 독립된 holdout에서 뒤 시점부터 검정한다(Algorithm 2).
    점수가 threshold 이상인 첫 시점에서 정지한다. threshold가 None이면 그 시점에서는 정지하지 않는다.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import binom


def stop_indices(confidence, thresholds):
    confidence = np.asarray(confidence)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if confidence.ndim != 2 or thresholds.shape != (confidence.shape[1],):
        raise ValueError("Expected confidence [N,K] and thresholds [K]")
    should_stop = confidence >= thresholds[None, :]
    should_stop[:, -1] = True
    return should_stop.argmax(axis=1)


def gap_losses(predictions, labels, stops):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    return ((predictions[:, -1] == labels)
            & (predictions[np.arange(len(labels)), stops] != labels))


def accumulated_diagnostics(confidence, predictions, labels, thresholds):
    stops = stop_indices(confidence, thresholds)
    losses = gap_losses(predictions, labels, stops)
    records = []
    for t in range(confidence.shape[1]):
        selected = stops <= t
        count = int(selected.sum())
        failures = int(losses[selected].sum())
        records.append({"observation_index": t, "count": count, "harmful_stops": failures,
                        "gap": failures / count if count else None})
    return records


def candidate_screening(confidence, predictions, labels, alpha, grid_step=0.001):
    """Algorithm 1: first grid value meeting the accumulated empirical gap."""
    if not 0 < alpha < 1 or not 0 < grid_step <= 1:
        raise ValueError("Invalid alpha or grid_step")
    k = confidence.shape[1]
    chosen = np.full(k, np.inf)
    # Integer construction avoids values slightly above one due to arange.
    grid = np.unique(np.r_[np.arange(math.ceil(1 / grid_step)) * grid_step, 1.])
    for t in range(k):
        trial = chosen.copy()
        for value in grid:
            trial[t] = value
            stops = stop_indices(confidence, trial)
            selected = stops <= t
            if not selected.any():
                break
            if gap_losses(predictions, labels, stops)[selected].mean() <= alpha:
                chosen[t] = value
                break
    return chosen


def fixed_sequence_test(confidence, predictions, labels, candidate, alpha, delta):
    """Algorithm 2: reveal candidate thresholds backwards; stop on first failure."""
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("Invalid alpha or delta")
    k = confidence.shape[1]
    chosen, audit = np.full(k, np.inf), []
    for t in reversed(range(k)):
        trial = chosen.copy()
        trial[t] = candidate[t]
        stops = stop_indices(confidence, trial)
        losses = gap_losses(predictions, labels, stops)
        accepted = True
        checks = []
        for end in range(t, k):
            selected = stops <= end
            n, failures = int(selected.sum()), int(losses[selected].sum())
            p_value = float(binom.cdf(failures, n, alpha)) if n else 1.0
            checks.append({"through_index": end, "n": n, "failures": failures, "p_value": p_value})
            if not n or p_value > delta:
                accepted = False
                break
        audit.append({"reveal_index": t, "accepted": accepted, "checks": checks})
        if not accepted:
            break
        chosen = trial
    return chosen, audit


def serialize_thresholds(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def deserialize_thresholds(values):
    return np.asarray([np.inf if value is None else value for value in values], dtype=np.float64)


def calibrate(screen, holdout, alpha, delta, grid_step):
    candidate = candidate_screening(screen["confidence"], screen["predictions"], screen["labels"],
                                    alpha, grid_step)
    chosen, audit = fixed_sequence_test(holdout["confidence"], holdout["predictions"], holdout["labels"],
                                        candidate, alpha, delta)
    return {"method": "conditional_AGC_algorithms_1_2", "alpha": alpha, "delta": delta,
            "grid_step": grid_step, "candidate_thresholds": serialize_thresholds(candidate),
            "thresholds": serialize_thresholds(chosen), "testing_audit": audit,
            "screening_n": len(screen["labels"]), "calibration_n": len(holdout["labels"]),
            "calibration_accumulated_gap": accumulated_diagnostics(
                holdout["confidence"], holdout["predictions"], holdout["labels"], chosen)}
