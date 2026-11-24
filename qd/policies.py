"""정지 정책을 저장된 관찰 시점별 출력(qd/dump.py)으로 평가하는 도구.

정지 정책은 스케치마다 "각 관찰 시점에서 멈출지"의 [N, K] bool 행렬로 표현한다.
실제 정지 시점은 처음 True가 나오는 시점이고, 마지막 시점에서는 무조건 멈춘다.
"""
from __future__ import annotations

import numpy as np


def first_stop(stop: np.ndarray) -> np.ndarray:
    """[N, K] bool → 스케치별 정지 시점 index [N]."""
    stop = stop.copy()
    stop[:, -1] = True  # 100%에서는 강제 정지
    return stop.argmax(1)


def fixed_threshold(confidence: np.ndarray, threshold: float) -> np.ndarray:
    """팀 공통 baseline 규칙: 최대 클래스 확률이 threshold 이상이면 정지."""
    return confidence >= threshold


def stopping_metrics(stop_index: np.ndarray, preds: np.ndarray, labels: np.ndarray,
                     fractions: np.ndarray) -> dict:
    """정지 시점 [N], 시점별 예측 [N, K], 정답 [N] → 조기 예측 지표.

    Harmful Gap: 완성된 스케치(100%)에서는 맞혔을 텐데 일찍 멈춰서 틀린 스케치의 비율.
    누적 조건부 Harmful Gap: 시점 t까지 멈춘 스케치들만 놓고 본 Harmful Gap (Ringel et al., 2024).
    """
    rows = np.arange(len(labels))
    correct = preds[rows, stop_index] == labels
    harmful = (preds[:, -1] == labels) & ~correct
    accumulated = []
    for k in range(len(fractions)):
        stopped = stop_index <= k
        accumulated.append(float(harmful[stopped].mean()) if stopped.any() else None)
    return {
        "stopping_accuracy": float(correct.mean()),
        "mean_earliness": float(fractions[stop_index].mean()),
        "harmful_gap": float(harmful.mean()),
        "max_accumulated_harmful_gap": max(a for a in accumulated if a is not None),
        "accumulated_harmful_gap": accumulated,
        "stop_share": [float((stop_index == k).mean()) for k in range(len(fractions))],
    }


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE: 확신도 구간별 |평균 확신도 − 실제 정확도|를 구간 크기로 가중 평균한 값."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        here = bins == b
        if here.any():
            ece += here.mean() * abs(confidence[here].mean() - correct[here].mean())
    return float(ece)


def earliness_at_accuracy(curve: list[dict], target_accuracy: float) -> float | None:
    """(정확도, 조기성) 곡선에서 목표 정확도를 내는 지점의 Mean Earliness를 선형 보간한다."""
    points = sorted((p["stopping_accuracy"], p["mean_earliness"]) for p in curve)
    acc = np.array([p[0] for p in points])
    earl = np.array([p[1] for p in points])
    if not (acc[0] <= target_accuracy <= acc[-1]):
        return None  # 곡선이 그 정확도에 도달하지 못함
    return float(np.interp(target_accuracy, acc, earl))
