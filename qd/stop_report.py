"""한 모델의 정지 정책들을 저장된 관찰 시점별 출력(obs_test.npz)으로 비교한다.

  - 고정 threshold (0.9가 팀 공통 baseline 규칙)의 정확도-조기성 곡선
  - 정지 head가 있는 모델(ELECTS)이면 "정지 확률 ≥ 기준"의 곡선
  - 관찰 비율별 정확도, 평균 확신도, ECE
  - --reference 를 주면 기준 모델(baseline)의 고정 threshold 곡선과 같은 정확도에서 비교

실행 (VM의 레포 루트에서, qd.dump 뒤):
  python -m qd.stop_report --config configs/mamba_20cls_elects.yaml --reference mamba2_20cls
출력: ~/runs/<run_name>/stop_report.json, stop_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qd.common import load_config, run_dir
from qd.dump import Observations
from qd.policies import (earliness_at_accuracy, expected_calibration_error, first_stop,
                         fixed_threshold, stopping_metrics)

THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995]
STOP_PROBS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def metrics(stop: np.ndarray, obs: Observations) -> dict:
    return stopping_metrics(first_stop(stop), obs.logits.argmax(-1), obs.labels, obs.fractions, obs.n_obs)


def analyze(obs: Observations) -> dict:
    probs = obs.probs
    confidence, correct = probs.max(-1), probs.argmax(-1) == obs.labels[:, None]
    result = {
        "n": int(len(obs.labels)),
        "fractions": obs.fractions.tolist(),
        "accuracy_by_fraction": correct.mean(0).tolist(),
        "confidence_by_fraction": confidence.mean(0).tolist(),
        "ece_by_fraction": [expected_calibration_error(confidence[:, k], correct[:, k])
                            for k in range(confidence.shape[1])],
        "fixed_curve": [{"threshold": t, **metrics(fixed_threshold(confidence, t), obs)} for t in THRESHOLDS],
        "stop_head_curve": [],
    }
    if len(obs.stop_logit):  # 정지 head가 있는 모델
        stop_prob = 1.0 / (1.0 + np.exp(-obs.stop_logit))
        result["stop_head_curve"] = [{"stop_prob": p, **metrics(stop_prob >= p, obs)} for p in STOP_PROBS]
        result["stop_prob_by_fraction"] = stop_prob.mean(0).tolist()
    return result


def to_markdown(name: str, r: dict, ref_name: str | None, ref: dict | None) -> str:
    def row(label: str, m: dict) -> str:
        return (f"| {label} | {m['stopping_accuracy']:.2%} | {m['mean_earliness']:.1%} | "
                f"{m['harmful_gap']:.2%} | {m['max_accumulated_harmful_gap']:.2%} | "
                f"{m['points_recompute']:.1f} | {m['points_streaming']:.1f} | {m['readouts']:.2f} |")

    header = ["| 분류 모델 / 정지 정책 | Stopping Accuracy | Mean Earliness | Harmful Gap | 최대 누적 조건부 Harmful Gap "
              "| 계산한 점 (재추론) | 계산한 점 (상태 유지) | 예측 횟수 |",
              "|---|---|---|---|---|---|---|---|"]
    fixed09 = next(p for p in r["fixed_curve"] if p["threshold"] == 0.9)
    lines = [f"# {name}: 정지 정책 비교 (test {r['n']:,}개)", "", "## 주요 조건", *header]
    if ref:
        lines.append(row(f"{ref_name} / 0.9 고정", next(p for p in ref["fixed_curve"] if p["threshold"] == 0.9)))
    lines.append(row(f"{name} / 0.9 고정", fixed09))
    if r["stop_head_curve"]:
        lines.append(row(f"{name} / 정지 확률 ≥ 0.5",
                         next(p for p in r["stop_head_curve"] if p["stop_prob"] == 0.5)))

    lines += ["", "## 같은 정확도에서의 Mean Earliness (곡선을 선형 보간)",
              "| 목표 Stopping Accuracy | " + (f"{ref_name} 고정 threshold | " if ref else "")
              + f"{name} 고정 threshold" + (f" | {name} 정지 head" if r["stop_head_curve"] else "") + " |",
              "|---|---|" + ("---|" if ref else "") + ("---|" if r["stop_head_curve"] else "")]
    for target in (0.90, 0.91, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945):
        cells = []
        if ref:
            cells.append(earliness_at_accuracy(ref["fixed_curve"], target))
        cells.append(earliness_at_accuracy(r["fixed_curve"], target))
        if r["stop_head_curve"]:
            cells.append(earliness_at_accuracy(r["stop_head_curve"], target))
        lines.append(f"| {target:.1%} | " + " | ".join("-" if c is None else f"{c:.1%}" for c in cells) + " |")

    lines += ["", "## 관찰 비율별 정확도와 확신도",
              "| 관찰 비율 | " + " | ".join(f"{f:.0%}" for f in r["fractions"]) + " |",
              "|---|" + "---|" * len(r["fractions"])]
    if ref:
        lines.append(f"| 정확도 ({ref_name}) | " + " | ".join(f"{a:.1%}" for a in ref["accuracy_by_fraction"]) + " |")
    lines.append(f"| 정확도 ({name}) | " + " | ".join(f"{a:.1%}" for a in r["accuracy_by_fraction"]) + " |")
    lines.append("| 평균 확신도 | " + " | ".join(f"{a:.1%}" for a in r["confidence_by_fraction"]) + " |")
    if ref:
        lines.append(f"| ECE ({ref_name}) | " + " | ".join(f"{a:.2%}" for a in ref["ece_by_fraction"]) + " |")
    lines.append(f"| ECE ({name}) | " + " | ".join(f"{a:.2%}" for a in r["ece_by_fraction"]) + " |")
    if r["stop_head_curve"]:
        lines.append("| 평균 정지 확률 | " + " | ".join(f"{a:.1%}" for a in r["stop_prob_by_fraction"]) + " |")

    lines += ["", "## 곡선", "| 고정 threshold | Stopping Accuracy | Mean Earliness | Harmful Gap |", "|---|---|---|---|"]
    for p in r["fixed_curve"]:
        lines.append(f"| {p['threshold']} | {p['stopping_accuracy']:.2%} | {p['mean_earliness']:.1%} | {p['harmful_gap']:.2%} |")
    if r["stop_head_curve"]:
        lines += ["", "| 정지 확률 기준 | Stopping Accuracy | Mean Earliness | Harmful Gap |", "|---|---|---|---|"]
        for p in r["stop_head_curve"]:
            lines.append(f"| {p['stop_prob']} | {p['stopping_accuracy']:.2%} | {p['mean_earliness']:.1%} | {p['harmful_gap']:.2%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 관찰 시점별 출력으로 정지 정책 비교")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--reference", default=None, help="비교 기준 run 이름 (예: mamba2_20cls)")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    out = run_dir(cfg)
    result = analyze(Observations.load(out / f"obs_{args.split}.npz"))
    ref = None
    if args.reference:
        ref = analyze(Observations.load(out.parent / args.reference / f"obs_{args.split}.npz"))
        result["reference"] = {"run_name": args.reference, **ref}
    with open(out / "stop_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    markdown = to_markdown(cfg["run_name"], result, args.reference, ref)
    Path(out / "stop_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
