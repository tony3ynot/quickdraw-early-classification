"""Deep Optimal Stopping으로 정지 정책을 학습한다 (분류 모델은 고정).

원 논문: Becker, Cheridito, Jentzen, "Deep Optimal Stopping", JMLR 2019
  https://www.jmlr.org/papers/v20/18-232.html  (옵션 행사 시점 결정 문제를 위한 방법)

아이디어
  관찰 시점 n = 0..K-1 마다 "멈춤(1)/계속(0)"을 내는 작은 신경망 f_n을 둔다.
  마지막 시점은 무조건 멈춘다(f_{K-1} = 1). 그 앞 시점들은 뒤에서부터 차례로 학습한다:
    시점 n의 신경망은  r_n · F + G_{n+1} · (1 − F)  의 평균을 최대화한다.
      r_n     : 지금 멈출 때의 보상
      G_{n+1} : 계속 가서, 이미 학습된 뒤 시점의 규칙을 따랐을 때 실제로 얻는 보상
      F       : 신경망의 출력(0~1). 추론 때는 F ≥ 0.5 이면 멈춘다.
  즉 "지금의 확신도"만 보는 고정 threshold와 달리, 기다렸을 때 얻을 것과 비교해서 결정한다.

이 프로젝트에 맞춘 부분 (원 논문과 다른 점)
  - 보상 r_n = 1[시점 n의 예측이 정답] − cost × (시점 n의 관찰 비율).
    원 논문의 보상은 상태에서 바로 계산되는 값인데, 여기서는 정답 여부라서 학습 때만 알 수 있다.
    신경망은 상태만 보고 결정하므로 추론 때 정답을 쓰지 않는다.
  - 상태 = 분류 모델이 그 시점에 낸 출력(확률, 또는 feature + 확률). Mamba의 hidden state가
    과거를 요약하므로 과거 전체를 따로 넣지 않는다.
  - 원 논문은 시뮬레이션으로 경로를 무한히 만들지만 여기서는 데이터가 유한하다.
    그래서 정책 학습용(A)과 설정 선택용(B) 데이터를 나누고, test는 최종 보고에만 쓴다.

실행 (VM의 레포 루트에서, qd.dump로 obs_*.npz를 만든 뒤):
  python -m qd.optimal_stopping --config configs/mamba_20cls.yaml
출력: ~/runs/<run_name>/dos_results.json, dos.md, dos_policy.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qd.common import load_config, run_dir
from qd.dump import Observations
from qd.policies import (earliness_at_accuracy, expected_calibration_error, first_stop,
                         fixed_threshold, stopping_metrics)

COSTS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995]


def build_states(obs: Observations, kind: str, device) -> torch.Tensor:
    """정지 신경망의 입력 [N, K, d]. 클래스 확률과, 크기순으로 정렬한 확률을 함께 준다."""
    probs = torch.from_numpy(obs.probs).float()
    parts = [probs, probs.sort(dim=-1, descending=True).values]
    if kind == "feature+probs":
        parts.insert(0, torch.from_numpy(obs.feature.astype(np.float32)))
    return torch.cat(parts, dim=-1).to(device)


def build_rewards(obs: Observations, cost: float, device) -> torch.Tensor:
    """보상 [N, K] = 정답 여부 − cost × 관찰 비율."""
    correct = obs.logits.argmax(-1) == obs.labels[:, None]
    return torch.from_numpy(correct - cost * obs.fractions[None, :]).float().to(device)


class StopNet(nn.Module):
    """한 관찰 시점의 멈춤/계속 결정 신경망. 출력이 0 이상이면 멈춘다."""

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def train_policy(states: torch.Tensor, rewards: torch.Tensor, hidden: int = 128, steps: int = 600,
                 batch_size: int = 2048, lr: float = 1e-3, weight_decay: float = 1e-4,
                 seed: int = 0) -> nn.ModuleList:
    """마지막 시점부터 거꾸로 K-1개의 정지 신경망을 학습한다."""
    n, n_times, d_in = states.shape
    rng = torch.Generator(device=states.device).manual_seed(seed)
    torch.manual_seed(seed)
    nets = nn.ModuleList(StopNet(d_in, hidden) for _ in range(n_times - 1)).to(states.device)
    future = rewards[:, -1].clone()  # G: 뒤 시점 규칙을 따랐을 때 얻는 보상. 마지막 시점은 강제 정지.
    for t in range(n_times - 2, -1, -1):
        net = nets[t]
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(steps):
            idx = torch.randint(0, n, (batch_size,), device=states.device, generator=rng)
            stop_prob = torch.sigmoid(net(states[idx, t]))
            expected_reward = rewards[idx, t] * stop_prob + future[idx] * (1 - stop_prob)
            optimizer.zero_grad(set_to_none=True)
            (-expected_reward.mean()).backward()
            optimizer.step()
        with torch.no_grad():  # 이 시점의 결정을 확정하고, 그에 따른 보상으로 G를 갱신
            stop_now = net(states[:, t]) >= 0
            future = torch.where(stop_now, rewards[:, t], future)
    return nets


@torch.no_grad()
def apply_policy(nets: nn.ModuleList, states: torch.Tensor) -> np.ndarray:
    """학습된 정책 → 정지 여부 행렬 [N, K] (마지막 시점은 항상 True)."""
    stop = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
    for t, net in enumerate(nets):
        stop[:, t] = net(states[:, t]) >= 0
    return stop.cpu().numpy()


def evaluate_stops(stop: np.ndarray, obs: Observations) -> dict:
    return stopping_metrics(first_stop(stop), obs.logits.argmax(-1), obs.labels, obs.fractions)


def threshold_curve(obs: Observations) -> list[dict]:
    confidence = obs.probs.max(-1)
    return [{"threshold": thr, **evaluate_stops(fixed_threshold(confidence, thr), obs)}
            for thr in THRESHOLDS]


def closest(curve: list[dict], target_accuracy: float) -> dict:
    return min(curve, key=lambda p: abs(p["stopping_accuracy"] - target_accuracy))


def to_markdown(r: dict) -> str:
    def row(name: str, m: dict) -> str:
        return (f"| {name} | {m['stopping_accuracy']:.2%} | {m['mean_earliness']:.1%} | "
                f"{m['harmful_gap']:.2%} | {m['max_accumulated_harmful_gap']:.2%} |")

    best = r["selected"]
    lines = [
        f"# {r['run_name']}: Deep Optimal Stopping 결과",
        "",
        "분류 모델(baseline Mamba)은 그대로 두고 정지 정책만 바꿨다. "
        "원 논문: Becker, Cheridito, Jentzen, *Deep Optimal Stopping*, JMLR 2019.",
        "",
        "## 조건",
        f"- 정책 학습 데이터: {best['source']} ({r['n_policy_train'][best['source']]:,}개), "
        f"설정 선택: valid 중 따로 뗀 {r['n_select']:,}개, 최종 보고: test {r['n_test']:,}개",
        f"- 정지 신경망 입력: {best['kind']}, 관찰 시점마다 MLP 1개 (은닉 {r['hidden']} × 2층), "
        f"시점당 {r['steps']} step 학습",
        "- 보상 = 정답이면 1 − cost × 관찰 비율. cost를 바꿔 정확도와 조기성의 교환 곡선을 만든다.",
        f"- 설정(입력 종류, 학습 데이터)은 선택용 데이터에서 '0.9 고정과 같은 정확도일 때의 Mean Earliness'가 "
        "가장 낮은 것으로 골랐다. test는 선택에 쓰지 않았다.",
        "",
        "## test 결과",
        "| 정지 정책 | Stopping Accuracy | Mean Earliness | Harmful Gap | 최대 누적 조건부 Harmful Gap |",
        "|---|---|---|---|---|",
        row("0.9 고정 (baseline)", r["test"]["fixed_0.9"]),
        row(f"Deep Optimal Stopping (cost {best['cost_matched']})", r["test"]["dos_matched"]),
        row(f"고정 threshold {r['test']['tuned_fixed']['threshold']} (선택용 데이터에서 DOS와 정확도를 맞춤)",
            r["test"]["tuned_fixed"]),
        "",
        "같은 정확도에서의 Mean Earliness (test 곡선을 선형 보간, 탐색적 비교):",
        "",
        "| 목표 Stopping Accuracy | 고정 threshold | Deep Optimal Stopping |",
        "|---|---|---|",
    ]
    for m in r["test"]["same_accuracy"]:
        fixed = "-" if m["fixed"] is None else f"{m['fixed']:.1%}"
        dos = "-" if m["dos"] is None else f"{m['dos']:.1%}"
        lines.append(f"| {m['accuracy']:.1%} | {fixed} | {dos} |")
    lines += [
        "",
        "## 교환 곡선 (test)",
        "| cost | Stopping Accuracy | Mean Earliness | Harmful Gap |",
        "|---|---|---|---|",
    ]
    for p in r["test"]["dos_curve"]:
        lines.append(f"| {p['cost']} | {p['stopping_accuracy']:.2%} | {p['mean_earliness']:.1%} | {p['harmful_gap']:.2%} |")
    lines += ["", "| 고정 threshold | Stopping Accuracy | Mean Earliness | Harmful Gap |", "|---|---|---|---|"]
    for p in r["test"]["fixed_curve"]:
        lines.append(f"| {p['threshold']} | {p['stopping_accuracy']:.2%} | {p['mean_earliness']:.1%} | {p['harmful_gap']:.2%} |")
    lines += [
        "",
        "## 설정별 비교 (선택용 데이터)",
        "0.9 고정과 같은 정확도일 때의 Mean Earliness. 낮을수록 좋다. "
        f"0.9 고정 자체는 {r['select_fixed_0.9']['mean_earliness']:.1%}.",
        "",
        "| 학습 데이터 | 입력 | Mean Earliness |",
        "|---|---|---|",
    ]
    for c in r["configs"]:
        value = "-" if c["select_earliness_at_baseline_accuracy"] is None else f"{c['select_earliness_at_baseline_accuracy']:.1%}"
        lines.append(f"| {c['source']} | {c['kind']} | {value} |")
    lines += [
        "",
        "## 참고: baseline의 확신도 보정 상태 (test, ECE 15구간)",
        "| 관찰 비율 | " + " | ".join(f"{f:.0%}" for f in r["fractions"]) + " |",
        "|---|" + "---|" * len(r["fractions"]),
        "| 정확도 | " + " | ".join(f"{a:.1%}" for a in r["test"]["accuracy_by_fraction"]) + " |",
        "| 평균 확신도 | " + " | ".join(f"{a:.1%}" for a in r["test"]["confidence_by_fraction"]) + " |",
        "| ECE | " + " | ".join(f"{a:.2%}" for a in r["test"]["ece_by_fraction"]) + " |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Optimal Stopping 정지 정책 학습")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--select-size", type=int, default=10000, help="valid 중 설정 선택용으로 떼는 개수")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    out = run_dir(cfg)
    device = torch.device("cuda")

    valid = Observations.load(out / "obs_valid.npz")
    test = Observations.load(out / "obs_test.npz")
    # valid는 클래스 순서로 저장돼 있으므로 섞어서 정책 학습용(A)과 설정 선택용(B)으로 나눈다.
    order = np.random.default_rng(cfg["seed"]).permutation(len(valid.labels))
    select = valid.subset(np.sort(order[: args.select_size]))
    sources = {"valid": valid.subset(np.sort(order[args.select_size:]))}
    if (out / "obs_train.npz").exists():
        sources["train"] = Observations.load(out / "obs_train.npz")

    select_fixed = threshold_curve(select)
    test_fixed = threshold_curve(test)
    baseline_select = next(p for p in select_fixed if p["threshold"] == 0.9)
    baseline_test = next(p for p in test_fixed if p["threshold"] == 0.9)

    # ---- 설정(학습 데이터 × 입력 종류)마다 cost를 바꿔 가며 정책 학습 ----
    configs, trained = [], {}
    for source, train_obs in sources.items():
        for kind in ("probs", "feature+probs"):
            t0 = time.time()
            s_train = build_states(train_obs, kind, device)
            s_select = build_states(select, kind, device)
            s_test = build_states(test, kind, device)
            select_curve, test_curve = [], []
            for cost in COSTS:
                nets = train_policy(s_train, build_rewards(train_obs, cost, device),
                                    hidden=args.hidden, steps=args.steps, seed=cfg["seed"])
                trained[(source, kind, cost)] = nets
                select_curve.append({"cost": cost, **evaluate_stops(apply_policy(nets, s_select), select)})
                test_curve.append({"cost": cost, **evaluate_stops(apply_policy(nets, s_test), test)})
            configs.append({
                "source": source, "kind": kind, "select_curve": select_curve, "test_curve": test_curve,
                "select_earliness_at_baseline_accuracy":
                    earliness_at_accuracy(select_curve, baseline_select["stopping_accuracy"]),
            })
            print(f"{source:5s} {kind:14s} 선택용 데이터, baseline 정확도에서의 earliness: "
                  f"{configs[-1]['select_earliness_at_baseline_accuracy']}  ({time.time() - t0:.0f}s)")
            del s_train

    # ---- 선택용 데이터로 설정과 cost를 고른다 (test는 보지 않는다) ----
    usable = [c for c in configs if c["select_earliness_at_baseline_accuracy"] is not None]
    best = min(usable, key=lambda c: c["select_earliness_at_baseline_accuracy"])
    matched_select = closest(best["select_curve"], baseline_select["stopping_accuracy"])
    cost_matched = matched_select["cost"]
    dos_test = next(p for p in best["test_curve"] if p["cost"] == cost_matched)
    # 비교 대상: 선택용 데이터에서 DOS와 같은 정확도를 내도록 고른 고정 threshold
    tuned_threshold = closest(select_fixed, matched_select["stopping_accuracy"])["threshold"]
    tuned_fixed_test = next(p for p in test_fixed if p["threshold"] == tuned_threshold)

    probs = test.probs
    confidence, correct = probs.max(-1), probs.argmax(-1) == test.labels[:, None]
    targets = [0.90, 0.91, 0.92, baseline_test["stopping_accuracy"], 0.93, 0.935, 0.94]
    result = {
        "run_name": cfg["run_name"],
        "fractions": test.fractions.tolist(),
        "hidden": args.hidden, "steps": args.steps,
        "n_policy_train": {name: int(len(o.labels)) for name, o in sources.items()},
        "n_select": int(len(select.labels)), "n_test": int(len(test.labels)),
        "select_fixed_0.9": baseline_select,
        "selected": {"source": best["source"], "kind": best["kind"], "cost_matched": cost_matched},
        "configs": configs,
        "test": {
            "fixed_0.9": baseline_test,
            "dos_matched": dos_test,
            "tuned_fixed": tuned_fixed_test,
            "dos_curve": best["test_curve"],
            "fixed_curve": test_fixed,
            "same_accuracy": [{"accuracy": a,
                               "fixed": earliness_at_accuracy(test_fixed, a),
                               "dos": earliness_at_accuracy(best["test_curve"], a)} for a in sorted(targets)],
            "accuracy_by_fraction": correct.mean(0).tolist(),
            "confidence_by_fraction": confidence.mean(0).tolist(),
            "ece_by_fraction": [expected_calibration_error(confidence[:, k], correct[:, k])
                                for k in range(confidence.shape[1])],
        },
    }
    with open(out / "dos_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    markdown = to_markdown(result)
    Path(out / "dos.md").write_text(markdown, encoding="utf-8")
    # batch size 1 비용 측정(qd.evaluate)에 쓸 수 있도록 고른 정책을 저장
    nets = trained[(best["source"], best["kind"], cost_matched)]
    torch.save({"kind": best["kind"], "cost": cost_matched, "hidden": args.hidden,
                "d_in": nets[0].net[0].in_features, "n_times": len(nets) + 1,
                "nets": nets.state_dict()}, out / "dos_policy.pt")
    print(markdown)


if __name__ == "__main__":
    main()
