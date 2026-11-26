"""후속 실험: 정지 점수를 무엇으로 학습시키는 것이 좋은가 (B0~B4).

분류 모델은 고정하고, 저장해 둔 관찰 시점별 예측 위에서 "지금 멈춰도 되는가"를 판단하는
작은 신경망(정지 점수 head)만 학습한다. 모든 조건의 정지 threshold는 같은 AGC 설정으로 보정한다.

  조건  학습 목표                     입력                정지 점수
  B0    없음                          -                   Max Softmax (기존 AGC 기준선)
  B1    Correctness                   현재 시점           q_C
  B2    Correctness                   현재 + 과거 이력    q_C
  B3    Harmful Risk                  현재 시점           1 − q_H
  B4    Harmful Risk                  현재 + 과거 이력    1 − q_H

  Correctness  z_C[t] = 1(시점 t의 예측 == 정답)
  Harmful Risk z_H[t] = 1(완성 스케치의 예측 == 정답  그리고  시점 t의 예측 != 정답)
  완성 스케치의 예측과 정답은 학습 목표와 평가에만 쓰고, head의 입력에는 넣지 않는다.

데이터 사용 (Transformer 담당 팀원의 분할을 그대로 따른다)
  train 1,400,000       head 학습
  valid model_selection head 체크포인트 선택 (10,000)
  valid screening       AGC threshold 후보 탐색 (20,000)
  valid calibration     AGC 최종 검정 (20,000, screening과 독립)
  test 50,000           최종 평가. 이미 여러 번 분석에 쓰인 데이터라 결과는 탐색적 성격이다.

실행 (VM의 레포 루트에서):
  python -m qd.stop_scores --config configs/stop_score_study_transformer.yaml
출력: <out_dir>/<name>/results.json, summary.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import rankdata

from qd import agc
from qd.policies import stopping_metrics

CONDITIONS = {  # 이름: (학습 목표, 과거 이력 사용 여부)
    "B1": ("correctness", False),
    "B2": ("correctness", True),
    "B3": ("harmful", False),
    "B4": ("harmful", True),
}
LABELS = {
    "B0": "Max Softmax (학습 없음)",
    "B1": "Correctness, 현재",
    "B2": "Correctness, 현재 + 이력",
    "B3": "Harmful Risk, 현재",
    "B4": "Harmful Risk, 현재 + 이력",
}


# --------------------------------------------------------------------------- #
# 1. 저장된 예측 읽기
# --------------------------------------------------------------------------- #
def load_traces(spec: dict) -> dict:
    """관찰 시점별 예측 → {"logits": [N, K, C] float32, "labels": [N]}.

    format
      npz       : logits, labels (+ indices) 가 든 .npz 파일 (Transformer handoff의 valid/test)
      npy_dir   : logits.npy, labels.npy 가 든 폴더 (export_train_logits.py 출력)
      mamba_obs : qd/dump.py 가 만든 obs_<split>.npz
    """
    path = Path(spec["path"]).expanduser()
    if spec["format"] == "npy_dir":
        out = {"logits": np.load(path / "logits.npy"), "labels": np.load(path / "labels.npy")}
    else:
        with np.load(path) as f:
            out = {"logits": f["logits"].astype(np.float32), "labels": f["labels"].astype(np.int64)}
            if "indices" in f.files:
                out["indices"] = f["indices"]
    out["indices"] = out.get("indices", np.arange(len(out["labels"])))
    return out


def targets(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Correctness [N, K], Harmful Risk [N, K])."""
    correct = logits.argmax(-1) == labels[:, None]
    return correct, correct[:, -1:] & ~correct


# --------------------------------------------------------------------------- #
# 2. head의 입력: 그 시점까지의 예측만 쓴다 (미래 시점의 값은 들어가지 않는다)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_features(logits: torch.Tensor, history: bool) -> torch.Tensor:
    """logits [N, K, C] → 입력 [N, K, F].

    현재 시점: 크기순으로 정렬한 확률, 클래스별 확률, margin(1등 − 2등), entropy, 관찰 시점 one-hot
    과거 이력: 직전 시점 대비 변화와, 지금까지의 예측이 얼마나 안정적이었는지를 요약한 값들
    """
    n, k, c = logits.shape
    p = logits.float().softmax(-1)
    ordered = p.sort(-1, descending=True).values
    conf, margin = ordered[..., 0], ordered[..., 0] - ordered[..., 1]
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / np.log(c)
    when = torch.eye(k, device=logits.device).unsqueeze(0).expand(n, k, k)
    parts = [ordered, p, margin.unsqueeze(-1), entropy.unsqueeze(-1), when]
    if history:
        pred = p.argmax(-1)
        # 첫 관찰 이전은 "아무 정보 없음"(균등 분포)으로 둔다.
        prev_p = torch.cat([torch.full_like(p[:, :1], 1.0 / c), p[:, :-1]], dim=1)
        prev_ordered = prev_p.sort(-1, descending=True).values
        prev_conf, prev_margin = prev_ordered[..., 0], prev_ordered[..., 0] - prev_ordered[..., 1]
        changed = torch.zeros_like(conf)
        changed[:, 1:] = (pred[:, 1:] != pred[:, :-1]).float()          # 예측 클래스가 직전과 달라졌는가
        streak = torch.zeros_like(conf)                                   # 같은 예측을 연속으로 유지한 시점 수
        for t in range(1, k):
            streak[:, t] = (streak[:, t - 1] + 1) * (1 - changed[:, t])
        steps = torch.arange(1, k + 1, device=logits.device, dtype=torch.float32)
        hist = [
            prev_conf, conf - prev_conf, prev_margin, margin - prev_margin,
            changed, changed.cumsum(1) / k, streak / k,
            prev_p.gather(-1, pred.unsqueeze(-1)).squeeze(-1),            # 지금 예측한 클래스를 직전에는 얼마나 믿었나
            (p - prev_p).abs().sum(-1) / 2,                               # 확률 분포가 직전 대비 얼마나 바뀌었나
            conf.cummax(1).values, conf.cumsum(1) / steps,                # 지금까지의 최대·평균 confidence
        ]
        parts.append(torch.stack(hist, dim=-1))
    return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------- #
# 3. 정지 점수 head 학습
# --------------------------------------------------------------------------- #
class ScoreHead(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_head(x_train, z_train, x_select, z_select, hp: dict, seed: int):
    """(시점, 스케치) 쌍을 무작위로 뽑아 BCE로 학습하고, 선택용 데이터의 BCE가 가장 낮은 상태를 고른다."""
    torch.manual_seed(seed)
    rng = torch.Generator(device=x_train.device).manual_seed(seed)
    flat_x, flat_z = x_train.reshape(-1, x_train.shape[-1]), z_train.reshape(-1)
    head = ScoreHead(flat_x.shape[1], hp["hidden"]).to(x_train.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    total = int(hp["epochs"] * len(flat_z) / hp["batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total)
    best, best_state, history = float("inf"), None, []
    for step in range(1, total + 1):
        idx = torch.randint(0, len(flat_z), (hp["batch_size"],), device=flat_x.device, generator=rng)
        loss = F.binary_cross_entropy_with_logits(head(flat_x[idx]), flat_z[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % hp["eval_every"] == 0 or step == total:
            with torch.no_grad():
                select_bce = F.binary_cross_entropy_with_logits(head(x_select), z_select).item()
            history.append({"step": step, "select_bce": select_bce})
            if select_bce < best:
                best = select_bce
                best_state = {name: value.clone() for name, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head.eval(), {"best_select_bce": best, "steps": total, "history": history}


# --------------------------------------------------------------------------- #
# 4. 평가 도구
# --------------------------------------------------------------------------- #
def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """점수가 높을수록 positive인지의 AUROC (순위 기반)."""
    positive = positive.astype(bool).ravel()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(score.ravel())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_policy(score: np.ndarray, thresholds, data: dict) -> tuple[dict, np.ndarray]:
    preds = data["logits"].argmax(-1)
    stops = agc.stop_indices(score, agc.deserialize_thresholds(thresholds))
    return stopping_metrics(stops, preds, data["labels"], data["fractions"]), stops


def paired_bootstrap(stops_a, stops_b, data: dict, repeats: int, seed: int) -> dict:
    """같은 test 스케치를 복원 추출해서, 조건 A − 기준 B 의 차이에 대한 95% 구간을 구한다."""
    preds, labels, fractions = data["logits"].argmax(-1), data["labels"], data["fractions"]
    rows = np.arange(len(labels))
    acc = (preds[rows, stops_a] == labels).astype(np.float64) - (preds[rows, stops_b] == labels)
    earl = fractions[stops_a] - fractions[stops_b]
    rng = np.random.default_rng(seed)
    acc_diffs, earl_diffs = np.empty(repeats), np.empty(repeats)
    for i in range(repeats):
        sample = rng.integers(0, len(labels), len(labels))
        acc_diffs[i], earl_diffs[i] = acc[sample].mean(), earl[sample].mean()
    return {"accuracy_diff": float(acc.mean()), "accuracy_ci95": np.percentile(acc_diffs, [2.5, 97.5]).tolist(),
            "earliness_diff": float(earl.mean()), "earliness_ci95": np.percentile(earl_diffs, [2.5, 97.5]).tolist()}


# --------------------------------------------------------------------------- #
# 5. 실험 실행
# --------------------------------------------------------------------------- #
def run(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fractions = np.asarray(cfg["fractions"], dtype=np.float64)
    hp, ac = cfg["heads"], cfg["agc"]

    train = load_traces(cfg["traces"]["train"])
    valid = load_traces(cfg["traces"]["valid"])
    test = load_traces(cfg["traces"]["test"])
    test["fractions"] = fractions

    # valid 내부 분할: 파일에는 "원래 캐시 행 번호"가 들어 있으므로 예측 파일의 행으로 바꾼다.
    row_of = {int(original): row for row, original in enumerate(valid["indices"])}
    with np.load(Path(cfg["partitions"]).expanduser()) as f:
        parts = {name: np.array([row_of[int(i)] for i in f[name]]) for name in f.files}

    def subset(name: str) -> dict:
        rows = parts[name]
        return {"logits": valid["logits"][rows], "labels": valid["labels"][rows], "fractions": fractions}

    select, screen, calib = subset("model_selection"), subset("screening"), subset("calibration")
    splits = {"select": select, "screen": screen, "calib": calib, "test": test}

    def confidence(data: dict) -> np.ndarray:
        z = torch.from_numpy(data["logits"]).float().softmax(-1)
        return z.max(-1).values.numpy()

    def calibrate(score_by_split: dict, delta: float) -> dict:
        pack = lambda name: {"confidence": score_by_split[name], "predictions": splits[name]["logits"].argmax(-1),
                             "labels": splits[name]["labels"]}
        return agc.calibrate(pack("screen"), pack("calib"), ac["alpha"], delta, ac["grid_step"])

    result = {"name": cfg["name"], "config": cfg, "n": {k: int(len(v["labels"])) for k, v in splits.items()},
              "n_train": int(len(train["labels"])), "conditions": {}, "labels": LABELS}

    # ---- B0: Max Softmax ----
    conf = {name: confidence(data) for name, data in splits.items()}
    policy = calibrate(conf, ac["delta"])
    metrics, stops_b0 = evaluate_policy(conf["test"], policy["thresholds"], test)
    result["conditions"]["B0"] = {"thresholds": policy["thresholds"], "candidate_thresholds": policy["candidate_thresholds"],
                                  "test": metrics}
    if ac.get("reference_delta"):  # 팀원의 기존 AGC 결과(δ가 다름)가 재현되는지 확인용
        ref = calibrate(conf, ac["reference_delta"])
        result["reference_b0"] = {"delta": ac["reference_delta"], "thresholds": ref["thresholds"],
                                  "test": evaluate_policy(conf["test"], ref["thresholds"], test)[0]}
    fixed = np.full(len(fractions), 0.9)  # 팀 공통 baseline 규칙 (참고용)
    result["fixed_0.9"] = stopping_metrics(agc.stop_indices(conf["test"], fixed), test["logits"].argmax(-1),
                                           test["labels"], fractions)

    # ---- B1~B4: 정지 점수 head ----
    z_correct, z_harmful = targets(train["logits"], train["labels"])
    train_logits = torch.from_numpy(train["logits"]).to(device)
    z_train = {"correctness": torch.from_numpy(z_correct).float().to(device),
               "harmful": torch.from_numpy(z_harmful).float().to(device)}
    features = {}
    for use_history in (False, True):
        features[use_history] = {
            "train": build_features(train_logits, use_history),
            **{name: build_features(torch.from_numpy(data["logits"]).to(device), use_history)
               for name, data in splits.items()},
        }
    del train_logits
    eval_targets = {name: targets(data["logits"], data["labels"]) for name, data in splits.items()}

    for cond, (target, use_history) in CONDITIONS.items():
        feats = features[use_history]
        z_select = torch.from_numpy(eval_targets["select"][0 if target == "correctness" else 1]).float().to(device)
        runs = []
        for seed in hp["seeds"]:
            t0 = time.time()
            head, log = train_head(feats["train"], z_train[target], feats["select"], z_select, hp, seed)
            with torch.no_grad():
                prob = {name: torch.sigmoid(head(feats[name])).cpu().numpy() for name in splits}
            # AGC는 "점수가 클수록 멈춰도 안전"을 가정하므로, 위험 확률은 1에서 뺀다.
            score = prob if target == "correctness" else {name: 1.0 - value for name, value in prob.items()}
            policy = calibrate(score, ac["delta"])
            metrics, stops = evaluate_policy(score["test"], policy["thresholds"], test)
            correct_test, harmful_test = eval_targets["test"]
            runs.append({
                "seed": seed, "thresholds": policy["thresholds"], "candidate_thresholds": policy["candidate_thresholds"],
                "test": metrics, "head": {k: v for k, v in log.items() if k != "history"},
                "auroc_correctness": auroc(score["test"][:, :-1], correct_test[:, :-1]),
                "auroc_not_harmful": auroc(score["test"][:, :-1], ~harmful_test[:, :-1]),
                "vs_B0": paired_bootstrap(stops, stops_b0, test, cfg["bootstrap"]["repeats"], cfg["bootstrap"]["seed"]),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"{cond} seed {seed}: acc {metrics['stopping_accuracy']:.4f} earliness {metrics['mean_earliness']:.4f} "
                  f"harmful {metrics['harmful_gap']:.4f} ({runs[-1]['seconds']}s)", flush=True)
        result["conditions"][cond] = {"target": target, "history": use_history,
                                      "n_features": int(feats["train"].shape[-1]), "runs": runs}

    correct_test, harmful_test = eval_targets["test"]
    result["conditions"]["B0"]["auroc_correctness"] = auroc(conf["test"][:, :-1], correct_test[:, :-1])
    result["conditions"]["B0"]["auroc_not_harmful"] = auroc(conf["test"][:, :-1], ~harmful_test[:, :-1])
    return result


def to_markdown(r: dict) -> str:
    cfg, ac = r["config"], r["config"]["agc"]

    def fmt_thr(values) -> str:
        return " | ".join("금지" if v is None else f"{v:.3f}" for v in values)

    def mean_std(values) -> str:
        return f"{np.mean(values):.2%} ± {np.std(values):.2%}" if len(values) > 1 else f"{values[0]:.2%}"

    b0 = r["conditions"]["B0"]
    lines = [
        f"# {r['name']}: 정지 점수 학습 목표 비교 (B0~B4)",
        "",
        "## 조건",
        f"- head 학습 train {r['n_train']:,}개, head 선택 {r['n']['select']:,}개, AGC screening {r['n']['screen']:,}개, "
        f"AGC calibration {r['n']['calib']:,}개, 최종 평가 test {r['n']['test']:,}개",
        f"- AGC: α = {ac['alpha']}, δ = {ac['delta']} (조건 5개에 δ 0.05를 나눔), grid {ac['grid_step']}",
        f"- head: MLP (은닉 {cfg['heads']['hidden']} × 2층), seed {cfg['heads']['seeds']}. "
        "B1~B4의 수치는 seed별 결과의 평균 ± 표준편차",
        "",
        "## test 결과",
        "| 조건 | 학습 목표, 입력 | Stopping Accuracy | Mean Earliness | Harmful Gap | 최대 누적 조건부 Harmful Gap |",
        "|---|---|---|---|---|---|",
        f"| Fixed 0.9 | (참고) | {r['fixed_0.9']['stopping_accuracy']:.2%} | {r['fixed_0.9']['mean_earliness']:.2%} | "
        f"{r['fixed_0.9']['harmful_gap']:.2%} | {r['fixed_0.9']['max_accumulated_harmful_gap']:.2%} |",
        f"| B0 | {r['labels']['B0']} | {b0['test']['stopping_accuracy']:.2%} | {b0['test']['mean_earliness']:.2%} | "
        f"{b0['test']['harmful_gap']:.2%} | {b0['test']['max_accumulated_harmful_gap']:.2%} |",
    ]
    for cond in CONDITIONS:
        tests = [run["test"] for run in r["conditions"][cond]["runs"]]
        lines.append(f"| {cond} | {r['labels'][cond]} | "
                     + " | ".join(mean_std([t[key] for t in tests]) for key in
                                  ("stopping_accuracy", "mean_earliness", "harmful_gap", "max_accumulated_harmful_gap"))
                     + " |")
    lines += ["", "## B0 대비 차이 (같은 test 스케치로 paired bootstrap, 95% 구간, 첫 번째 seed)",
              "| 조건 | Stopping Accuracy 차이 | Mean Earliness 차이 |", "|---|---|---|"]
    for cond in CONDITIONS:
        d = r["conditions"][cond]["runs"][0]["vs_B0"]
        lines.append(f"| {cond} | {d['accuracy_diff']:+.2%} [{d['accuracy_ci95'][0]:+.2%}, {d['accuracy_ci95'][1]:+.2%}] | "
                     f"{d['earliness_diff']:+.2%} [{d['earliness_ci95'][0]:+.2%}, {d['earliness_ci95'][1]:+.2%}] |")
    lines += ["", "## 점수가 두 목표를 얼마나 잘 가르는가 (test, 10~90% 시점, AUROC)",
              "| 조건 | 지금 예측이 정답인지 | 멈춰도 손해가 없는지 |", "|---|---|---|",
              f"| B0 | {b0['auroc_correctness']:.4f} | {b0['auroc_not_harmful']:.4f} |"]
    for cond in CONDITIONS:
        runs = r["conditions"][cond]["runs"]
        lines.append(f"| {cond} | {np.mean([x['auroc_correctness'] for x in runs]):.4f} | "
                     f"{np.mean([x['auroc_not_harmful'] for x in runs]):.4f} |")
    lines += ["", "## AGC가 고른 시점별 threshold (B1~B4는 첫 번째 seed)",
              "| 조건 | " + " | ".join(f"{f:.0%}" for f in cfg["fractions"]) + " |",
              "|---|" + "---|" * len(cfg["fractions"]),
              f"| B0 | {fmt_thr(b0['thresholds'])} |"]
    for cond in CONDITIONS:
        lines.append(f"| {cond} | {fmt_thr(r['conditions'][cond]['runs'][0]['thresholds'])} |")
    if r.get("reference_b0"):
        ref = r["reference_b0"]
        lines += ["", f"참고: B0를 δ = {ref['delta']}로 보정하면 Stopping Accuracy {ref['test']['stopping_accuracy']:.3%}, "
                  f"Mean Earliness {ref['test']['mean_earliness']:.3%} (기존 Baseline + AGC 결과와 대조용)"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="정지 점수 학습 목표 비교 (B0~B4)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["out_dir"]).expanduser() / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    result = run(cfg)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    markdown = to_markdown(result)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
