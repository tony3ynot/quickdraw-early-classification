"""조기 예측(early stopping) 평가.

실행 (VM의 레포 루트에서, 가상환경 활성화 후):
  python -m qd.evaluate --config configs/mamba_20cls.yaml              # test 전체
  python -m qd.evaluate --config configs/mamba_20cls.yaml --limit 2000 # 빠른 확인

팀 공통 프로토콜
  - 관찰 시점: 전체 점 개수의 10%, 20%, ..., 100% (qd.data.prefix_lengths)
  - 정지 정책: 그 시점의 최대 클래스 확률이 threshold(0.90) 이상이면 예측 확정.
               100%까지 넘지 못하면 100% 시점의 예측을 사용.
  - batch size 1

연산량과 지연 시간을 세는 두 가지 방식 (둘 다 실제로 실행해서 기록한다)
  streaming : 상태를 유지하면서 새로 들어온 점만 처리한다. 이미 본 점은 다시 계산하지 않는다.
              Mamba/GRU처럼 순서대로 읽는 모델에서 가능하며, 예측은 recompute와 같다.
              model.init_stream_state / step / classify 가 필요하다.
  recompute : 관찰 시점마다 그때까지 본 prefix 전체를 처음부터 다시 추론한다.
              어떤 모델에도 똑같이 적용할 수 있는 기준선. model(x, last_only=True)만 있으면 된다.
두 방식은 같은 예측과 같은 정지 시점을 내고 비용만 다르다 (일치율을 결과에 기록한다).
FLOPs는 streaming이 이 모델에 실제로 필요한 계산량이다. 지연 시간은 구현에 좌우된다:
streaming은 점 하나마다 모델을 호출하고, recompute는 관찰 시점마다 한 번만 호출한다.

공통 가정
  - FLOPs = 처리한 점 개수 × 점 하나당 FLOPs + 예측 횟수 × 분류 head FLOPs (qd/flops.py)
  - 관찰 시점이 바뀌어도 본 점 개수가 그대로면(아주 짧은 스케치) 다시 예측하지 않는다.
  - 지연 시간은 GPU에서 모델 계산 + softmax + 결과를 CPU로 가져오는 시간이다.
    입력 정규화 같은 전처리는 포함하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from qd.common import count_parameters, load_config, run_dir
from qd.data import SketchSplit, build_cache, load_meta, load_split, prefix_lengths
from qd.flops import flops_per_token, head_flops
from qd.model import build_model

MODE_LABEL = {"streaming": "상태 유지 (새 점만 처리)", "recompute": "시점마다 재추론"}


def fraction_predictions(logits: torch.Tensor, lengths: torch.Tensor, fractions) -> torch.Tensor:
    """매 시점 logits [B, L, C] → 각 관찰 비율 시점의 예측 클래스 [B, K].

    causal 모델에서는 앞 m개 점만 본 예측이 위치 m-1의 출력과 같으므로,
    전체 시퀀스를 한 번 추론한 뒤 해당 위치만 골라내면 된다.
    """
    last_index = prefix_lengths(lengths, fractions) - 1
    return logits.argmax(-1).gather(1, last_index)


@torch.inference_mode()
def accuracy_by_fraction(model, split: SketchSplit, fractions, batch_size: int) -> list[float]:
    """정지 정책 없이, 각 관찰 비율에서 예측했을 때의 정확도."""
    model.eval()
    correct = torch.zeros(len(fractions), device=split.points.device)
    for x, lengths, labels in split.batches(batch_size):
        preds = fraction_predictions(model(x), lengths, fractions)
        correct += (preds == labels.unsqueeze(1)).sum(0)
    return (correct / len(split)).tolist()


# --------------------------------------------------------------------------- #
# 스케치 한 장에 정지 정책을 실행하는 두 가지 방식
#   입력: x [1, 점 개수, 3], n_obs [K] 시점별 관찰 점 개수
#   출력: (정지 시점 index, 예측, 확률, 누적 지연(초), 처리한 점 개수, 예측 횟수)
# --------------------------------------------------------------------------- #
def _read_top(logits: torch.Tensor) -> tuple[float, int]:
    """logits [1, C] → (최대 확률, 그 클래스). .item()이 GPU 계산이 끝나기를 기다린다."""
    top = logits.softmax(-1).max(-1)
    return top.values.item(), top.indices.item()


# ---- 정지 정책: (시점 index, feature, 클래스 확률, 정지 logit) → 멈출지 여부 [1] bool 텐서 ----
class FixedThresholdPolicy:
    """팀 공통 baseline 규칙: 최대 클래스 확률이 threshold 이상이면 정지."""
    time_dependent = False  # 같은 입력이면 어느 시점이든 같은 결정
    flops = 0

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.label = f"Fixed threshold {threshold}"

    def __call__(self, k, feature, probs, stop_logit):
        return probs.max(-1).values >= self.threshold


class StopHeadPolicy:
    """ELECTS: 모델의 정지 head가 낸 정지 확률이 기준 이상이면 정지."""
    time_dependent = False
    flops = 0  # 정지 head 비용은 qd/flops.py의 head_flops에 이미 포함된다

    def __init__(self, stop_prob: float):
        self.logit = math.log(stop_prob / (1 - stop_prob))
        self.label = f"ELECTS 정지 head (정지 확률 ≥ {stop_prob})"

    def __call__(self, k, feature, probs, stop_logit):
        return stop_logit >= self.logit


class DOSPolicy:
    """Deep Optimal Stopping: 관찰 시점마다 따로 학습된 정지 신경망 (qd/optimal_stopping.py)."""
    time_dependent = True  # 시점마다 다른 신경망을 쓴다

    def __init__(self, path, device):
        from qd.optimal_stopping import StopNet
        saved = torch.load(path, map_location=device, weights_only=True)
        self.kind = saved["kind"]
        self.nets = torch.nn.ModuleList(
            StopNet(saved["d_in"], saved["hidden"]) for _ in range(saved["n_times"] - 1)).to(device)
        self.nets.load_state_dict(saved["nets"])
        self.nets.eval()
        self.flops = sum(m.in_features * m.out_features
                         for m in self.nets[0].modules() if isinstance(m, torch.nn.Linear))
        self.label = f"Deep Optimal Stopping (cost {saved['cost']}, 입력 {self.kind})"

    def __call__(self, k, feature, probs, stop_logit):
        if k >= len(self.nets):  # 마지막 시점은 강제 정지
            return torch.ones(1, dtype=torch.bool, device=probs.device)
        parts = [probs, probs.sort(dim=-1, descending=True).values]  # qd.optimal_stopping.build_states와 같은 입력
        if self.kind == "feature+probs":
            parts.insert(0, feature)
        return self.nets[k](torch.cat(parts, dim=-1)) >= 0


def _decide(policy, k: int, readout) -> tuple[float, int, bool]:
    """모델 출력 → (최대 확률, 예측 클래스, 멈출지). GPU에서 값을 한 번에 가져온다."""
    feature, logits, stop_logit = readout
    probs = logits.softmax(-1)
    top = probs.max(-1)
    stop = policy(k, feature, probs, stop_logit)
    conf, pred, stop = torch.stack([top.values, top.indices.float(), stop.float()]).flatten().tolist()
    return conf, int(pred), bool(stop)


def _run_recompute(model, x, n_obs, policy, state=None):
    """관찰 시점마다 앞 m개 점을 처음부터 다시 추론한다."""
    seen, readout, conf, pred, stop = 0, None, 0.0, -1, False
    latency, points_processed, n_readouts, n_decisions = 0.0, 0, 0, 0
    for k, m in enumerate(n_obs):
        m = int(m)
        new_points = m != seen
        if new_points or policy.time_dependent:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if new_points:  # 새 점이 들어왔을 때만 모델을 다시 돌린다
                readout = model.readout(model.encode(x[:, :m], last_only=True))
                points_processed += m
                n_readouts += 1
                seen = m
            conf, pred, stop = _decide(policy, k, readout)
            latency += time.perf_counter() - t0
            n_decisions += 1
        if stop or k == len(n_obs) - 1:
            break
    return k, pred, conf, latency, points_processed, n_readouts, n_decisions, stop


def _run_streaming(model, x, n_obs, policy, state):
    """상태를 유지하면서, 직전 관찰 시점 이후에 새로 들어온 점만 하나씩 처리한다."""
    for conv_state, ssm_state in state:  # 새 스케치이므로 상태를 비운다
        conv_state.zero_()
        ssm_state.zero_()
    seen, readout, conf, pred, stop = 0, None, 0.0, -1, False
    latency, n_readouts, n_decisions = 0.0, 0, 0
    for k, m in enumerate(n_obs):
        m = int(m)
        new_points = m != seen
        if new_points or policy.time_dependent:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if new_points:
                for t in range(seen, m):  # 새 점들만
                    hidden = model.step(x[:, t], state)
                readout = model.readout(hidden)
                n_readouts += 1
                seen = m
            conf, pred, stop = _decide(policy, k, readout)
            latency += time.perf_counter() - t0
            n_decisions += 1
        if stop or k == len(n_obs) - 1:
            break
    return k, pred, conf, latency, seen, n_readouts, n_decisions, stop


RUNNERS = {"recompute": _run_recompute, "streaming": _run_streaming}


@torch.inference_mode()
def early_stopping_eval(model, split: SketchSplit, indices: torch.Tensor, fractions,
                        policy, warmup_iters: int, mode: str) -> dict:
    """batch size 1로 정지 정책을 실제로 실행하면서 스케치별 기록을 남긴다."""
    model.eval()
    device = split.points.device
    n_obs = prefix_lengths(split.lengths, fractions).cpu().numpy()  # [N, K]
    run = RUNNERS[mode]
    state = model.init_stream_state(batch_size=1) if mode == "streaming" else None

    def sketch(i: int) -> torch.Tensor:
        return split.batch(torch.tensor([i], device=device))[0]  # [1, 점 개수, 3]

    # GPU 예열: 처음 몇 번은 느리므로 측정에서 뺀다.
    for i in indices[:warmup_iters].tolist():
        run(model, sketch(i), n_obs[i], policy, state)

    n = len(indices)
    rec = {
        "stop_index": np.zeros(n, dtype=np.int64),        # 몇 번째 관찰 시점에서 멈췄는지
        "pred": np.zeros(n, dtype=np.int64),
        "confidence": np.zeros(n, dtype=np.float64),
        "latency": np.zeros(n, dtype=np.float64),          # 정지까지 누적 추론 시간(초)
        "points_processed": np.zeros(n, dtype=np.int64),   # 정지까지 모델이 계산한 점 개수
        "n_readouts": np.zeros(n, dtype=np.int64),         # 정지까지 모델 예측을 읽은 횟수
        "n_decisions": np.zeros(n, dtype=np.int64),        # 정지까지 정지 정책을 실행한 횟수
        "stopped_by_policy": np.zeros(n, dtype=bool),      # 강제 정지가 아니라 정책이 멈췄는지
    }
    for row, i in enumerate(indices.tolist()):
        (rec["stop_index"][row], rec["pred"][row], rec["confidence"][row], rec["latency"][row],
         rec["points_processed"][row], rec["n_readouts"][row], rec["n_decisions"][row],
         rec["stopped_by_policy"][row]) = run(model, sketch(i), n_obs[i], policy, state)
    return rec


@torch.inference_mode()
def single_pass_latency(model, split: SketchSplit, indices: torch.Tensor, warmup_iters: int) -> np.ndarray:
    """참고용: 조기 정지 없이 완성된 스케치를 한 번만 추론하는 시간(초)."""
    model.eval()
    device = split.points.device
    out = np.zeros(len(indices))
    for i in indices[:warmup_iters].tolist():  # 예열
        _read_top(model(split.batch(torch.tensor([i], device=device))[0], last_only=True))
    for row, i in enumerate(indices.tolist()):
        x = split.batch(torch.tensor([i], device=device))[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _read_top(model(x, last_only=True))
        out[row] = time.perf_counter() - t0
    return out


def summarize(rec: dict, labels: np.ndarray, fractions, policy_flops: int,
              classes: list[str], per_token: int, head: int) -> dict:
    """스케치별 기록 → 팀 공통 지표."""
    fractions = np.asarray(fractions)
    correct = rec["pred"] == labels
    stop_fraction = fractions[rec["stop_index"]]
    flops = (per_token * rec["points_processed"] + head * rec["n_readouts"]
             + policy_flops * rec["n_decisions"])
    latency_ms = rec["latency"] * 1e3

    result = {
        "stopping_accuracy": float(correct.mean()),
        "mean_earliness": float(stop_fraction.mean()),
        "cumulative_flops_mean": float(flops.mean()),
        "cumulative_latency_ms": {
            "mean": float(latency_ms.mean()),
            "median": float(np.median(latency_ms)),
            "p95": float(np.percentile(latency_ms, 95)),
        },
        "mean_points_processed": float(rec["points_processed"].mean()),
        "mean_readouts": float(rec["n_readouts"].mean()),
        "stopped_by_policy_rate": float(rec["stopped_by_policy"].mean()),
        "stop_distribution": [],
        "per_class": [],
    }
    for k, fraction in enumerate(fractions):
        here = rec["stop_index"] == k
        result["stop_distribution"].append({
            "fraction": float(fraction),
            "share": float(here.mean()),
            "accuracy": float(correct[here].mean()) if here.any() else None,
        })
    for c, name in enumerate(classes):
        here = labels == c
        if here.any():
            result["per_class"].append({
                "class": name,
                "n": int(here.sum()),
                "stopping_accuracy": float(correct[here].mean()),
                "mean_earliness": float(stop_fraction[here].mean()),
            })
    return result


def to_markdown(r: dict) -> str:
    """결과 요약을 팀에 공유하기 좋은 markdown으로 만든다."""
    es, info, single = r["early_stopping"], r["info"], r["single_pass"]
    modes = list(es["modes"])
    first = es["modes"][modes[0]]  # 정지 분포·클래스별 표에 쓴다 (두 방식의 결정은 같다)

    def row(name: str, value) -> str:
        return f"| {name} | " + " | ".join(value(es["modes"][m]) for m in modes) + " |"

    lines = [
        f"# {info['run_name']}: {info['split']} 평가 결과",
        "",
        "## 조건",
        f"- 데이터: 공식 Sketch-RNN QuickDraw 분할, {info['n_classes']}개 클래스, "
        f"{info['split']} {info['n_split']:,}개 중 {es['n_sketches']:,}개 평가",
        f"- 모델: {info['model']} (파라미터 {info['n_params']:,}개), 체크포인트 epoch {info['checkpoint_epoch']}",
        f"- 정지 정책: {es['policy']}",
        f"- 평가: batch size 1, {info['gpu']}, fp32, "
        f"관찰 시점 {len(r['fractions'])}개 ({r['fractions'][0]:.0%}~{r['fractions'][-1]:.0%})",
    ]
    if info.get("train"):
        t = info["train"]
        lines.append(f"- 학습: train {t['n_train']:,}개, {t['epochs_done']} epoch, "
                     f"{t['train_hours']:.2f}시간 ({t['gpu']}), best epoch {t['best_epoch']}")
    lines += [
        "",
        "## 조기 예측 지표",
        "| 지표 | " + " | ".join(MODE_LABEL[m] for m in modes) + " |",
        "|---|" + "---|" * len(modes),
        row("Stopping Accuracy", lambda s: f"{s['stopping_accuracy']:.2%}"),
        row("Mean Earliness", lambda s: f"{s['mean_earliness']:.1%}"),
        row("Cumulative FLOPs", lambda s: f"{s['cumulative_flops_mean'] / 1e6:.1f} M"),
        row("Cumulative Latency (평균)", lambda s: f"{s['cumulative_latency_ms']['mean']:.2f} ms"),
        row("Cumulative Latency (중앙값 / p95)",
            lambda s: f"{s['cumulative_latency_ms']['median']:.2f} / {s['cumulative_latency_ms']['p95']:.2f} ms"),
        row("정지까지 모델이 계산한 점 개수 (평균)", lambda s: f"{s['mean_points_processed']:.1f}"),
        row("정지까지 예측 횟수 (평균)", lambda s: f"{s['mean_readouts']:.2f}"),
        row("강제 정지(100%) 전에 정책이 멈춘 비율", lambda s: f"{s['stopped_by_policy_rate']:.1%}"),
        "",
    ]
    if es.get("decision_agreement") is not None:
        lines.append(f"- 두 방식의 정지 시점과 예측이 일치한 스케치: {es['decision_agreement']:.3%} "
                     "(비용만 다르고 결정은 같다)")
    if "streaming" in es["modes"]:
        s = es["modes"]["streaming"]
        lines.append("- 상태 유지는 점 하나마다 모델을 호출한다: 점 하나당 평균 "
                     f"{s['cumulative_latency_ms']['mean'] / s['mean_points_processed']:.2f} ms")
    if "recompute" in es["modes"]:
        s = es["modes"]["recompute"]
        lines.append("- 재추론은 관찰 시점마다 한 번만 호출한다: 호출 한 번당 평균 "
                     f"{s['cumulative_latency_ms']['mean'] / s['mean_readouts']:.2f} ms")
    lines += [
        "",
        "참고 (조기 정지 없이 완성된 스케치를 한 번 추론): "
        f"정확도 {single['accuracy']:.2%}, FLOPs {single['flops_mean'] / 1e6:.1f} M, "
        f"지연 {single['latency_ms_mean']:.2f} ms",
        "",
        "## 관찰 비율별 정확도 (정지 정책 없이 그 시점에 예측)",
        "| 관찰 비율 | " + " | ".join(f"{f:.0%}" for f in r["fractions"]) + " |",
        "|---|" + "---|" * len(r["fractions"]),
        "| 정확도 | " + " | ".join(f"{a:.1%}" for a in r["accuracy_by_fraction"]) + " |",
        "",
        "## 정지 시점 분포",
        "| 정지 시점 | 비율 | 그 시점에 멈춘 스케치의 정확도 |",
        "|---|---|---|",
    ]
    for d in first["stop_distribution"]:
        acc = "-" if d["accuracy"] is None else f"{d['accuracy']:.1%}"
        lines.append(f"| {d['fraction']:.0%} | {d['share']:.1%} | {acc} |")
    lines += ["", "## 클래스별", "| 클래스 | 개수 | Stopping Accuracy | Mean Earliness |", "|---|---|---|---|"]
    for c in first["per_class"]:
        lines.append(f"| {c['class']} | {c['n']} | {c['stopping_accuracy']:.1%} | {c['mean_earliness']:.1%} |")
    lines += [
        "",
        "## 측정 가정",
        "- 관찰 비율 f 시점에는 앞에서부터 ceil(점 개수 × f)개(최소 1개)의 점을 본다.",
        "- 상태 유지: 상태를 이어 가며 새로 들어온 점만 하나씩 처리한다 (mamba-ssm의 step). "
        "시점마다 재추론: 그때까지의 prefix 전체를 처음부터 다시 추론한다. 두 방식 모두 실제로 실행해 측정했다.",
        "- FLOPs는 곱셈-덧셈 1회를 1로 세고(fvcore 기준), 활성화·정규화는 세지 않는다. "
        f"점 하나당 {r['flops_per_point']:,} FLOPs에 모델이 계산한 점 개수를 곱한다.",
        "- 지연 시간은 모델 계산 + softmax + 결과를 CPU로 가져오는 시간이며 전처리는 제외했다.",
        "- 모델은 causal이어서 각 시점의 예측에 그 뒤의 점을 쓰지 않는다. "
        "단, 원본 데이터의 좌표는 완성된 스케치 기준으로 0~255로 맞춰져 있다 (README의 '알려진 한계').",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="조기 예측 평가")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="기본값: ~/runs/<run_name>/best.pt")
    parser.add_argument("--split", default=None, choices=["valid", "test"])
    parser.add_argument("--limit", type=int, default=None, help="무작위로 이 개수만 평가 (빠른 확인용)")
    parser.add_argument("--run-name", default=None, help="설정 파일의 run_name 덮어쓰기")
    parser.add_argument("--policy", default="fixed", choices=["fixed", "stop_head", "dos"],
                        help="정지 정책: fixed(설정의 threshold), stop_head(ELECTS), dos(~/runs/<run>/dos_policy.pt)")
    parser.add_argument("--stop-prob", type=float, default=0.5, help="stop_head 정책의 정지 확률 기준")
    parser.add_argument("--modes", nargs="+", default=None, choices=["streaming", "recompute"],
                        help="설정 파일의 eval.modes 덮어쓰기")
    parser.add_argument("--tag", default=None, help="결과 파일 이름에 붙일 꼬리표 (eval_<split>_<tag>)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.modes:
        cfg["eval"]["modes"] = args.modes
    ec = cfg["eval"]
    out = run_dir(cfg)
    split_name = args.split or ec["split"]
    device = torch.device("cuda")

    # 모델과 데이터 설정은 학습 때 체크포인트에 저장해 둔 것을 그대로 쓴다.
    ckpt = torch.load(args.checkpoint or out / "best.pt", map_location="cpu", weights_only=True)
    cache = build_cache(ckpt["config"]["data"])
    meta = load_meta(cache)
    split = load_split(cache, split_name).to(device)
    model = build_model(ckpt["config"]["model"], n_classes=len(meta["classes"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 캐시는 클래스 순서대로 저장돼 있으므로, 일부만 평가할 때는 무작위로 뽑는다.
    indices = torch.arange(len(split))
    if args.limit and args.limit < len(split):
        rng = torch.Generator().manual_seed(cfg["seed"])
        indices = torch.randperm(len(split), generator=rng)[: args.limit].sort().values
    labels = split.labels[indices].cpu().numpy()
    n_points = split.lengths[indices].cpu().numpy()
    per_token, head = flops_per_token(model), head_flops(model)
    if args.policy == "fixed":
        policy = FixedThresholdPolicy(ec["threshold"])
    elif args.policy == "stop_head":
        policy = StopHeadPolicy(args.stop_prob)
    else:
        policy = DOSPolicy(out / "dos_policy.pt", device)

    curve = accuracy_by_fraction(model, split, ec["fractions"], cfg["train"]["eval_batch_size"])

    # 정지 정책을 방식별로 실제 실행
    records, seconds = {}, {}
    for mode in ec["modes"]:
        t0 = time.time()
        records[mode] = early_stopping_eval(model, split, indices, ec["fractions"],
                                            policy, ec["warmup_iters"], mode)
        seconds[mode] = round(time.time() - t0, 1)
    agreement = None
    if len(records) == 2:  # 두 방식이 같은 결정을 내리는지 (수치 오차로 아주 드물게 달라질 수 있다)
        a, b = records.values()
        agreement = float(((a["stop_index"] == b["stop_index"]) & (a["pred"] == b["pred"])).mean())
    full_latency = single_pass_latency(model, split, indices, ec["warmup_iters"])

    result = {
        "info": {
            "run_name": cfg["run_name"],
            "split": split_name,
            "n_split": len(split),
            "n_classes": len(meta["classes"]),
            "classes": meta["classes"],
            "model": f"{ckpt['config']['model']['n_layers']}-layer Mamba, d_model {ckpt['config']['model']['d_model']}",
            "model_config": ckpt["config"]["model"],
            "n_params": count_parameters(model),
            "checkpoint_epoch": ckpt["epoch"],
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "eval_seconds": seconds,
        },
        "fractions": list(ec["fractions"]),
        "accuracy_by_fraction": curve,
        "flops_per_point": per_token,
        "head_flops": head,
        "single_pass": {  # 조기 정지를 쓰지 않을 때: 완성된 스케치를 한 번 추론
            "accuracy": curve[-1],
            "flops_mean": float(per_token * n_points.mean() + head),
            "latency_ms_mean": float(full_latency.mean() * 1e3),
        },
        "early_stopping": {
            "policy": policy.label,
            "policy_flops_per_decision": policy.flops,
            "n_sketches": int(len(indices)),
            "decision_agreement": agreement,
            "modes": {
                mode: summarize(rec, labels, ec["fractions"], policy.flops,
                                meta["classes"], per_token, head)
                for mode, rec in records.items()
            },
        },
    }
    summary_path = out / "train_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            result["info"]["train"] = json.load(f)

    suffix = (f"_{args.tag}" if args.tag else "") + (f"_limit{args.limit}" if args.limit else "")
    json_path = out / f"eval_{split_name}{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    markdown = to_markdown(result)
    Path(out / f"eval_{split_name}{suffix}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
