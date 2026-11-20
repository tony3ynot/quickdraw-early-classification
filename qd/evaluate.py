"""조기 예측(early stopping) 평가.

실행 (VM의 레포 루트에서, 가상환경 활성화 후):
  python -m qd.evaluate --config configs/mamba_20cls.yaml              # test 전체
  python -m qd.evaluate --config configs/mamba_20cls.yaml --limit 2000 # 빠른 확인

팀 공통 프로토콜
  - 관찰 시점: 전체 점 개수의 10%, 20%, ..., 100% (qd.data.prefix_lengths)
  - 정지 정책: 그 시점의 최대 클래스 확률이 threshold(0.90) 이상이면 예측 확정.
               100%까지 넘지 못하면 100% 시점의 예측을 사용.
  - batch size 1

연산량과 지연 시간을 세는 방식 (결과 파일에도 기록된다)
  (A) recompute: 관찰 시점마다 그때까지 본 prefix 전체를 처음부터 다시 추론한다.
      어떤 모델에도 똑같이 적용할 수 있어 팀 비교용 주 지표로 쓴다. 지연 시간은 이 방식으로 실측.
  (B) streaming: 상태를 유지하면서 새로 들어온 점만 처리한다고 가정한 FLOPs.
      Mamba/GRU처럼 상태를 이어 갈 수 있는 모델에서만 가능하다. 계산값만 보고하고 지연 시간은 재지 않는다.
  - 관찰 시점이 바뀌어도 본 점 개수가 그대로면(아주 짧은 스케치) 다시 추론하지 않는다.
  - 지연 시간은 GPU에서 forward + softmax + 결과를 CPU로 가져오는 시간이다.
    입력 정규화 같은 전처리는 포함하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qd.common import count_parameters, load_config, run_dir
from qd.data import SketchSplit, build_cache, load_meta, load_split, prefix_lengths
from qd.flops import flops_per_token, head_flops
from qd.model import build_model


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


@torch.inference_mode()
def early_stopping_eval(model, split: SketchSplit, indices: torch.Tensor, fractions,
                        threshold: float, warmup_iters: int) -> dict:
    """batch size 1로 정지 정책을 실제로 실행하면서 스케치별 기록을 남긴다."""
    model.eval()
    device = split.points.device
    n_obs = prefix_lengths(split.lengths, fractions).cpu().numpy()  # [N, K] 시점별 관찰 점 개수
    last = len(fractions) - 1

    # GPU 예열: 처음 몇 번의 forward는 느리므로 측정에서 뺀다.
    for i in indices[:warmup_iters].tolist():
        x, _, _ = split.batch(torch.tensor([i], device=device))
        model(x, last_only=True)
    torch.cuda.synchronize()

    n = len(indices)
    rec = {
        "stop_index": np.zeros(n, dtype=np.int64),         # 몇 번째 관찰 시점에서 멈췄는지
        "pred": np.zeros(n, dtype=np.int64),
        "confidence": np.zeros(n, dtype=np.float64),
        "latency": np.zeros(n, dtype=np.float64),          # 정지까지 누적 추론 시간(초), 방식 A
        "points_recomputed": np.zeros(n, dtype=np.int64),  # 방식 A에서 처리한 점 개수의 합
        "points_seen": np.zeros(n, dtype=np.int64),        # 정지 시점까지 관찰한 점 개수
        "n_forward": np.zeros(n, dtype=np.int64),
        "full_latency": np.zeros(n, dtype=np.float64),     # 참고: 전체 스케치를 한 번 추론하는 시간
    }

    for row, i in enumerate(indices.tolist()):
        x, _, _ = split.batch(torch.tensor([i], device=device))  # [1, n_points, 3]
        seen, conf, pred = 0, 0.0, -1
        for k in range(len(fractions)):
            m = int(n_obs[i, k])
            if m != seen:  # 새 점이 들어왔을 때만 다시 추론
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                probs = model(x[:, :m], last_only=True).softmax(-1)  # 앞 m개 점만 입력
                top = probs.max(-1)
                conf, pred = top.values.item(), top.indices.item()   # .item()이 GPU 완료를 기다린다
                rec["latency"][row] += time.perf_counter() - t0
                rec["points_recomputed"][row] += m
                rec["n_forward"][row] += 1
                seen = m
            if conf >= threshold or k == last:
                break
        rec["stop_index"][row], rec["pred"][row], rec["confidence"][row] = k, pred, conf
        rec["points_seen"][row] = seen

        # 참고용: 조기 정지 없이 전체 스케치를 한 번만 추론할 때의 시간
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x, last_only=True).softmax(-1).max(-1).values.item()
        rec["full_latency"][row] = time.perf_counter() - t0
    return rec


def summarize(rec: dict, labels: np.ndarray, n_points: np.ndarray, fractions, threshold: float,
              classes: list[str], per_token: int, head: int) -> dict:
    """스케치별 기록 → 팀 공통 지표."""
    fractions = np.asarray(fractions)
    correct = rec["pred"] == labels
    stop_fraction = fractions[rec["stop_index"]]
    flops_recompute = per_token * rec["points_recomputed"] + head * rec["n_forward"]
    flops_streaming = per_token * rec["points_seen"] + head * rec["n_forward"]
    latency_ms = rec["latency"] * 1e3

    result = {
        "n_sketches": int(len(labels)),
        "threshold": threshold,
        "stopping_accuracy": float(correct.mean()),
        "mean_earliness": float(stop_fraction.mean()),
        "cumulative_flops_recompute_mean": float(flops_recompute.mean()),
        "cumulative_flops_streaming_mean": float(flops_streaming.mean()),
        "cumulative_latency_ms": {
            "mean": float(latency_ms.mean()),
            "median": float(np.median(latency_ms)),
            "p95": float(np.percentile(latency_ms, 95)),
        },
        "mean_forward_calls": float(rec["n_forward"].mean()),
        "stopped_by_threshold_rate": float((rec["confidence"] >= threshold).mean()),
        "full_sketch_single_pass": {  # 조기 정지를 안 쓸 때의 비용 (정확도는 accuracy_by_fraction의 100%)
            "flops_mean": float(per_token * n_points.mean() + head),
            "latency_ms_mean": float(rec["full_latency"].mean() * 1e3),
        },
        "flops_per_point": per_token,
        "head_flops": head,
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
    es, info = r["early_stopping"], r["info"]
    lat = es["cumulative_latency_ms"]
    full = es["full_sketch_single_pass"]
    lines = [
        f"# {info['run_name']}: {info['split']} 평가 결과",
        "",
        "## 조건",
        f"- 데이터: 공식 Sketch-RNN QuickDraw 분할, {info['n_classes']}개 클래스, "
        f"{info['split']} {info['n_split']:,}개 중 {es['n_sketches']:,}개 평가",
        f"- 모델: {info['model']} (파라미터 {info['n_params']:,}개), 체크포인트 epoch {info['checkpoint_epoch']}",
        f"- 평가: batch size 1, {info['gpu']}, fp32, threshold {es['threshold']}, "
        f"관찰 시점 {len(r['fractions'])}개 ({r['fractions'][0]:.0%}~{r['fractions'][-1]:.0%})",
    ]
    if info.get("train"):
        t = info["train"]
        lines.append(f"- 학습: train {t['n_train']:,}개, {t['epochs_done']} epoch, "
                     f"{t['train_hours']:.2f}시간 ({t['gpu']}), best epoch {t['best_epoch']}")
    lines += [
        "",
        "## 조기 예측 지표",
        "| 지표 | 값 |",
        "|---|---|",
        f"| Stopping Accuracy | {es['stopping_accuracy']:.2%} |",
        f"| Mean Earliness | {es['mean_earliness']:.1%} |",
        f"| Cumulative FLOPs (A: 시점마다 처음부터 재추론) | {es['cumulative_flops_recompute_mean'] / 1e6:.1f} M |",
        f"| Cumulative FLOPs (B: 상태 유지, 새 점만 처리) | {es['cumulative_flops_streaming_mean'] / 1e6:.1f} M |",
        f"| Cumulative Latency (A, 실측) | 평균 {lat['mean']:.2f} ms / 중앙값 {lat['median']:.2f} ms / p95 {lat['p95']:.2f} ms |",
        f"| 정지까지 평균 추론 횟수 | {es['mean_forward_calls']:.2f} |",
        f"| threshold를 넘겨서 멈춘 비율 | {es['stopped_by_threshold_rate']:.1%} |",
        "",
        "참고 (조기 정지 없이 전체 스케치를 한 번 추론): "
        f"정확도 {r['accuracy_by_fraction'][-1]:.2%}, "
        f"FLOPs {full['flops_mean'] / 1e6:.1f} M, 지연 {full['latency_ms_mean']:.2f} ms",
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
    for d in es["stop_distribution"]:
        acc = "-" if d["accuracy"] is None else f"{d['accuracy']:.1%}"
        lines.append(f"| {d['fraction']:.0%} | {d['share']:.1%} | {acc} |")
    lines += ["", "## 클래스별", "| 클래스 | 개수 | Stopping Accuracy | Mean Earliness |", "|---|---|---|---|"]
    for c in es["per_class"]:
        lines.append(f"| {c['class']} | {c['n']} | {c['stopping_accuracy']:.1%} | {c['mean_earliness']:.1%} |")
    lines += [
        "",
        "## 측정 가정",
        "- 관찰 비율 f 시점에는 앞에서부터 ceil(점 개수 × f)개(최소 1개)의 점을 본다.",
        "- FLOPs는 곱셈-덧셈 1회를 1로 세고(fvcore 기준), 활성화·정규화는 세지 않는다. "
        f"점 하나당 {es['flops_per_point']:,} FLOPs.",
        "- (A)는 관찰 시점마다 prefix 전체를 다시 추론한 값이고, 지연 시간은 (A)로만 실측했다.",
        "- 지연 시간은 forward + softmax + 결과를 CPU로 가져오는 시간이며 전처리는 제외했다.",
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
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

    curve = accuracy_by_fraction(model, split, ec["fractions"], cfg["train"]["eval_batch_size"])
    t0 = time.time()
    rec = early_stopping_eval(model, split, indices, ec["fractions"], ec["threshold"], ec["warmup_iters"])
    eval_seconds = time.time() - t0

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
            "eval_seconds": round(eval_seconds, 1),
        },
        "fractions": list(ec["fractions"]),
        "accuracy_by_fraction": curve,
        "early_stopping": summarize(
            rec,
            labels=split.labels[indices].cpu().numpy(),
            n_points=split.lengths[indices].cpu().numpy(),
            fractions=ec["fractions"], threshold=ec["threshold"], classes=meta["classes"],
            per_token=flops_per_token(model), head=head_flops(model),
        ),
    }
    summary_path = out / "train_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            result["info"]["train"] = json.load(f)

    suffix = f"_limit{args.limit}" if args.limit else ""
    json_path = out / f"eval_{split_name}{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    markdown = to_markdown(result)
    Path(out / f"eval_{split_name}{suffix}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
