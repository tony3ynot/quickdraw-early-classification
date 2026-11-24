"""학습 스크립트.

실행 (VM의 레포 루트에서, 가상환경 활성화 후):
  python -m qd.train --config configs/mamba_20cls.yaml            # 처음부터
  python -m qd.train --config configs/mamba_20cls.yaml --resume   # 중단된 곳부터 이어서

학습 방식: 스케치의 모든 시점에 분류 loss를 건다 (dense per-step cross-entropy).
모델이 causal이므로 위치 t의 loss는 "앞 t+1개 점만 보고 맞히기"를 학습시키는 것과 같다.
마지막 시점에만 loss를 걸면 그리다 만 스케치에 대한 예측이 거의 학습되지 않는다
(SketchMamba, arXiv 2607.23580의 ablation 참고).

출력 (~/runs/<run_name>/):
  train.log           진행 로그
  history.json        epoch별 검증 결과와 소요 시간
  best.pt / last.pt   체크포인트 (best 기준: valid에서 10개 관찰 시점 정확도의 평균)
  train_summary.json  총 학습 시간, 파라미터 수 등 요약
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from qd.common import Logger, count_parameters, load_config, run_dir, set_seed
from qd.data import SketchSplit, build_cache, load_meta, load_split, prefix_lengths
from qd.evaluate import fraction_predictions
from qd.model import build_model


def dense_cross_entropy(logits: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor,
                        reduction: str = "mean") -> torch.Tensor:
    """패딩이 아닌 모든 시점에 대한 cross-entropy. logits [B, L, C]."""
    batch, seq_len, _ = logits.shape
    is_real = torch.arange(seq_len, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    targets = labels.unsqueeze(1).expand(batch, seq_len)
    return F.cross_entropy(logits[is_real].float(), targets[is_real], reduction=reduction)


def elects_loss(logits: torch.Tensor, stop_logit: torch.Tensor, lengths: torch.Tensor,
                labels: torch.Tensor, fractions, alpha: float,
                fixed_stop_weights=None) -> torch.Tensor:
    """ELECTS loss (Rußwurm et al., arXiv 1901.10681)를 10개 관찰 시점에 적용한 것.

    관찰 시점 k마다 모델이 정지 확률 δ_k를 낸다 (마지막 시점은 δ = 1).
      P(k에서 멈춤) = δ_k · Π_{j<k} (1 − δ_j)
      loss = Σ_k P(k에서 멈춤) · [ α · CE_k − (1 − α) · p_k(정답) · (1 − 관찰 비율_k) ]
    첫 항은 멈추는 시점의 분류 loss, 둘째 항은 일찍 맞힐수록 큰 보상이다.

    원 논문과 다른 점: 원 논문은 모든 시점에 ε/T를 더해 초반 붕괴를 막는데,
    여기서는 그 역할을 baseline의 dense cross-entropy가 맡는다 (main에서 두 loss를 더한다).

    fixed_stop_weights (길이 K)를 주면 정지 head를 쓰지 않고 P(k에서 멈춤)을 이 값으로 고정한다.
    "정지를 함께 학습한 효과"와 "관찰 시점에 loss 가중치를 더 준 효과"를 가르기 위한 비교 실험용이다.
    """
    pos = prefix_lengths(lengths, fractions) - 1                                  # [B, K]
    obs_logits = logits.float().gather(1, pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
    if fixed_stop_weights is not None:
        p_stop = torch.tensor(list(fixed_stop_weights), device=logits.device,
                              dtype=torch.float32).unsqueeze(0)                    # [1, K]
    else:
        delta = torch.sigmoid(stop_logit.float().gather(1, pos))
        delta = torch.cat([delta[:, :-1], torch.ones_like(delta[:, :1])], dim=1)   # 마지막은 강제 정지
        still_going = torch.cumprod(1 - delta[:, :-1], dim=1)
        still_going = torch.cat([torch.ones_like(delta[:, :1]), still_going], dim=1)  # Π_{j<k} (1 − δ_j)
        p_stop = delta * still_going                                               # [B, K], 합이 1

    log_p = F.log_softmax(obs_logits, dim=-1)
    ce = -log_p.gather(-1, labels.view(-1, 1, 1).expand(-1, log_p.shape[1], 1)).squeeze(-1)
    frac = torch.tensor(list(fractions), device=logits.device, dtype=torch.float32)
    earliness_reward = torch.exp(-ce) * (1 - frac)
    return (p_stop * (alpha * ce - (1 - alpha) * earliness_reward)).sum(1).mean()


@torch.no_grad()
def validate(model, split: SketchSplit, fractions, batch_size: int, amp: bool) -> dict:
    """검증 loss와, 10%~100% 각 시점에서 예측했을 때의 정확도(정지 정책 없이).

    정지 head가 있는 모델이면 "정지 확률 ≥ 0.5에서 멈춤" 규칙의 정확도와 조기성도 함께 낸다.
    """
    model.eval()
    loss_sum, n_points = 0.0, 0
    correct = torch.zeros(len(fractions), device=split.points.device)
    frac = torch.tensor(list(fractions), device=split.points.device)
    stop_correct, stop_fraction = 0.0, 0.0
    for x, lengths, labels in split.batches(batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _, logits, stop_logit = model.readout(model.encode(x))
        loss_sum += dense_cross_entropy(logits, lengths, labels, reduction="sum").item()
        n_points += int(lengths.sum())
        preds = fraction_predictions(logits, lengths, fractions)
        correct += (preds == labels.unsqueeze(1)).sum(0)
        if stop_logit is not None:
            stop = stop_logit.float().gather(1, prefix_lengths(lengths, fractions) - 1) >= 0
            stop[:, -1] = True
            stop_index = stop.float().argmax(1)  # 처음 멈추는 시점
            stop_correct += (preds.gather(1, stop_index.unsqueeze(1)).squeeze(1) == labels).sum().item()
            stop_fraction += frac[stop_index].sum().item()
    acc = (correct / len(split)).tolist()
    result = {
        "loss": loss_sum / n_points,
        "acc_by_fraction": [round(a, 4) for a in acc],
        "mean_acc": sum(acc) / len(acc),
        "final_acc": acc[-1],
    }
    if model.stop_head is not None:
        result["stop_accuracy"] = stop_correct / len(split)
        result["stop_earliness"] = stop_fraction / len(split)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="QuickDraw Mamba 분류기 학습")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="last.pt가 있으면 이어서 학습")
    parser.add_argument("--max-steps", type=int, default=None, help="동작 확인용: 이 step 수만 돌고 종료")
    parser.add_argument("--run-name", default=None, help="설정 파일의 run_name 덮어쓰기")
    parser.add_argument("--init", default=None, help="이 체크포인트의 가중치에서 시작 (미세조정)")
    parser.add_argument("--epochs", type=int, default=None, help="설정 파일의 epochs 덮어쓰기")
    parser.add_argument("--lr", type=float, default=None, help="설정 파일의 lr 덮어쓰기")
    parser.add_argument("--alpha", type=float, default=None, help="설정 파일의 ELECTS alpha 덮어쓰기")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.alpha is not None:
        cfg["train"]["elects"]["alpha"] = args.alpha
    tc = cfg["train"]
    elects = tc.get("elects")  # None이면 baseline (dense cross-entropy만)
    out = run_dir(cfg)
    log = Logger(out / "train.log")
    set_seed(cfg["seed"])
    device = torch.device("cuda")

    # ---- 데이터: 전체를 GPU에 올린다 ----
    cache = build_cache(cfg["data"])
    meta = load_meta(cache)
    train = load_split(cache, "train").to(device)
    valid = load_split(cache, "valid").to(device)

    # ---- 모델, 옵티마이저, 스케줄 ----
    model = build_model(cfg["model"], n_classes=len(meta["classes"])).to(device)
    if args.init:
        init = torch.load(args.init, map_location=device, weights_only=True)["model"]
        missing = model.load_state_dict(init, strict=False).missing_keys  # 정지 head는 새로 초기화
        log(f"init: {args.init} 에서 가중치를 불러옴 (새로 초기화된 파라미터: {missing or '없음'})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = len(train) // tc["batch_size"]
    total_steps = steps_per_epoch * tc["epochs"]

    def lr_scale(step: int) -> float:
        """선형 warmup 후 cosine 감소."""
        if step < tc["warmup_steps"]:
            return (step + 1) / tc["warmup_steps"]
        progress = (step - tc["warmup_steps"]) / max(1, total_steps - tc["warmup_steps"])
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    state = {"epoch": 0, "step": 0, "best": -1.0, "best_epoch": 0,
             "train_seconds": 0.0, "val_seconds": 0.0, "history": []}
    last_path = out / "last.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        log(f"resume: epoch {state['epoch']} / step {state['step']} 부터 이어서 학습")

    n_params = count_parameters(model)
    log(f"run={cfg['run_name']}  params={n_params:,}  gpu={torch.cuda.get_device_name(0)}")
    log(f"train={len(train):,}  valid={len(valid):,}  classes={len(meta['classes'])}  "
        f"batch={tc['batch_size']}  steps/epoch={steps_per_epoch}  epochs={tc['epochs']}")

    shuffle_rng = torch.Generator().manual_seed(cfg["seed"] + state["epoch"])
    stop_early = False

    for epoch in range(state["epoch"], tc["epochs"]):
        model.train()
        t_epoch = time.time()
        loss_window, t_window = 0.0, time.time()
        for i, (x, lengths, labels) in enumerate(
                train.batches(tc["batch_size"], shuffle=True, generator=shuffle_rng, drop_last=True)):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc["amp"]):
                _, logits, stop_logit = model.readout(model.encode(x))
            loss = dense_cross_entropy(logits, lengths, labels)
            if elects:
                loss = loss + elects_loss(logits, stop_logit, lengths, labels,
                                          cfg["eval"]["fractions"], elects["alpha"],
                                          elects.get("fixed_stop_weights"))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()
            scheduler.step()
            state["step"] += 1

            loss_window += loss.item()
            if (i + 1) % tc["log_every"] == 0:
                speed = tc["log_every"] / (time.time() - t_window)
                log(f"epoch {epoch + 1} step {i + 1}/{steps_per_epoch}  "
                    f"loss {loss_window / tc['log_every']:.4f}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}  {speed:.1f} it/s")
                loss_window, t_window = 0.0, time.time()
            if args.max_steps and state["step"] >= args.max_steps:
                stop_early = True
                break
        train_seconds = time.time() - t_epoch
        state["train_seconds"] += train_seconds

        # ---- epoch 끝: 검증 + 체크포인트 ----
        t_val = time.time()
        val = validate(model, valid, cfg["eval"]["fractions"], tc["eval_batch_size"], tc["amp"])
        state["val_seconds"] += time.time() - t_val
        state["epoch"] = epoch + 1
        state["history"].append({"epoch": epoch + 1, "train_seconds": round(train_seconds, 1), **val})
        log(f"epoch {epoch + 1} 완료 ({train_seconds / 60:.1f}분)  val_loss {val['loss']:.4f}  "
            f"val_acc@100% {val['final_acc']:.4f}  val_acc 평균(10~100%) {val['mean_acc']:.4f}"
            + (f"  정지 head: 정확도 {val['stop_accuracy']:.4f} / 조기성 {val['stop_earliness']:.3f}"
               if "stop_accuracy" in val else ""))

        weights = {"model": model.state_dict(), "config": cfg, "classes": meta["classes"],
                   "scale": meta["scale"], "epoch": epoch + 1, "val": val}
        if val["mean_acc"] > state["best"]:
            state["best"], state["best_epoch"] = val["mean_acc"], epoch + 1
            torch.save(weights, out / "best.pt")
        torch.save({**weights, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, last_path)
        with open(out / "history.json", "w", encoding="utf-8") as f:
            json.dump(state["history"], f, indent=2)
        if stop_early:
            log("--max-steps에 도달해 종료")
            break

    summary = {
        "run_name": cfg["run_name"],
        "completed": not stop_early,
        "epochs_done": state["epoch"],
        "steps_done": state["step"],
        "train_seconds": round(state["train_seconds"], 1),
        "val_seconds": round(state["val_seconds"], 1),
        "train_hours": round(state["train_seconds"] / 3600, 3),
        "best_epoch": state["best_epoch"],
        "best_val_mean_acc": round(state["best"], 4),
        "n_params": n_params,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "n_train": len(train),
        "n_valid": len(valid),
        "torch": torch.__version__,
    }
    with open(out / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"학습 종료: 학습 시간 {summary['train_hours']:.2f}시간, "
        f"best epoch {summary['best_epoch']} (val 평균 정확도 {summary['best_val_mean_acc']:.4f})")


if __name__ == "__main__":
    main()
