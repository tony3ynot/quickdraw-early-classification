"""학습된 모델의 "관찰 시점별 출력"을 파일로 저장한다.

정지 정책(고정 threshold, Deep Optimal Stopping, AGC 등)은 분류 모델을 다시 돌리지 않고
이 파일만으로 학습·평가할 수 있다. 스케치마다 10개 관찰 시점(10%~100%)에서
  feature [K, d_model], 클래스 logits [K, C], (있으면) 정지 logit [K]
을 저장한다. causal 모델이라 전체 시퀀스를 한 번 추론하고 해당 위치만 골라내면 된다.

실행 (VM의 레포 루트에서):
  python -m qd.dump --config configs/mamba_20cls.yaml                       # valid, test
  python -m qd.dump --config configs/mamba_20cls.yaml --splits train --limit 200000

출력: ~/runs/<run_name>/obs_<split>.npz
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from qd.common import load_config, run_dir
from qd.data import SketchSplit, build_cache, load_meta, load_split, prefix_lengths
from qd.model import build_model


@dataclass
class Observations:
    """한 분할의 관찰 시점별 출력. N = 스케치 수, K = 관찰 시점 수."""

    feature: np.ndarray     # float16 [N, K, d_model]
    logits: np.ndarray      # float32 [N, K, C]
    stop_logit: np.ndarray  # float32 [N, K] (정지 head가 없는 모델이면 길이 0)
    labels: np.ndarray      # int64 [N]
    n_obs: np.ndarray       # int64 [N, K] 각 시점에 관찰한 점 개수
    fractions: np.ndarray   # float64 [K]

    @property
    def probs(self) -> np.ndarray:
        z = self.logits - self.logits.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def save(self, path: Path) -> None:
        np.savez(path, **self.__dict__)

    @staticmethod
    def load(path: Path) -> "Observations":
        with np.load(path) as f:
            return Observations(**{k: f[k] for k in f.files})

    def subset(self, idx: np.ndarray) -> "Observations":
        return Observations(self.feature[idx], self.logits[idx],
                            self.stop_logit[idx] if len(self.stop_logit) else self.stop_logit,
                            self.labels[idx], self.n_obs[idx], self.fractions)


@torch.inference_mode()
def collect(model, split: SketchSplit, fractions, batch_size: int,
            limit: int | None = None, seed: int = 0) -> Observations:
    model.eval()
    device = split.points.device
    order = torch.arange(len(split))
    if limit and limit < len(split):  # 캐시가 클래스 순서라서 무작위로 뽑는다
        order = torch.randperm(len(split), generator=torch.Generator().manual_seed(seed))[:limit].sort().values
    feats, logits_all, stops, labels_all, n_obs_all = [], [], [], [], []
    for start in range(0, len(order), batch_size):
        x, lengths, labels = split.batch(order[start:start + batch_size].to(device))
        feature, logits, stop_logit = model.readout(model.encode(x))   # [B, L, ...]
        n_obs = prefix_lengths(lengths, fractions)                     # [B, K]
        pos = n_obs - 1
        feats.append(feature.gather(1, pos.unsqueeze(-1).expand(-1, -1, feature.shape[-1])).half().cpu())
        logits_all.append(logits.gather(1, pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1])).float().cpu())
        if stop_logit is not None:
            stops.append(stop_logit.gather(1, pos).float().cpu())
        labels_all.append(labels.cpu())
        n_obs_all.append(n_obs.cpu())
    return Observations(
        feature=torch.cat(feats).numpy(),
        logits=torch.cat(logits_all).numpy(),
        stop_logit=torch.cat(stops).numpy() if stops else np.zeros(0, dtype=np.float32),
        labels=torch.cat(labels_all).numpy(),
        n_obs=torch.cat(n_obs_all).numpy(),
        fractions=np.asarray(fractions, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="관찰 시점별 모델 출력 저장")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None, help="기본값: ~/runs/<run_name>/best.pt")
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--limit", type=int, default=None, help="분할에서 무작위로 이 개수만 저장")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    out = run_dir(cfg)
    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint or out / "best.pt", map_location="cpu", weights_only=True)
    cache = build_cache(ckpt["config"]["data"])
    model = build_model(ckpt["config"]["model"], n_classes=len(load_meta(cache)["classes"])).to(device)
    model.load_state_dict(ckpt["model"])

    for name in args.splits:
        split = load_split(cache, name).to(device)
        obs = collect(model, split, cfg["eval"]["fractions"], cfg["train"]["eval_batch_size"],
                      limit=args.limit, seed=cfg["seed"])
        obs.save(out / f"obs_{name}.npz")
        acc = (obs.logits.argmax(-1) == obs.labels[:, None]).mean(0)
        print(f"{name}: {len(obs.labels):,}개 저장 → {out / f'obs_{name}.npz'}")
        print("  관찰 비율별 정확도: " + "  ".join(f"{a:.3f}" for a in acc))
        del split


if __name__ == "__main__":
    main()
