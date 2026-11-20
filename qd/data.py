"""QuickDraw 데이터: 다운로드, 전처리 캐시, 배치 생성.

데이터 출처: 공식 Sketch-RNN QuickDraw 데이터셋
  https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset
클래스마다 `<class>.npz` 파일이 하나씩 있고, 그 안에 공식 분할
(train 70,000 / valid 2,500 / test 2,500)이 이미 들어 있다. 분할을 여기서 새로 만들지 않는다.

스케치 하나는 stroke-3 형식의 [점 개수, 3] int16 배열이다.
  (Δx, Δy, pen_lift): 직전 점으로부터의 이동량, 그리고 이 점에서 펜을 떼면 1

실행 (VM의 레포 루트에서):
  python -m qd.data --config configs/mamba_20cls.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from qd.common import load_config

RAW_URL = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{name}.npz"
SPLITS = ("train", "valid", "test")


# --------------------------------------------------------------------------- #
# 1. 다운로드와 전처리 캐시
# --------------------------------------------------------------------------- #
def download_raw(classes: Sequence[str], raw_dir: Path) -> None:
    """없는 클래스 파일만 공식 버킷에서 내려받는다."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in classes:
        path = raw_dir / f"{name}.npz"
        if path.exists():
            continue
        url = RAW_URL.format(name=urllib.parse.quote(name))
        print(f"downloading {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)


def cache_dir_for(data_cfg: dict) -> Path:
    """클래스 목록과 max_len이 같으면 같은 캐시를 재사용하도록 이름을 정한다."""
    classes = data_cfg["classes"]
    digest = hashlib.sha1(",".join(classes).encode()).hexdigest()[:8]
    return Path(data_cfg["cache_dir"]) / f"{len(classes)}cls_len{data_cfg['max_len']}_{digest}"


def _pack(groups: list, max_len: int):
    """가변 길이 스케치들을 0으로 패딩한 [N, T, 3] 배열 하나로 묶는다."""
    n_total = sum(len(sketches) for sketches, _ in groups)
    longest = max(len(s) for sketches, _ in groups for s in sketches)
    pad_len = min(max_len, longest)
    points = np.zeros((n_total, pad_len, 3), dtype=np.int16)
    lengths = np.zeros(n_total, dtype=np.int16)
    labels = np.zeros(n_total, dtype=np.int16)
    n_truncated = 0
    i = 0
    for sketches, label in groups:
        for sketch in sketches:
            n = min(len(sketch), pad_len)
            n_truncated += len(sketch) > pad_len
            points[i, :n] = sketch[:n]
            lengths[i] = n
            labels[i] = label
            i += 1
    return points, lengths, labels, int(n_truncated)


def build_cache(data_cfg: dict) -> Path:
    """원본 .npz → 학습에 바로 쓸 수 있는 배열 캐시. 이미 있으면 그대로 쓴다."""
    out = cache_dir_for(data_cfg)
    if (out / "meta.json").exists():
        return out

    classes = list(data_cfg["classes"])
    raw_dir = Path(data_cfg["raw_dir"])
    max_len = int(data_cfg["max_len"])
    download_raw(classes, raw_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = {split: [] for split in SPLITS}  # split -> [(스케치 배열들, 라벨)]
    counts, length_stats = {}, {}
    for label, name in enumerate(classes):
        with np.load(raw_dir / f"{name}.npz", encoding="latin1", allow_pickle=True) as raw:
            counts[name] = {}
            for split in SPLITS:
                sketches = raw[split]
                groups[split].append((sketches, label))
                counts[name][split] = len(sketches)
                if split == "train":
                    n_points = np.array([len(s) for s in sketches])
                    length_stats[name] = {
                        "min": int(n_points.min()),
                        "mean": round(float(n_points.mean()), 1),
                        "max": int(n_points.max()),
                    }

    meta = {
        "classes": classes,
        "max_len": max_len,
        "counts": counts,
        "train_points_per_sketch": length_stats,
        "truncated": {},
        "pad_len": {},
    }
    for split in SPLITS:
        points, lengths, labels, n_truncated = _pack(groups[split], max_len)
        np.save(out / f"{split}_points.npy", points)
        np.save(out / f"{split}_lengths.npy", lengths)
        np.save(out / f"{split}_labels.npy", labels)
        meta["truncated"][split] = n_truncated
        meta["pad_len"][split] = int(points.shape[1])
        if split == "train":
            # 좌표 정규화 상수: train의 모든 (Δx, Δy) 표준편차 (Sketch-RNN과 같은 방식).
            # 데이터셋 전체에 대한 상수 하나이므로 개별 스케치의 미래 점 정보를 쓰지 않는다.
            valid_mask = np.arange(points.shape[1])[None, :] < lengths[:, None]
            meta["scale"] = float(points[..., :2][valid_mask].std(dtype=np.float64))

    # meta.json을 마지막에 써서, 이 파일이 있으면 캐시가 완성된 것으로 본다.
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out


def load_meta(cache_dir: Path) -> dict:
    with open(cache_dir / "meta.json", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 2. 메모리(GPU)에 올려 두고 쓰는 데이터 분할
# --------------------------------------------------------------------------- #
@dataclass
class SketchSplit:
    """한 분할(train/valid/test) 전체. 텐서를 통째로 GPU에 올려 인덱싱만으로 배치를 만든다.

    데이터가 작아서(train 140만 개 ≈ 1.5GB) DataLoader 워커 없이도 충분히 빠르다.
    """

    points: torch.Tensor   # int16 [N, T, 3], 뒤쪽은 0으로 패딩
    lengths: torch.Tensor  # int64 [N], 스케치별 실제 점 개수
    labels: torch.Tensor   # int64 [N]
    scale: float           # (Δx, Δy)를 나누는 정규화 상수

    def __len__(self) -> int:
        return self.points.shape[0]

    def to(self, device) -> "SketchSplit":
        return SketchSplit(self.points.to(device), self.lengths.to(device),
                           self.labels.to(device), self.scale)

    def batch(self, idx: torch.Tensor):
        """인덱스 → (모델 입력 x [B, L, 3] float, lengths [B], labels [B]).

        L은 이 배치에서 가장 긴 스케치의 길이. 패딩은 항상 뒤에 붙으므로
        causal 모델에서는 실제 점 위치의 출력에 영향을 주지 않는다.
        """
        lengths = self.lengths[idx]
        x = self.points[idx, : int(lengths.max())].float()
        x[..., :2] /= self.scale
        return x, lengths, self.labels[idx]

    def batches(self, batch_size: int, shuffle: bool = False,
                generator: torch.Generator | None = None,
                drop_last: bool = False) -> Iterator:
        n = len(self)
        if shuffle:
            order = torch.randperm(n, generator=generator).to(self.points.device)
        else:
            order = torch.arange(n, device=self.points.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if drop_last and len(idx) < batch_size:
                break
            yield self.batch(idx)


def load_split(cache_dir: Path, split: str) -> SketchSplit:
    meta = load_meta(cache_dir)
    return SketchSplit(
        points=torch.from_numpy(np.load(cache_dir / f"{split}_points.npy")),
        lengths=torch.from_numpy(np.load(cache_dir / f"{split}_lengths.npy")).long(),
        labels=torch.from_numpy(np.load(cache_dir / f"{split}_labels.npy")).long(),
        scale=meta["scale"],
    )


# --------------------------------------------------------------------------- #
# 3. 관찰 비율 → 관찰한 점 개수
# --------------------------------------------------------------------------- #
def prefix_lengths(lengths: torch.Tensor, fractions: Sequence[float]) -> torch.Tensor:
    """스케치 길이 [N] → 각 관찰 비율에서 본 점 개수 [N, K].

    전체 점 개수 n의 비율 f 시점에는 앞에서부터 ceil(n * f)개(최소 1개)를 본 것으로 정의한다.
    팀 공통 평가 정의이므로 모든 모델이 이 함수를 같이 써야 한다.
    """
    f = torch.tensor(list(fractions), dtype=torch.float64, device=lengths.device)
    # 1e-9: 40 * 0.1 = 4.000000001 같은 부동소수점 오차로 한 점 더 보는 것을 막는다.
    k = torch.ceil(lengths.double().unsqueeze(1) * f - 1e-9).long()
    return torch.minimum(k.clamp(min=1), lengths.unsqueeze(1))


# --------------------------------------------------------------------------- #
# CLI: 데이터 준비 + 클래스별 개수 출력
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="QuickDraw 데이터 다운로드 및 캐시 생성")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache = build_cache(cfg["data"])
    meta = load_meta(cache)

    print(f"cache: {cache}")
    print(f"정규화 상수(scale): {meta['scale']:.4f}   잘린 스케치 수: {meta['truncated']}")
    print(f"{'class':12s} {'train':>8s} {'valid':>7s} {'test':>7s}   점 개수(train) min/mean/max")
    totals = {split: 0 for split in SPLITS}
    for name in meta["classes"]:
        c, s = meta["counts"][name], meta["train_points_per_sketch"][name]
        for split in SPLITS:
            totals[split] += c[split]
        print(f"{name:12s} {c['train']:8d} {c['valid']:7d} {c['test']:7d}   "
              f"{s['min']} / {s['mean']} / {s['max']}")
    print(f"{'total':12s} {totals['train']:8d} {totals['valid']:7d} {totals['test']:7d}")


if __name__ == "__main__":
    main()
