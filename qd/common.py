"""설정 로딩, 시드 고정, 로그 등 공통 유틸."""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """YAML 설정 파일을 읽고, `~`가 들어간 경로를 절대 경로로 바꾼다."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section, key in (("data", "raw_dir"), ("data", "cache_dir"), ("train", "out_dir")):
        cfg[section][key] = str(Path(cfg[section][key]).expanduser())
    return cfg


def run_dir(cfg: dict) -> Path:
    """이 실험의 체크포인트·로그·결과가 저장되는 디렉터리."""
    return Path(cfg["train"]["out_dir"]) / cfg["run_name"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger:
    """화면에 출력하면서 같은 내용을 파일에도 덧붙여 쓴다."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
