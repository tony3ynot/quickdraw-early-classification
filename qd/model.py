"""2-layer Mamba 스케치 분류기.

Mamba 구현은 공식 패키지 mamba-ssm을 사용한다.
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  https://arxiv.org/abs/2312.00752 / https://github.com/state-spaces/mamba

핵심 성질: Mamba 블록은 causal(단방향)이라 위치 t의 출력은 t번째 점까지만 본다.
그래서 "앞 m개 점만 관찰했을 때의 예측"은 위치 m-1의 출력과 같고,
조기 예측에서 미래의 점 정보를 쓰지 않는다. (tests/test_causality.py에서 검증)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """pre-norm 잔차 블록: x + Mamba(LayerNorm(x))."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MambaClassifier(nn.Module):
    """점 시퀀스 [B, L, 3] → 매 시점의 클래스 logits [B, L, C]."""

    def __init__(self, n_classes: int, d_in: int = 3, d_model: int = 256, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)  # 점 하나 (Δx, Δy, pen_lift) → 임베딩
        self.blocks = nn.ModuleList(
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """last_only=True면 마지막 위치의 logits [B, C]만 계산한다 (추론용)."""
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        if last_only:
            h = h[:, -1]
        return self.head(self.norm(h))


def build_model(model_cfg: dict, n_classes: int) -> MambaClassifier:
    return MambaClassifier(n_classes=n_classes, **model_cfg)
