"""2-layer Mamba 스케치 분류기.

Mamba 구현은 공식 패키지 mamba-ssm을 사용한다.
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  https://arxiv.org/abs/2312.00752 / https://github.com/state-spaces/mamba

핵심 성질: Mamba 블록은 causal(단방향)이라 위치 t의 출력은 t번째 점까지만 본다.
그래서 "앞 m개 점만 관찰했을 때의 예측"은 위치 m-1의 출력과 같고,
조기 예측에서 미래의 점 정보를 쓰지 않는다. (tests/test_causality.py에서 검증)
"""
from __future__ import annotations

import mamba_ssm.modules.mamba_simple as _mamba_simple
import torch
import torch.nn as nn
from mamba_ssm import Mamba

# Mamba.step()은 기본적으로 triton 커널을 쓰는데, 이 커널은 처음 실행할 때 C 컴파일이 필요해
# Python 개발 헤더(python3-dev)가 없는 환경에서는 실패한다. None으로 두면 mamba-ssm이 같은 계산을
# 하는 순수 PyTorch 경로를 쓴다. 학습과 일반 추론에 쓰는 forward()에는 영향이 없다.
_mamba_simple.selective_state_update = None


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

    # ---- 스트리밍 추론: 상태를 유지하면서 새로 들어온 점만 처리한다 ----
    # forward()로 prefix를 처음부터 다시 계산한 것과 같은 결과를 내지만 (tests/test_causality.py),
    # 이미 본 점을 다시 계산하지 않는다. mamba-ssm의 step() API를 그대로 쓴다.

    def init_stream_state(self, batch_size: int = 1) -> list:
        """스케치를 시작할 때의 빈 상태. 블록마다 (conv 상태, SSM 상태) 한 쌍."""
        return [block.mixer.allocate_inference_cache(batch_size, max_seqlen=None)
                for block in self.blocks]

    def step(self, point: torch.Tensor, state: list) -> torch.Tensor:
        """점 하나 [B, 3]을 처리한다. state는 제자리에서 갱신되고, 이 시점의 feature [B, d_model]을 돌려준다."""
        h = self.embed(point).unsqueeze(1)  # [B, 1, d_model]
        for block, (conv_state, ssm_state) in zip(self.blocks, state):
            out, _, _ = block.mixer.step(block.norm(h), conv_state, ssm_state)
            h = h + out
        return h.squeeze(1)

    def classify(self, feature: torch.Tensor) -> torch.Tensor:
        """step()이 돌려준 feature → 클래스 logits [B, C]."""
        return self.head(self.norm(feature))


def build_model(model_cfg: dict, n_classes: int) -> MambaClassifier:
    return MambaClassifier(n_classes=n_classes, **model_cfg)
