"""Mamba 분류기의 추론 연산량(FLOPs)을 수식으로 계산한다.

fvcore/thop 같은 도구는 Mamba의 selective scan(커스텀 CUDA 연산)을 세지 못해 값이 작게 나온다.
그래서 직접 센다. 세는 기준은 fvcore와 같다.
  - 곱셈-덧셈 1회 = 1 FLOP (Linear: in*out, Conv1d: (in/groups)*kernel*out)
  - 활성화 함수, 정규화, exp/softplus 같은 원소별 연산은 세지 않는다.
  - selective scan은 토큰 하나, 채널 하나, 상태 하나마다 곱셈 4회로 센다:
      exp(Δ·A)의 Δ·A, Δ·B·x, 상태 갱신 h = Ā·h + B̄x, 출력 y += C·h
    여기에 채널마다 D·x skip과 gate 곱(y·silu(z)) 2회를 더한다.

모든 층이 토큰(점)마다 같은 연산을 하므로 FLOPs는 관찰한 점 개수에 정확히 비례한다.
"""
from __future__ import annotations

import torch.nn as nn
from mamba_ssm import Mamba


def flops_per_token(model: nn.Module) -> int:
    """점 하나를 처리하는 데 드는 FLOPs (분류 head 제외)."""
    total = 0
    for module in model.modules():
        if module is model.head or module is getattr(model, "stop_head", None):
            continue  # 출력 head는 점마다가 아니라 예측을 읽을 때만 계산한다
        if isinstance(module, nn.Linear):
            total += module.in_features * module.out_features
        elif isinstance(module, nn.Conv1d):
            total += (module.in_channels // module.groups) * module.kernel_size[0] * module.out_channels
        elif isinstance(module, Mamba):
            total += 4 * module.d_inner * module.d_state + 2 * module.d_inner
    return total


def head_flops(model: nn.Module) -> int:
    """예측을 한 번 읽는 비용: 분류 head (+ 있으면 정지 head). 추론에서는 마지막 위치에만 적용."""
    total = model.head.in_features * model.head.out_features
    stop_head = getattr(model, "stop_head", None)
    if stop_head is not None:
        total += stop_head.in_features * stop_head.out_features
    return total


def forward_flops(model: nn.Module, n_points: int) -> int:
    """점 n_points개짜리 prefix를 처음부터 한 번 추론하는 FLOPs."""
    return n_points * flops_per_token(model) + head_flops(model)
