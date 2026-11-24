#!/usr/bin/env bash
# ELECTS 실험을 순서대로 실행한다. 중간에 끊겨도 같은 명령으로 이어서 진행된다.
#   1) 보고용 본 학습: baseline과 같은 조건(30 epoch, 처음부터)에 ELECTS loss만 추가, alpha 0.5
#   2) alpha 민감도 확인: baseline 체크포인트에서 5 epoch 미세조정 (alpha 0.3 / 0.5 / 0.7)
# 각 학습이 끝나면 관찰 시점별 출력(obs_*.npz)을 저장해 정지 정책을 재학습 없이 분석할 수 있게 한다.
#
#   tmux new -s train
#   bash scripts/run_elects.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/venvs/qd/bin/activate
CONFIG=configs/mamba_20cls_elects.yaml
BASELINE=~/runs/mamba2_20cls/best.pt

python -m qd.train --config "$CONFIG" --resume
python -m qd.dump --config "$CONFIG"

for ALPHA in 0.3 0.5 0.7; do
  NAME="mamba2_20cls_elects_ft_a${ALPHA}"
  python -m qd.train --config "$CONFIG" --run-name "$NAME" --init "$BASELINE" \
    --epochs 5 --lr 3e-4 --alpha "$ALPHA" --resume
  python -m qd.dump --config "$CONFIG" --run-name "$NAME"
done
