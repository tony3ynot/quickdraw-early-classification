#!/usr/bin/env bash
# 학습 → test 평가를 이어서 실행한다. 중간에 끊겼으면 같은 명령으로 이어서 학습한다.
# SSH가 끊겨도 계속 돌도록 tmux 안에서 실행한다.
#
#   tmux new -s train
#   bash scripts/run_baseline.sh configs/mamba_20cls.yaml
#   (Ctrl+B 누른 뒤 D 로 빠져나오기, 다시 보기: tmux attach -t train)
set -euo pipefail

CONFIG=${1:-configs/mamba_20cls.yaml}
cd "$(dirname "$0")/.."
source ~/venvs/qd/bin/activate

python -m qd.train --config "$CONFIG" --resume
python -m qd.evaluate --config "$CONFIG"
