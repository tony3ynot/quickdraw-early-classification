#!/usr/bin/env bash
# ELECTS 학습(scripts/run_elects.sh)이 끝난 뒤의 분석. 저장된 obs_*.npz만 쓰므로 몇 분이면 끝난다.
#   1) ELECTS 모델 4개 각각: 고정 threshold와 정지 head를 baseline과 비교 (stop_report)
#   2) baseline에 대한 Deep Optimal Stopping을 촘촘한 cost 간격으로 다시 실행
#
#   bash scripts/post_elects.sh
set -uo pipefail

cd "$(dirname "$0")/.."
source ~/venvs/qd/bin/activate

for NAME in mamba2_20cls_elects mamba2_20cls_elects_ft_a0.3 mamba2_20cls_elects_ft_a0.5 mamba2_20cls_elects_ft_a0.7; do
  python -m qd.stop_report --config configs/mamba_20cls_elects.yaml --run-name "$NAME" --reference mamba2_20cls
done
python -m qd.optimal_stopping --config configs/mamba_20cls.yaml
