#!/usr/bin/env bash
# 후속 실험 (B0~B4)을 Mamba 모델 세 개에 실행한다.
#   1) 각 모델의 train 전체 예측을 logits만 저장 (정지 점수 head 학습용)
#   2) valid/test 예측이 없으면 저장
#   3) B0~B4 비교 실행
# 분류 모델은 재학습하지 않는다. 모델당 5분 안팎.
#
#   bash scripts/run_stop_score_study_mamba.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/venvs/qd/bin/activate

run_one() {  # 인자: 모델 설정 파일, run 이름, 실험 설정 파일
  local model_config=$1 run_name=$2 study_config=$3
  [ -f ~/runs/"$run_name"/obs_train_logits.npz ] || python -m qd.dump --config "$model_config" --splits train --logits-only
  [ -f ~/runs/"$run_name"/obs_valid.npz ] || python -m qd.dump --config "$model_config"
  python -m qd.stop_scores --config "$study_config"
}

run_one configs/mamba_20cls.yaml           mamba2_20cls           configs/stop_score_study_mamba_baseline.yaml
run_one configs/mamba_20cls_elects.yaml    mamba2_20cls_elects    configs/stop_score_study_mamba_elects.yaml
run_one configs/mamba_20cls_fixedstop.yaml mamba2_20cls_fixedstop configs/stop_score_study_mamba_fixedweight.yaml
