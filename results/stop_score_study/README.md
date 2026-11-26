# 후속 실험 B0~B4: 정지 점수를 무엇으로 학습시키는 것이 좋은가

분류 모델은 고정하고, 저장된 시점별 예측 위에서 "지금 멈춰도 되는가"를 판단하는 작은 MLP(정지 점수 head)만 학습해서 비교함. 모든 조건의 정지 threshold는 같은 AGC 설정으로 보정함. Transformer 2개와 Mamba 3개, 총 5개 분류 모델에서 같은 실험을 반복함

## 실험 조건

| 조건 | 학습 목표 | 입력 | AGC에 넣는 정지 점수 |
|---|---|---|---|
| B0 | 없음 | - | Max Softmax |
| B1 | Correctness: 지금 예측이 정답인가 | 현재 시점 | q_C |
| B2 | Correctness | 현재 + 과거 이력 | q_C |
| B3 | Harmful Risk: 지금 멈추면 완성 시 맞힐 답을 잃는가 | 현재 시점 | 1 − q_H |
| B4 | Harmful Risk | 현재 + 과거 이력 | 1 − q_H |

- 분류 모델 (모두 가중치 고정)
    - Transformer Baseline, Transformer KD: 공유받은 모델. KD는 초기 과신이 있는 모델
    - Mamba Baseline, Mamba ELECTS, Mamba Fixed-weight: Fixed-weight는 초기 과신이 있는 모델
- 현재 시점 입력: 정렬한 확률, 클래스별 확률, margin, entropy, 관찰 시점 (52차원)
- 과거 이력 입력: 직전 대비 confidence·margin 변화, 예측 클래스 변경 여부와 횟수, 같은 예측을 유지한 길이, 확률 분포 변화량 등 11개 추가 (63차원). 미래 시점 값이 섞이지 않는지 테스트로 확인함
- head: MLP (128 × 2층), BCE, 해당 분류 모델의 train 1,400,000개 예측으로 학습. seed 3개
- 데이터: head 선택 valid 10,000 / AGC screening 20,000 / AGC calibration 20,000 / 평가 test 50,000. 다섯 모델 모두 같은 분할 (`data_splits/validation_partitions.npz`)
- AGC: α = 0.03, δ = 0.01 (조건 5개에 0.05를 나눔), grid 0.001. 구현은 Transformer 쪽에서 공유받은 `qd/agc.py`를 그대로 사용

Transformer의 B0를 기존 설정(δ = 0.025)으로 보정하면 Baseline 93.552% / 51.250%, KD 93.300% / 54.604%로 기존 T1, T3 결과가 그대로 재현됨

Transformer의 valid/test 예측은 batch 1로 뽑은 것이고 Mamba는 여러 장을 한 번에 넣어 뽑은 것임. Mamba는 두 방식의 출력 차이가 1e-6 미만임을 확인했음. 두 모델의 valid/test 라벨 순서와 시점별 관찰 점 개수가 같은 것도 확인함

## 실험 결과 ①: 학습한 정지 점수는 Max Softmax보다 나은가?

B0 대비 Mean Earliness 차이(%p). 괄호 안은 Stopping Accuracy 차이(%p). AGC 인증이 정상적으로 된 seed의 평균

| 분류 모델 | B0 (정확도 / 관찰률) | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| Transformer Baseline | 93.55% / 51.25% | −1.27 (+0.01) | −1.25 (−0.02) | −1.33 (+0.06) | −1.40 (+0.03) |
| Transformer KD | 93.30% / 54.60% | −0.90 (+0.12) | −1.00 (+0.10) | −1.24 (+0.09) | −1.51 (+0.05) |
| Mamba Baseline | 93.01% / 52.41% | −1.21 (+0.05) | −0.99 (+0.05) | −1.41 (+0.01) | −1.46 (+0.00) |
| Mamba ELECTS | 93.62% / 53.06% | −1.27 (−0.17) | −1.21 (−0.12) | −1.52 (−0.05) | −1.64 (−0.01) |
| Mamba Fixed-weight | 93.20% / 51.14% | −0.74 (+0.01) | −0.69 (+0.03) | −1.06 (+0.03) | −1.07 (+0.06) |

다섯 모델 모두에서 학습한 점수를 쓰면 평균 관찰률이 0.7~1.6%p 줄어듦. 정확도는 대부분 그대로이고, Mamba ELECTS의 B1·B2에서만 0.1~0.2%p 낮아짐. 최대 누적 조건부 Harmful Gap은 모든 조건에서 3% 이내였음

같은 test 스케치로 paired bootstrap을 했을 때 Mean Earliness 차이의 95% 구간은 0을 포함하지 않았음 (예: Transformer Baseline B4 −1.51%p [−1.59, −1.43], Mamba Baseline B4 −1.55%p [−1.63, −1.47])

**소결론:** 정지 점수를 따로 학습시키면 Max Softmax보다 조금 더 일찍 멈출 수 있고, 이 결과는 모델 구조와 학습 방법이 달라도 반복됨. 다만 폭은 1%p 안팎임

## 실험 결과 ②: Correctness와 Harmful Risk 중 어느 쪽이 나은가?

Mean Earliness 차이(%p). 음수면 Harmful Risk 쪽이 더 일찍 멈춤

| 분류 모델 | B3 − B1 (현재 입력) | B4 − B2 (이력 포함) |
|---|---|---|
| Transformer Baseline | −0.06 | −0.15 |
| Transformer KD | −0.34 | −0.51 |
| Mamba Baseline | −0.20 | −0.47 |
| Mamba ELECTS | −0.25 | −0.43 |
| Mamba Fixed-weight | −0.32 | −0.38 |

10번의 비교 모두에서 Harmful Risk로 학습한 쪽이 더 일찍 멈췄음. 차이는 0.1~0.5%p로 작지만 방향은 한 번도 뒤집히지 않았음. Stopping Accuracy 차이는 ±0.1%p 안에서 엇갈렸음 (Mamba ELECTS에서는 Harmful Risk 쪽이 0.1%p 높고, Transformer KD와 Mamba Baseline에서는 0.03~0.05%p 낮음)

점수가 각 목표를 얼마나 잘 가르는지 (test 10~90% 시점 AUROC, "정답 여부 / 멈춰도 손해 없음")

| 분류 모델 | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| Transformer Baseline | 0.894 / 0.899 | 0.912 / 0.913 | 0.912 / 0.914 | 0.904 / 0.917 | 0.904 / 0.917 |
| Transformer KD | 0.891 / 0.895 | 0.911 / 0.911 | 0.911 / 0.911 | 0.902 / 0.916 | 0.902 / 0.917 |
| Mamba Baseline | 0.889 / 0.894 | 0.907 / 0.908 | 0.908 / 0.909 | 0.899 / 0.912 | 0.900 / 0.912 |
| Mamba ELECTS | 0.890 / 0.894 | 0.906 / 0.909 | 0.907 / 0.910 | 0.902 / 0.912 | 0.903 / 0.913 |
| Mamba Fixed-weight | 0.889 / 0.892 | 0.905 / 0.908 | 0.906 / 0.908 | 0.898 / 0.911 | 0.898 / 0.911 |

Harmful Risk로 학습한 점수는 "멈춰도 손해 없음"을 더 잘 가르고 "정답 여부"는 덜 가름. 학습 목표에 따라 점수의 성격이 예상대로 달라짐

**소결론:** 학습 목표를 AGC가 통제하는 위험(Harmful Gap)에 맞추면 일관되게 조금 더 일찍 멈춤. 효과는 작지만 5개 모델에서 방향이 같았음

## 실험 결과 ③: 과거 예측 이력은 도움이 되는가?

Mean Earliness 차이(%p). 음수면 이력을 넣은 쪽이 더 일찍 멈춤

| 분류 모델 | B2 − B1 (Correctness) | B4 − B3 (Harmful Risk) |
|---|---|---|
| Transformer Baseline | +0.02 | −0.07 |
| Transformer KD | −0.10 | −0.27 |
| Mamba Baseline | +0.22 | −0.05 |
| Mamba ELECTS | +0.06 | −0.12 |
| Mamba Fixed-weight | +0.05 | −0.01 |

방향이 일정하지 않고 크기도 대부분 0.1%p 안팎임. AUROC도 이력을 넣었을 때 0.001 이하로만 달라졌음

**소결론:** 과거 이력은 거의 도움이 되지 않았음. 현재 시점의 확률 분포에 이미 필요한 정보가 대부분 들어 있는 것으로 보임

## 실험 결과 ④: 과신하는 모델에서 효과가 더 큰가?

Transformer KD와 Mamba Fixed-weight는 Fixed 0.9에서 초기 과신 때문에 크게 손해를 봤던 모델임 (최대 누적 조건부 Harmful Gap 14.08%, 10.98%). 학습한 점수의 효과가 더 클 것으로 예상했음

그런데 개선 폭은 과신이 없는 모델과 비슷하거나 오히려 작았음 (KD 0.9~1.5%p, Fixed-weight 0.7~1.1%p, 나머지 1.0~1.6%p)

과신 문제 자체는 B0에서 이미 해결돼 있었음. Fixed 0.9 대신 AGC로 시점별 threshold를 정한 것만으로 Stopping Accuracy가 KD 88.29% → 93.30%, Fixed-weight 91.85% → 93.20%로 회복됨

**소결론:** 과신은 AGC의 시점별 threshold만으로 대부분 보정되고, 학습한 점수가 추가로 주는 것은 정답·오답을 가르는 능력의 소폭 향상임

## 실험 결과 ⑤: AGC 인증은 얼마나 안정적인가?

head 학습 60회(5개 모델 × 4개 조건 × seed 3개) 중 5회에서, AGC가 30% 시점의 정지를 인증하지 못해 그 시점이 "정지 금지"가 됐음

| 분류 모델 | 조건 | 인증 실패 seed의 관찰률 | 같은 모델의 B0 |
|---|---|---|---|
| Transformer Baseline | B1 | 53.45% | 51.25% |
| Mamba Baseline | B4 | 54.29% | 52.41% |
| Mamba Fixed-weight | B2 | 54.37% | 51.14% |
| Mamba Fixed-weight | B3 (2회) | 53.70%, 53.98% | 51.14% |

인증에 실패하면 B0보다 오히려 2~3%p 늦게 멈춤. 조건 간 차이(0.1~1.5%p)보다 이 효과가 더 큼. 같은 조건에서 seed만 달라도 생기기 때문에 학습 목표나 입력의 문제는 아님

**소결론:** AGC는 시점 단위로 통과/실패가 갈려서, 점수가 조금만 달라져도 결과가 크게 바뀔 수 있음. 학습한 점수를 실제로 쓰려면 seed를 여러 개 돌려 인증 결과를 확인해야 함

## 정리

1. 정지 점수를 따로 학습시키면 같은 AGC 위험 수준에서 0.7~1.6%p 더 일찍 멈춤. 5개 모델 모두에서 재현됨
2. 학습 목표를 Harmful Risk로 맞추면 Correctness보다 0.1~0.5%p 더 일찍 멈춤. 폭은 작지만 10번의 비교에서 방향이 모두 같았음
3. 과거 예측 이력은 거의 도움이 되지 않음
4. 과신 문제는 학습한 점수가 아니라 AGC의 시점별 threshold가 해결함
5. AGC 인증이 한 시점에서 실패하면 위의 이득이 통째로 사라짐 (60회 중 5회)

종합하면 "정지 위험에 맞는 학습 목표" 가설은 방향으로는 지지되지만 효과가 작음. 조기 정지 성능을 좌우한 것은 시점별 threshold 보정이었고, 정지 점수를 무엇으로 학습하느냐는 그 위에 얹는 소폭의 개선임

다만 QuickDraw 20-class 조건의 결과이고, test는 이미 여러 번 분석에 쓰인 데이터라 탐색적 결과로 봐야 함. head 구조와 입력 구성은 한 가지만 시도했고, 이력의 효과는 여기서 고른 요약값 11개에 한정된 결론임

## 재현

```bash
# Transformer: 공유받은 패키지 폴더에서 train 예측 생성 후, 이 레포 루트에서 실행
python export_train_logits.py --output ~/runs/transformer_handoff/head_train_full --device cuda --batch-size 256
python export_train_logits.py --checkpoint models/kd_best.pt --output ~/runs/transformer_handoff/head_train_full_kd --device cuda --batch-size 256
python -m qd.stop_scores --config configs/stop_score_study_transformer.yaml
python -m qd.stop_scores --config configs/stop_score_study_transformer_kd.yaml

# Mamba: 예측 저장과 실험을 한 번에
bash scripts/run_stop_score_study_mamba.sh

# 입력에 미래 정보가 섞이지 않는지 확인
python -m tests.test_stop_features
```

모델별 원본 수치는 각 폴더의 `results.json`과 `summary.md`에 있음
