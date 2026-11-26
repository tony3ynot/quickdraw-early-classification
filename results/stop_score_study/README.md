# 후속 실험 B0~B4: 정지 점수를 무엇으로 학습시키는 것이 좋은가 (Transformer)

분류 모델은 고정하고, 저장된 시점별 예측 위에서 "지금 멈춰도 되는가"를 판단하는 작은 MLP(정지 점수 head)만 학습해서 비교함. 모든 조건의 정지 threshold는 같은 AGC 설정으로 보정함

## 실험 조건

| 조건 | 학습 목표 | 입력 | AGC에 넣는 정지 점수 |
|---|---|---|---|
| B0 | 없음 | - | Max Softmax |
| B1 | Correctness: 지금 예측이 정답인가 | 현재 시점 | q_C |
| B2 | Correctness | 현재 + 과거 이력 | q_C |
| B3 | Harmful Risk: 지금 멈추면 완성 시 맞힐 답을 잃는가 | 현재 시점 | 1 − q_H |
| B4 | Harmful Risk | 현재 + 과거 이력 | 1 − q_H |

- 분류 모델: 공유받은 Transformer 2개 (Baseline, KD). 가중치는 건드리지 않음
- 현재 시점 입력: 정렬한 확률, 클래스별 확률, margin, entropy, 관찰 시점 (52차원)
- 과거 이력 입력: 직전 대비 confidence·margin 변화, 예측 클래스 변경 여부와 횟수, 같은 예측을 유지한 길이, 확률 분포 변화량 등 11개 추가 (63차원). 미래 시점 값이 섞이지 않는지 테스트로 확인함
- head: MLP (128 × 2층), BCE, train 1,400,000개의 (스케치, 시점) 쌍으로 학습. seed 3개
- 데이터: head 선택 valid 10,000 / AGC screening 20,000 / AGC calibration 20,000 / 평가 test 50,000 (공유받은 분할 그대로)
- AGC: α = 0.03, δ = 0.01 (조건 5개에 0.05를 나눔), grid 0.001. 구현은 공유받은 `qd/agc.py`를 그대로 사용

B0를 기존 설정(δ = 0.025)으로 보정하면 Baseline 93.552% / 51.250%, KD 93.300% / 54.604%로 기존 T1, T3 결과가 그대로 재현됨. δ = 0.01로 바꿔도 B0의 threshold는 달라지지 않았음

## 실험 결과 ①: 학습한 정지 점수는 Max Softmax보다 나은가?

B1~B4는 seed 3개의 평균 ± 표준편차

**Baseline Transformer**

| 조건 | Stopping Accuracy ↑ | Mean Earliness ↓ | Harmful Gap | 최대 누적 조건부 Harmful Gap |
|---|---|---|---|---|
| Fixed 0.9 (참고) | 92.53% | 48.50% | 3.39% | 7.01% |
| B0 Max Softmax | 93.55% | 51.25% | 2.61% | 2.69% |
| B1 Correctness, 현재 | 93.67% ± 0.15% | 51.14% ± 1.64% | 2.59% ± 0.17% | 2.66% ± 0.17% |
| B2 Correctness, 현재 + 이력 | 93.53% ± 0.02% | 50.00% ± 0.06% | 2.70% ± 0.02% | 2.79% ± 0.02% |
| B3 Harmful Risk, 현재 | 93.61% ± 0.04% | 49.92% ± 0.08% | 2.66% ± 0.04% | 2.72% ± 0.04% |
| B4 Harmful Risk, 현재 + 이력 | 93.58% ± 0.03% | 49.84% ± 0.14% | 2.65% ± 0.03% | 2.72% ± 0.03% |

**KD Transformer**

| 조건 | Stopping Accuracy ↑ | Mean Earliness ↓ | Harmful Gap | 최대 누적 조건부 Harmful Gap |
|---|---|---|---|---|
| Fixed 0.9 (참고) | 88.29% | 41.43% | 7.23% | 14.08% |
| B0 Max Softmax | 93.30% | 54.60% | 2.21% | 2.29% |
| B1 Correctness, 현재 | 93.42% ± 0.03% | 53.70% ± 0.11% | 2.31% ± 0.02% | 2.40% ± 0.03% |
| B2 Correctness, 현재 + 이력 | 93.40% ± 0.01% | 53.60% ± 0.13% | 2.28% ± 0.02% | 2.38% ± 0.02% |
| B3 Harmful Risk, 현재 | 93.39% ± 0.00% | 53.36% ± 0.04% | 2.33% ± 0.01% | 2.39% ± 0.01% |
| B4 Harmful Risk, 현재 + 이력 | 93.35% ± 0.03% | 53.10% ± 0.07% | 2.38% ± 0.04% | 2.45% ± 0.04% |

두 모델 모두 학습한 점수를 쓰면 정확도는 그대로이거나 조금 높고, 평균 관찰률은 0.9~1.5%p 줄어듦. 최대 누적 조건부 Harmful Gap은 전부 3% 이내로 유지됨

같은 test 스케치로 paired bootstrap을 했을 때 Mean Earliness 차이의 95% 구간이 0을 포함하지 않았음 (예: Baseline B4는 −1.51%p [−1.59, −1.43], KD B4는 −1.41%p [−1.47, −1.35])

Baseline의 B1만 표준편차가 큰데(±1.64%), seed 하나에서 AGC가 30% 시점의 정지를 인증하지 못해 "정지 금지"가 됐기 때문임. 그 seed는 B0보다 오히려 2.2%p 늦게 멈췄고(53.45%), 나머지 두 seed는 50.10%, 49.87%로 다른 조건과 비슷했음

**소결론:** 정지 점수를 학습시키면 Max Softmax보다 조금 더 일찍 멈출 수 있음. 다만 효과는 1%p 안팎이고, AGC 인증이 한 시점에서 실패하면 이득이 통째로 사라질 수 있음

## 실험 결과 ②: Correctness와 Harmful Risk 중 어느 쪽이 나은가?

| 비교 | Baseline | KD |
|---|---|---|
| B3 − B1 (현재 입력) | −0.12%p (인증에 실패한 seed 제외) | −0.34%p |
| B4 − B2 (이력 포함) | −0.16%p | −0.50%p |

Mean Earliness 차이이고 음수면 Harmful Risk 쪽이 더 일찍 멈춤

방향은 일관되게 Harmful Risk가 조금 앞서지만 차이가 작음. Baseline에서는 seed 간 표준편차(0.06~0.14%p)와 비슷한 크기이고, KD에서는 0.3~0.5%p로 그보다는 큼. Stopping Accuracy는 KD에서 Harmful Risk 쪽이 0.03~0.05%p 낮았고, Baseline에서는 차이가 없었음

점수가 각 목표를 얼마나 잘 가르는지(AUROC, test 10~90% 시점)를 보면 학습 목표에 따라 예상대로 갈림

| 조건 | Baseline: 정답 여부 | Baseline: 손해 없음 | KD: 정답 여부 | KD: 손해 없음 |
|---|---|---|---|---|
| B0 | 0.8938 | 0.8987 | 0.8914 | 0.8945 |
| B1 | 0.9118 | 0.9132 | 0.9109 | 0.9108 |
| B2 | 0.9123 | 0.9136 | 0.9114 | 0.9114 |
| B3 | 0.9041 | 0.9169 | 0.9020 | 0.9163 |
| B4 | 0.9044 | 0.9172 | 0.9020 | 0.9166 |

Harmful Risk로 학습한 점수는 "손해 없음"을 더 잘 가르고(0.913 → 0.917) "정답 여부"는 덜 가름(0.912 → 0.904). 학습 목표를 AGC가 통제하는 위험에 맞추면 그 위험은 실제로 더 잘 예측하게 되지만, 정지 효율로 이어지는 폭은 작았음

**소결론:** 학습 목표를 Harmful Risk로 맞추는 쪽이 조금 유리한 방향이지만, 이 실험 규모에서는 뚜렷한 차이라고 하기 어려움

## 실험 결과 ③: 과거 예측 이력은 도움이 되는가?

| 비교 | Baseline | KD |
|---|---|---|
| B2 − B1 (Correctness) | −0.02%p (인증에 실패한 seed 제외) | −0.10%p |
| B4 − B3 (Harmful Risk) | −0.08%p | −0.26%p |

AUROC도 이력을 넣었을 때 0~0.0006 오르는 정도였음

**소결론:** 과거 이력은 거의 도움이 되지 않았음. 현재 시점의 확률 분포에 이미 필요한 정보가 대부분 들어 있는 것으로 보임

## 실험 결과 ④: KD처럼 과신하는 모델에서 효과가 더 큰가?

KD 모델은 Fixed 0.9에서 초기 과신 때문에 Stopping Accuracy가 88.29%까지 떨어졌던 모델이라, 학습한 점수의 효과가 더 클 것으로 예상했음

그런데 B0 대비 개선 폭은 Baseline(1.2~1.4%p, B2~B4 기준)과 KD(0.9~1.5%p)가 비슷했음. AGC가 시점별 threshold를 따로 정하면서 과신을 이미 상당 부분 보정했기 때문으로 보임 (KD의 B0는 30%까지 정지 금지, 40~50%에서 0.96 요구)

**소결론:** 과신 문제는 AGC의 시점별 threshold만으로 대부분 해결되고, 학습한 점수가 추가로 주는 것은 정답·오답을 가르는 능력의 소폭 향상임

## 정리

1. 정지 점수를 따로 학습시키면 같은 AGC 위험 수준에서 0.9~1.5%p 더 일찍 멈춤. 통계적으로는 유의하지만 폭은 작음
2. 학습 목표(Correctness vs Harmful Risk)의 차이는 0.1~0.5%p로, Harmful Risk가 조금 앞서는 방향이지만 뚜렷하지 않음
3. 과거 예측 이력은 거의 도움이 되지 않음
4. AGC 인증은 시점 단위로 통과/실패가 갈리기 때문에, 점수가 조금만 달라져도 한 시점이 "정지 금지"가 되면서 결과가 크게 바뀔 수 있음

다만 Transformer 두 모델, QuickDraw 20-class 조건의 결과이고, test는 이미 여러 번 분석에 쓰인 데이터라 탐색적 결과로 봐야 함. head 구조와 입력 구성은 한 가지만 시도했음

## 재현

```bash
# train 예측 생성 (공유받은 패키지 폴더에서)
python export_train_logits.py --output ~/runs/transformer_handoff/head_train_full --device cuda --batch-size 256
python export_train_logits.py --checkpoint models/kd_best.pt --output ~/runs/transformer_handoff/head_train_full_kd --device cuda --batch-size 256

# B0~B4 실행 (이 레포 루트에서)
python -m tests.test_stop_features
python -m qd.stop_scores --config configs/stop_score_study_transformer.yaml
python -m qd.stop_scores --config configs/stop_score_study_transformer_kd.yaml
```

조건별 원본 수치는 `transformer_baseline/`, `transformer_kd/`의 `results.json`과 `summary.md`에 있음
