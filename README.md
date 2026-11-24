# QuickDraw 조기 스케치 분류

'딥러닝의 기초' 파이널 프로젝트 코드. QuickDraw 스케치를 그리는 도중에 클래스를 맞히고, 정확도와 계산 비용(연산량, 지연 시간)의 관계를 모델별로 비교한다.

현재 들어 있는 것은 **20개 클래스 / 2-layer Mamba baseline**이다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `configs/mamba_20cls.yaml` | 클래스 목록, 모델 크기, 학습·평가 설정 |
| `qd/data.py` | 데이터 다운로드, 전처리 캐시, 배치 생성, 관찰 비율 → 점 개수 정의 |
| `qd/model.py` | 2-layer Mamba 분류기 |
| `qd/train.py` | 학습 (매 시점 분류 loss) |
| `qd/evaluate.py` | 조기 예측 평가 (팀 공통 지표 4개) |
| `qd/flops.py` | Mamba FLOPs 계산 |
| `qd/dump.py` | 관찰 시점별 모델 출력 저장 (정지 정책을 재학습 없이 실험하기 위한 기반) |
| `qd/policies.py` | 정지 정책 평가 도구 (Harmful Gap, ECE 등) |
| `qd/optimal_stopping.py` | Deep Optimal Stopping 정지 정책 학습 |
| `qd/stop_report.py` | 한 모델에서 정지 정책들을 비교하는 보고서 생성 |
| `configs/mamba_20cls_elects.yaml` | ELECTS(정지 확률을 분류와 함께 학습) 설정 |
| `tests/test_causality.py` | 모델이 미래의 점을 쓰지 않는지 검사 |
| `scripts/` | VM 환경 설치, 학습+평가 실행 스크립트 |
| `results/` | 실험 결과 요약 (작은 파일만) |

데이터, 체크포인트, 로그는 레포에 올리지 않는다. VM에서는 `~/data/quickdraw/`와 `~/runs/`에 둔다.

## 데이터

- 출처: 공식 [Sketch-RNN QuickDraw 데이터셋](https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset) (`gs://quickdraw_dataset/sketchrnn/<class>.npz`)
- 분할: 파일에 들어 있는 공식 분할을 그대로 쓴다. 클래스당 train 70,000 / valid 2,500 / test 2,500.
  20개 클래스 합계는 1,400,000 / 50,000 / 50,000이다.
- 형식: 스케치 하나가 `[점 개수, 3]` 배열이고 각 점은 `(Δx, Δy, pen_lift)`이다.
- 전처리: `(Δx, Δy)`를 train 전체의 표준편차(50.84)로 나눈다. 길이 200 초과는 자르지만 20개 클래스에서는 해당 스케치가 없다(최대 184점).
- 20개 클래스: cat, dog, rabbit, elephant, fish, car, bus, bicycle, airplane, train, chair, table, clock, cup, umbrella, tree, flower, mushroom, cloud, mountain

SketchMamba 논문([arXiv 2607.23580](https://arxiv.org/abs/2607.23580))은 58개 클래스에서 train 406만 / val 14.5만 / test 14.5만을 썼다고 적는데, 클래스당 70,000 / 2,500 / 2,500으로 이 공식 분할과 숫자가 맞는다. 다만 논문에 분할 출처와 58개 클래스 목록은 나와 있지 않다.

## 모델과 학습

- 입력 점 `(Δx, Δy, pen_lift)` → Linear(3→256) → Mamba 블록 2개 → LayerNorm → Linear(256→20)
- 파라미터 883,220개. Mamba 구현은 공식 [mamba-ssm](https://github.com/state-spaces/mamba) 패키지([논문](https://arxiv.org/abs/2312.00752)).
- 모델이 causal이라 위치 t의 출력은 t번째 점까지만 본다. 그래서 스케치의 모든 시점에 분류 loss를 걸어, 그리다 만 스케치도 맞히도록 학습한다.
- AdamW, lr 1e-3 (warmup 1,000 step 후 cosine), weight decay 0.05, batch 512, bfloat16 mixed precision, 30 epoch.
- best 체크포인트는 valid에서 10개 관찰 시점 정확도의 평균으로 고른다. test는 마지막 평가에만 쓴다.

## 평가 방식 (팀 공통)

- 관찰 시점: 전체 점 개수의 10%, 20%, ..., 100%. 비율 f 시점에는 앞에서부터 `ceil(점 개수 × f)`개(최소 1개)를 본다.
- 정지 정책: 그 시점의 최대 클래스 확률이 0.90 이상이면 예측 확정. 100%까지 못 넘으면 100% 시점의 예측을 쓴다.
- batch size 1.

| 지표 | 정의 |
|---|---|
| Stopping Accuracy | 예측을 확정한 시점의 정확도 |
| Mean Earliness | 예측을 확정한 관찰 비율의 평균 |
| Cumulative FLOPs | 확정 전까지 수행한 추론의 FLOPs 합 |
| Cumulative Latency | 확정 전까지 수행한 추론 시간의 합 |

### 측정 가정

연산량과 지연 시간은 두 가지 방식으로 실제 실행해 나란히 기록한다. 두 방식은 같은 예측과 같은 정지 시점을 내고 비용만 다르다.

- **상태 유지**: 상태를 이어 가며 새로 들어온 점만 처리한다(mamba-ssm의 `step`). 이미 본 점을 다시 계산하지 않으므로 이 FLOPs가 Mamba에 실제로 필요한 계산량이다. Mamba와 GRU처럼 순서대로 읽는 모델에서만 가능하다.
- **시점마다 재추론**: 관찰 시점마다 그때까지의 prefix 전체를 처음부터 다시 추론한다. 어떤 모델에도 똑같이 적용할 수 있는 기준선이다.
- FLOPs는 `모델이 계산한 점 개수 × 점 하나당 FLOPs`다. 곱셈-덧셈 1회를 1로 세고(fvcore 기준) 활성화·정규화는 세지 않는다. fvcore는 Mamba의 selective scan을 세지 못하므로 `qd/flops.py`에서 수식으로 계산한다.
- 지연 시간은 L4 GPU에서 모델 계산 + softmax + 결과를 CPU로 가져오는 시간이다. 전처리는 포함하지 않는다. 이 값은 구현에 좌우된다. 상태 유지는 점 하나마다 모델을 호출하고, 재추론은 관찰 시점마다 한 번만 호출한다.
- 관찰 시점이 바뀌어도 본 점 개수가 같으면(점이 10개 미만인 스케치) 다시 예측하지 않는다.

### 알려진 한계

QuickDraw 원본은 스케치를 완성한 뒤 전체 크기를 0~255로 맞춘 좌표다. 그래서 앞부분 점의 좌표 크기에 완성된 그림의 크기 정보가 조금 섞여 있다. 모델과 평가 코드는 미래의 점을 쓰지 않지만, 이 부분은 데이터 자체의 성질이라 같은 데이터를 쓰는 모든 팀 모델에 똑같이 적용된다.

## 결과: 20개 클래스 / 2-layer Mamba

test 50,000개 전체, batch size 1, NVIDIA L4. 전체 수치는 [results/mamba2_20cls/eval_test.md](results/mamba2_20cls/eval_test.md)에 있다.

| 지표 | 상태 유지 (새 점만 처리) | 시점마다 재추론 |
|---|---|---|
| Stopping Accuracy | 92.55% | 92.55% |
| Mean Earliness | 51.2% | 51.2% |
| Cumulative FLOPs | 26.5 M | 99.2 M |
| Cumulative Latency (평균) | 41.73 ms | 7.56 ms |

참고로 조기 정지 없이 완성된 스케치를 한 번만 추론하면 정확도 94.85%, 53.2 M, 1.51 ms다.

| 관찰 비율 | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| 정확도 (정지 정책 없이) | 41.1% | 58.8% | 71.0% | 79.3% | 84.7% | 88.4% | 91.2% | 93.3% | 94.4% | 94.8% |

- 학습: train 1,400,000개, 30 epoch, 2.90시간 (L4 1장, GPU 메모리 최대 4.5GB). best는 epoch 28.
- 평균적으로 스케치의 절반(51.2%)만 보고 예측을 확정하며, 그때 정확도는 전체를 본 경우보다 2.3%p 낮다.
- 두 방식의 예측과 정지 시점은 test 50,000개 전부에서 일치한다. 차이는 비용뿐이다.
- FLOPs는 상태 유지가 재추론의 약 1/4이다. 재추론은 평균 5.1번 추론하면서 앞쪽 점을 반복 계산하기 때문에, 완성된 스케치를 한 번 추론하는 것(53.2 M)보다도 연산량이 크다.
- 지연 시간은 반대로 상태 유지가 5.5배 길다. 이 크기의 모델을 batch 1로 돌리면 계산보다 모델을 한 번 호출하는 고정 비용(약 1.5 ms)이 시간을 지배한다. 상태 유지는 점마다 호출하고(평균 28.6회), 재추론은 관찰 시점마다 호출한다(평균 5.1회). 즉 FLOPs는 계산한 점 개수에, 지연 시간은 호출 횟수에 비례한다.
- 상태 유지의 지연 시간은 mamba-ssm이 점 단위 `step`만 제공해서 생긴 값이다. 관찰 시점마다 새 점을 묶어 처리하도록 구현하면 줄어들 수 있으나 아직 구현하지 않았다.
- 가장 어려운 클래스는 dog(75.4%)이고, table·clock·dog은 예측 확정이 늦다(관찰 비율 67~71%).

## 실행 방법

모든 명령은 VM에 SSH로 접속한 뒤 레포 루트에서 입력한다.

```bash
# 1. 환경 설치 (한 번만)
bash scripts/setup_env.sh

# 2. 데이터 다운로드 + 캐시 생성, 클래스별 개수 출력
source ~/venvs/qd/bin/activate
python -m qd.data --config configs/mamba_20cls.yaml

# 3. 미래 정보 미사용 검사
python -m tests.test_causality

# 4. 학습 → test 평가 (SSH가 끊겨도 계속 돌도록 tmux 안에서)
tmux new -s train
bash scripts/run_baseline.sh configs/mamba_20cls.yaml
#   빠져나오기: Ctrl+B 누른 뒤 D      다시 보기: tmux attach -t train
```

진행 상황은 `tail -f ~/runs/mamba2_20cls/train.log`로 본다. 끝나면 `~/runs/mamba2_20cls/`에 `eval_test.md`(요약)와 `eval_test.json`(전체 수치)이 생긴다.

## 다른 모델을 같은 조건으로 비교하려면

- 데이터와 분할: `qd.data.build_cache`, `load_split`을 그대로 쓴다.
- 관찰 시점 정의: `qd.data.prefix_lengths`를 쓴다.
- 평가: `qd/evaluate.py`의 재추론 방식은 `model(x, last_only=True)`가 `[1, 클래스 수]` logits을 돌려주면 동작한다. 상태 유지 방식은 모델에 `init_stream_state`, `step`, `classify`가 있어야 한다. 쓸 방식은 설정의 `eval.modes`에서 고른다.
- FLOPs 계산(`qd/flops.py`)은 모델에 맞게 바꿔야 한다.

## 참고 자료

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, [arXiv 2312.00752](https://arxiv.org/abs/2312.00752), 코드 [state-spaces/mamba](https://github.com/state-spaces/mamba)
- *SketchMamba: A Lightweight State-Space Model for Joint Progressive Sketch Classification and Stroke Auto-Completion*, [arXiv 2607.23580](https://arxiv.org/abs/2607.23580)
- Ha & Eck, *A Neural Representation of Sketch Drawings* (Sketch-RNN), [arXiv 1704.03477](https://arxiv.org/abs/1704.03477)
- [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
