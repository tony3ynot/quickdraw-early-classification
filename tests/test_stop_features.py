"""정지 점수 head의 입력이 미래 시점의 예측을 쓰지 않는지 확인한다.

실행 (레포 루트에서):  python -m tests.test_stop_features

시점 t 이후의 logits를 전부 바꿔도 시점 t까지의 입력이 변하지 않아야 한다.
현재 시점 입력과 과거 이력 입력 둘 다 검사한다.
"""
import torch

from qd.stop_scores import build_features


def test_features_are_causal() -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 10, 20)
    for history in (False, True):
        base = build_features(logits, history)
        for cut in range(1, 10):
            changed = logits.clone()
            changed[:, cut:] = torch.randn_like(changed[:, cut:])
            diff = (build_features(changed, history)[:, :cut] - base[:, :cut]).abs().max().item()
            assert diff == 0.0, f"history={history}, cut={cut}: 미래 예측이 입력에 섞임 ({diff})"
        print(f"features OK (history={history}, 입력 차원 {base.shape[-1]})")


if __name__ == "__main__":
    test_features_are_causal()
