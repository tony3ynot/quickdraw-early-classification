"""모델이 미래의 점 정보를 쓰지 않는지(causality) 확인한다.

실행 (VM의 레포 루트에서):  python -m tests.test_causality

두 가지를 검사한다.
  1. 뒤쪽 점들을 바꿔도 앞쪽 위치의 출력은 변하지 않는다.
  2. prefix만 넣어 추론한 결과가, 전체를 넣고 같은 위치를 읽은 결과와 같다.
     (학습 중 검증은 2번 성질에 기대어 전체 시퀀스를 한 번만 추론한다.)
"""
import torch

from qd.model import MambaClassifier


@torch.no_grad()
def test_causality() -> None:
    torch.manual_seed(0)
    model = MambaClassifier(n_classes=20).cuda().eval()
    x = torch.randn(2, 60, 3, device="cuda")
    full = model(x)

    cut = 25
    changed = x.clone()
    changed[:, cut:] = torch.randn_like(changed[:, cut:])  # 25번째 점 이후를 전부 다른 값으로
    diff_future = (model(changed)[:, :cut] - full[:, :cut]).abs().max().item()
    assert diff_future < 1e-4, f"미래 점이 과거 출력에 영향을 준다: {diff_future}"

    diff_prefix = (model(x[:, :cut], last_only=True) - full[:, cut - 1]).abs().max().item()
    assert diff_prefix < 1e-4, f"prefix 추론과 전체 추론이 다르다: {diff_prefix}"

    print(f"causality OK (미래 변경 시 차이 {diff_future:.2e}, prefix 추론 차이 {diff_prefix:.2e})")


if __name__ == "__main__":
    test_causality()
