#!/usr/bin/env bash
# VM에서 한 번만 실행: 가상환경 ~/venvs/qd 를 만들고 PyTorch와 Mamba를 설치한다.
# 전제: Ubuntu 22.04, NVIDIA 드라이버 설치됨 (nvidia-smi 동작). sudo는 쓰지 않는다.
#
#   bash scripts/setup_env.sh
set -euo pipefail

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d ~/venvs/qd ] || uv venv ~/venvs/qd --python 3.10
source ~/venvs/qd/bin/activate

uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy pyyaml einops transformers packaging
# mamba-ssm은 torch 버전에 맞춰 미리 빌드된 wheel을 쓴다 (g++ 불필요).
# --no-deps: 의존성 해석 과정에서 torch가 다른 버전으로 바뀌는 것을 막는다.
uv pip install --no-deps \
  "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

python -c "import torch, mamba_ssm; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
