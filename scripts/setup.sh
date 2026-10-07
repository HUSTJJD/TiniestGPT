#!/usr/bin/env bash
# TiniestGPT 环境初始化（Linux / macOS）
#   bash scripts/setup.sh
#
# Linux 上额外受益：可以装上 Triton 高性能内核（Windows 不支持，会自动降级）。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/4] 创建 uv 虚拟环境 ..."
if ! command -v uv >/dev/null 2>&1; then
    echo "未找到 uv，请先安装：  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

uv venv --python 3.14

echo "[2/4] 安装依赖（PyTorch cu128）..."
# --extra kernel 会额外装 Triton（Linux 专用；Windows 会自动回退到 PyTorch 参考实现）
if [[ "$(uname -s)" == "Linux" ]]; then
    uv sync --extra all --extra kernel
else
    uv sync --extra all
fi

echo "[3/4] 验证 ..."
.venv/bin/python scripts/verify_env.py

echo "[4/4] 完成。使用方式："
echo "     source .venv/bin/activate"
echo "     uv run python -m tiniestgpt.cli info"
echo "     uv run python benchmarks/inference_ablation.py --checkpoint out/tiny/last.pt"
