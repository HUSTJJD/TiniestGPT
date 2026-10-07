# TiniestGPT 推理服务镜像
#
# 选 **-devel** 而不是 -runtime：镜像里带 nvcc，
# 于是 tiniestgpt/kernels 里的手写 CUDA kernel 可以在容器内 JIT 编译
# （这是本项目第一层内容的必要条件）。
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions

# ninja-build：torch 的 C++/CUDA JIT 需要它
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-dev \
        build-essential ninja-build git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv：与本地开发保持一致，直接复用 uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
COPY tiniestgpt ./tiniestgpt
COPY scripts ./scripts
COPY recipes ./recipes
COPY benchmarks ./benchmarks

# --extra kernel 里包含 triton + ninja；CUDA 版 torch 由 pyproject 的 index 指定
RUN uv venv --python 3.11 \
    && uv sync --extra all --extra kernel --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# 权重与数据用 volume 挂进来（见 docker-compose.yml）
CMD ["python", "-m", "tiniestgpt.cli", "serve", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--checkpoint", "/app/out/tiny/last.pt"]
