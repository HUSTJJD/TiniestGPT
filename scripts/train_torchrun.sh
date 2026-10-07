#!/usr/bin/env bash
# 多卡训练启动脚本（DDP / FSDP）
#
#   bash scripts/train_torchrun.sh                                  # 默认 2 卡，pretrain_tiny
#   NPROC=4 CONFIG=recipes/pretrain_moe.yaml bash scripts/train_torchrun.sh
#
# 为什么用 torchrun 而不是 python -m：
#   torchrun 负责拉起 N 个进程、设置 RANK / WORLD_SIZE / LOCAL_RANK、
#   以及失败重启（elastic）—— 这些都是 DDP 的前置条件。
#   train/distributed.py::init_distributed 正是读这几个环境变量。
set -euo pipefail

NPROC="${NPROC:-2}"
CONFIG="${CONFIG:-recipes/pretrain_tiny.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "torchrun: nproc_per_node=${NPROC} config=${CONFIG}"
exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  -m tiniestgpt.cli pretrain --config "${CONFIG}"
