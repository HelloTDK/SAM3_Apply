#!/usr/bin/env bash
set -euo pipefail

# 自动挂载当前 SAM3_Apply 目录，避免部署路径变化后还要修改脚本。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 前台运行服务，使服务进程退出时 Docker 可以按 unless-stopped 策略自动重启。
# 显式向容器暴露 NVIDIA GPU 和宿主机驱动库。
docker run --name=sam3_trt_dev_runtime \
  --hostname=localhost.localdomain \
  --entrypoint sh \
  --volume /etc/localtime:/etc/localtime:ro \
  --volume "${SCRIPT_DIR}:/sam3" \
  --network=host \
  --privileged \
  --workdir=/sam3 \
  --restart=unless-stopped \
  --gpus all \
  --env SAM3_FOREGROUND=1 \
  --detach=true \
  givap/sam3-trt:v1.1 \
  -c 'exec ./start_offline.sh'
