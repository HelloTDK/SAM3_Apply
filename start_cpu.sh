#!/usr/bin/env bash
set -euo pipefail

# 全 TensorRT 后端必须使用 NVIDIA GPU；保留文件名仅用于给旧部署脚本明确报错。
echo "ERROR: SAM3 TensorRT service does not support CPU inference. Use start_offline.sh in sam3_trt_dev."
exit 1
