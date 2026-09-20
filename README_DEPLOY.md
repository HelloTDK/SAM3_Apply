# SAM3 TensorRT 部署说明

`SAM3_Apply` 保留原有 HTTP 接口、鉴权、前端页面和返回字段，模型推理已切换为全 TensorRT：

- `vision-encoder.engine`：图像特征提取。
- `text-encoder.engine`：文字 prompt 编码。
- `geometry-encoder.engine`：正/负框及跨图样例 prompt 编码。
- `decoder.engine`：检测框、置信度和分割 mask 输出。

运行时不再加载 `sam3.pt`，也不依赖 PyTorch、torchvision 或 Ultralytics。中文 prompt 仍由本地 Argos 模型翻译成英文。

## 运行环境

当前开发环境是 Docker 容器 `sam3_trt_dev` 的 conda `base` 环境。该环境需要能够导入：

```bash
python - <<'PY'
import cupy
import tensorrt
import tokenizers
print("CuPy:", cupy.__version__)
print("TensorRT:", tensorrt.__version__)
PY
```

非交互 shell 不会读取容器的 `~/.bashrc`。启动脚本会默认设置：

```bash
SAM3_CUDA_HOME=/usr/local/cuda-12.8
SAM3_TRT_ROOT=/usr/local/TensorRT-10.8.0.43
```

镜像路径不同时，通过同名环境变量覆盖，启动脚本会据此设置 `PATH` 和 `LD_LIBRARY_PATH`。

通用 Web/图像依赖见 `requirements.txt`。不要仅为启动服务重新安装容器中已有的 CUDA/TensorRT 包；TensorRT 和 CuPy 必须与容器 CUDA 版本匹配。

## Engine 目录

目录中必须包含：

```text
vision-encoder.engine
text-encoder.engine
geometry-encoder.engine
decoder.engine
tokenizer.json
```

服务按以下顺序使用 engine：

1. 环境变量 `SAM3_TRT_ENGINE_DIR`。
2. `SAM3_Apply/engines/`。

默认资源已放在 `SAM3_Apply/engines/`，程序不会回退读取仓库中的其他目录。

engine 与 GPU compute capability、TensorRT 版本相关。运行时会优先查找带 `_sm<算力>_trt<版本>` 后缀的 engine，没有时使用通用文件名。

指定自定义目录：

```bash
export SAM3_TRT_ENGINE_DIR=/absolute/path/to/Engines
```

## 翻译资源

中文 prompt 需要以下离线资源：

```text
argos_models/translate-zh_en-1_9.argosmodel
argos-packages/translate-zh_en-1_9/metadata.json
```

相关环境变量：

```bash
export ARGOS_PACKAGES_DIR="$PWD/argos-packages"
export SAM3_ARGOS_MODEL_DIR="$PWD/argos_models"
export SAM3_ARGOS_AUTO_INSTALL=0
export SAM3_ARGOS_FORCE_OFFLINE=1
```

## 启动

在 `sam3_trt_dev` 容器中执行：

```bash
cd /expdata/givap/givap-train-server-v2.2.1/src/SAM3_Apply
SAM3_CONDA_ENV=base ./start_offline.sh
```

默认后台监听 `0.0.0.0:8006`。前台运行：

```bash
SAM3_CONDA_ENV=base SAM3_FOREGROUND=1 ./start_offline.sh
```

健康检查：

```bash
curl http://127.0.0.1:8006/health
```

响应中的 `backend` 应为 `tensorrt`，并包含实际 `engine_dir` 和 engine 输入尺寸。

## 关键环境变量

- `SAM3_TRT_ENGINE_DIR`：engine 和 `tokenizer.json` 所在目录。
- `SAM3_TRT_ROOT`：TensorRT 安装根目录，当前容器默认 `/usr/local/TensorRT-10.8.0.43`。
- `SAM3_CUDA_HOME`：CUDA 安装根目录，当前容器默认 `/usr/local/cuda-12.8`。
- `SAM3_TRT_NMS_IOU`：底层结果 NMS IoU，默认 `0.7`。
- `SAM3_VISUAL_PROMPT_MAX_CANDIDATES`：样例推理进入后处理的候选上限，默认 `32`。
- `SAM3_MAX_CONCURRENT_INFERENCES`：进入后台线程池的请求上限，默认 `4`。
- `SAM3_CUDA_CLEANUP_AFTER_REQUEST`：请求结束后释放 CuPy 内存池中的空闲块，默认 `1`。
- `SAM3_HOST`、`SAM3_PORT`：监听地址和端口，默认 `0.0.0.0:8006`。
- `SAM3_API_KEYS_FILE`：API Key 文件路径。

共享 TensorRT execution context 会在运行时内部加锁。样例 URL/视频任务会缓存已编码的几何 prompt，查询图只需重新执行图像编码和 decoder。

## 兼容接口

- `POST /detect`
- `POST /v1/segmentations`
- `POST /v1/box-segmentations`
- `POST /v1/similar-object-segmentations`
- `POST /v1/multi-similar-object-segmentations`
- `POST /v1/similar-object-segmentations/by-url`
- `POST/GET/DELETE /v1/similar-object-segmentations/tasks`
- `POST /similar-detect`
- `POST /multi-similar-detect`
- `GET /health`

请求和响应的完整说明见 `API_DOC.md`。

## 限制

当前 engine profile 为 batch size 1、geometry 单次最多 20 个框、decoder 最多 60 个 prompt token。服务会自动按参考图片和 token 上限分块执行，再按类别 NMS 合并结果，因此无需调用方拆分样例。

`start_cpu.sh` 属于旧 PyTorch 部署入口，全 TensorRT 服务不支持 CPU 推理。
