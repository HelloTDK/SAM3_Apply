# SAM3 TensorRT 部署说明

`SAM3_Apply` 保留原有 HTTP 接口、鉴权、前端页面和返回字段，模型推理已切换为全 TensorRT：

- `shared-image-encoder.engine`：HTTP 检测和 TCP 原生点击共享的图像编码器，一个 ViT 主干、两个特征 neck。
- `text-encoder.engine`：文字 prompt 编码。
- `geometry-encoder.engine`：正/负框及跨图样例 prompt 编码。
- `decoder.engine`：检测框、置信度和分割 mask 输出。
- `native_interactive/interactive-decoder-first.engine`、`interactive-decoder-refine.engine`：TCP 首点及 mask 细化。

`sam_app.py` 在一个进程中提供 HTTP `8006` 和 TCP `6063`，不导入 Gradio 或监听 `7861`。
TCP 复用 HTTP runtime 中同一个图像 encoder、TensorRT execution context 和锁，额外保留原生点击解码头，
不会加载第二份视觉主干或 `interactive-image-encoder.engine`。TCP 图片不落盘。

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
shared-image-encoder.engine
text-encoder.engine
geometry-encoder.engine
decoder.engine
tokenizer.json
native_interactive/interactive-decoder-first.engine
native_interactive/interactive-decoder-refine.engine
```

服务按以下顺序使用 engine：

1. 环境变量 `SAM3_TRT_ENGINE_DIR`。
2. `SAM3_Apply/engines/`。

默认资源已放在 `SAM3_Apply/engines/`，程序不会回退读取仓库中的其他目录。

engine 与 GPU compute capability、TensorRT 版本相关。运行时会优先查找带 `_sm<算力>_trt<版本>` 后缀的 engine，没有时使用通用文件名。

旧 `vision-encoder.engine` 和 `native_interactive/interactive-image-encoder.engine` 可保留作历史对照，
合并服务不会加载它们。共享 encoder 必须使用 FP32 并关闭 TF32，原生点分割精度依赖此设置。
共享图同时输出检测 FPN 和交互 embedding，不能直接用其中一个旧 encoder 替代。

使用已有 SAM3 导出环境生成共享图，再在目标 GPU 的现有 TensorRT 环境构建（无须重建其余解码头）：

```bash
# 在 src/ 下，使用已有 sam3 环境与本地权重
python tools/sam3/export_native_interactive_onnx.py --shared-encoder-only \
  --output-dir SAM3_Apply/Onnx-Models --device cuda

# 在 SAM3_Apply/ 下，使用已有 TensorRT 环境
trtexec --onnx=Onnx-Models/shared-image-encoder.onnx \
  --saveEngine=engines/shared-image-encoder.engine --noTF32 --skipInference
```

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
SAM3_CONDA_ENV=base ./start_offline.sh --background
```

后台启动一个进程，同时监听 HTTP `0.0.0.0:8006` 和 TCP `0.0.0.0:6063`；脚本等待两个端口都就绪后才返回。`--background` 可覆盖容器中继承的 `SAM3_FOREGROUND=1`。TCP 与 HTTP 共用同一模型进程，不要另行启动 TCP 进程。前台运行：

```bash
SAM3_CONDA_ENV=base SAM3_FOREGROUND=1 ./start_offline.sh
```

TCP 端口可修改脚本中的 `SAM3_TCP_PORT` 默认值，或在启动时覆盖：

```bash
SAM3_CONDA_ENV=base SAM3_TCP_PORT=6065 ./start_offline.sh
```

HTTP 退出时同步关闭 TCP 监听；TCP 端口占用会使服务启动失败。不要同时启动旧独立 TCP/Gradio 服务，
也不要为该应用开启多个 uvicorn workers，否则会重复加载模型并发生 TCP 端口冲突。

健康检查：

```bash
curl http://127.0.0.1:8006/health
```

响应中的 `backend` 应为 `tensorrt`，并包含实际 `engine_dir` 和 engine 输入尺寸。

## 关键环境变量

- `SAM3_TRT_ENGINE_DIR`：engine 和 `tokenizer.json` 所在目录。
- `SAM3_NATIVE_ENGINE_DIR`：原生点击解码头目录，默认 `${SAM3_TRT_ENGINE_DIR}/native_interactive`。
- `SAM3_TRT_ROOT`：TensorRT 安装根目录，当前容器默认 `/usr/local/TensorRT-10.8.0.43`。
- `SAM3_CUDA_HOME`：CUDA 安装根目录，当前容器默认 `/usr/local/cuda-12.8`。
- `SAM3_TRT_NMS_IOU`：底层结果 NMS IoU，默认 `0.7`。
- `SAM3_VISUAL_PROMPT_MAX_CANDIDATES`：样例推理进入后处理的候选上限，默认 `32`。
- `SAM3_MAX_CONCURRENT_INFERENCES`：进入后台线程池的请求上限，默认 `4`。
- `SAM3_CUDA_CLEANUP_AFTER_REQUEST`：请求结束后释放 CuPy 内存池中的空闲块，默认 `1`。
- `SAM3_HOST`、`SAM3_PORT`：监听地址和端口，默认 `0.0.0.0:8006`。
- `SAM3_TCP_HOST`、`SAM3_TCP_PORT`：TCP 监听地址和端口，默认 `0.0.0.0:6063`。
- `SAM3_TCP_POLYGON_EPSILON`：TCP 返回轮廓简化容差，像素单位，默认 `2.0`；调大可减少顶点，`0` 关闭额外简化，范围 `0–50`。
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

## EISeg TCP 兼容服务

使用上面的 `start_offline.sh`，TCP 由 HTTP 应用生命周期启动，不另起 Python 进程。
前端沿用 EISeg `8000/8001/8002/8004` 命令和 `##@##` 分隔符，将连接端口改为 `6063`。

打开 HTTP 首页 `http://<服务地址>:8006/` 可使用“TCP 交互分割测试”面板：默认连接 `127.0.0.1:6063`，
断开后修改端口为 `6062` 即可测试原 EISeg，也可指定其他 loopback TCP 端口。选择图片并点击“载入图片”后，
支持正负点击、原图坐标输入、撤销、完成、重置、模型初始化/查询、多边形预览和 JSON 响应。
浏览器通过同源 `/ui/tcp-test` WebSocket 连接实际 TCP；主机地址从 HTTP 服务容器内访问，
反向代理需要支持 WebSocket Upgrade 并保留 Host。桥接不另载模型、不保存图片；原 EISeg `6062` 自身的图片缓存行为沿用旧实现。
用 `node --test tests/test_tcp_frontend.cjs` 可运行无依赖的前端状态/坐标测试；桥接测试包含在 Python unittest 中。

正/负点击使用原生 point decoder 和上轮 logits 细化，每个目标最多 32 点，撤销恢复对应历史。
TCP 图片只在内存中解码和编码，不保存图片、预览或 mask；独立小区域按最大区域面积的 1% 过滤，正点击选中的区域保留。
模型槽位 `0/1` 均复用 SAM3。每个连接缓存本图特征，目标完成/重置后可以继续标下一个目标。
HTTP/TCP 共享一份模型常驻显存；每个 TCP 会话仍需自己的约 20.25 MiB 图像特征及 CPU logits/轮廓历史。
旧 `app_tensorrt.py`、`ann_tcp_6063.py` 仅保留作独立调试入口，不属于生产启动流程。
完整协议、兼容边界、容器启动命令和测试命令见 [API_DOC.md 第 8 节](API_DOC.md#8-eiseg-tcp-兼容服务6063)。

2026-10-23 已在 TensorRT 10.8 / RTX A4000 环境通过 `start_offline.sh` 完成实际 HTTP/TCP
并行请求与关闭检查，确认只新增一个 GPU 进程、不监听 `7861`；该次请求后进程显存约 `7032 MiB`。
共享 encoder 对旧 FP32 交互 encoder 的首点/负点 refinement mask IoU 为 `0.999824/0.999967`。
共享 ONNX 和 `engines/shared-image-encoder.engine` 已在本地生成；部署时需连同该新 engine 一起复制，
其余已有解码头不需重建。完整 Python 测试集为 `python -m unittest discover -s tests -v`。

## 限制

当前 engine profile 为 batch size 1、geometry 单次最多 20 个框、decoder 最多 60 个 prompt token。服务会自动按参考图片和 token 上限分块执行，再按类别 NMS 合并结果，因此无需调用方拆分样例。

`start_cpu.sh` 属于旧 PyTorch 部署入口，全 TensorRT 服务不支持 CPU 推理。
