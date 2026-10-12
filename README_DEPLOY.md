# SAM3 服务部署（Ascend OM / Ultralytics）

当前分支默认使用 Ascend OM 后端，保留已有 HTTP 接口、鉴权、前端页面和返回字段。
`SAM3_BACKEND=ultralytics` 可以切回原来的 `SAM3SemanticPredictor` PT 后端。

## Ascend OM 部署

生产包需要 `sam_app.py`、整个 `sam3_service/`、`static/` 和模型资源；不依赖 `tools/`。
把以下文件放到 `om_weights/`（可用 `SAM3_OM_MODEL_DIR` 指定其他目录）：

```text
vision-encoder.om
text-encoder.om
geometry-encoder.om
decoder.om
decoder-pos.om
decoder-mix.om
tokenizer.json
```

ACL 由 Ascend CANN 提供，运行前需要加载 CANN 环境及驱动库路径。OM 的图像推理不
依赖 PyTorch、Ultralytics 或 CLIP；中文翻译继续使用 Argos 及其依赖。
`requirements.txt` 引用 `requirements-om.txt`，两个入口都固定安装官方 CPU 版
`torch==2.6.0+cpu`，满足 Argos 的 Stanza 依赖，不会拉取 CUDA/NVIDIA 包。
安装时需要能访问 PyPI 和 `https://download.pytorch.org/whl/cpu`。
OM 模式始终设置 `ARGOS_DEVICE_TYPE=cpu`，翻译在 CPU 上运行，SAM3 推理在 NPU 上运行。
启动脚本会检查必需模块并报告缺项，不会自动安装或替换环境中的包。

在 `sam3-om-dev` 中，推理 demo 对应的 Python 为 `/opt/python310_venv/bin/python`，
项目挂载在 `/sam3`。使用该 Python 安装服务依赖并启动：

```bash
docker exec -it sam3-om-dev bash -lc \
  'cd /sam3 && /opt/python310_venv/bin/python -m pip install -r requirements.txt'

docker exec -it sam3-om-dev bash -lc \
  'cd /sam3 && ./start_server.sh'
```

`start_offline.sh` 同样支持 `SAM3_PYTHON`，默认后台运行；加 `SAM3_FOREGROUND=1`
可前台运行。OM 模式优先使用显式指定的 `SAM3_PYTHON`，否则在存在时自动选择
`/opt/python310_venv/bin/python`，其他环境使用 `python`，无需 Conda 激活和 `.pt`
权重。直接入口同样会在导入 Argos 前设置 CPU 翻译：

```bash
cd /sam3
SAM3_BACKEND=om SAM3_OM_DEVICE_ID=0 /opt/python310_venv/bin/python sam_app.py
```

测试命令（额外需要 httpx，图片使用项目已有篮球场示例）：

```bash
# 无 NPU 的单元检查
python -m unittest discover -s tests -v

# 配好服务依赖的容器内，包含真实 NPU HTTP、URL 图片/视频任务、并发和卸载重载
SAM3_RUN_OM_INTEGRATION=1 SAM3_OM_TEST_IMAGE=/sam3/sam3-demo.jpg \
  /opt/python310_venv/bin/python -m unittest discover -s tests -v
```

接口兼容性和固定 OM 的框补齐、提示容量限制见 `API_DOC.md` 开头。
以下 PT 权重准备和 Ultralytics 安装步骤仅适用于 `SAM3_BACKEND=ultralytics`。

## 包含内容

- 服务代码：`sam_app.py`
- 前端静态页：`static/`
- 启动脚本：`start_offline.sh`、`start_server.sh`
- 依赖清单：`requirements.txt`
- OM 依赖清单：`requirements-om.txt`
- PT 依赖清单：`requirements-ultralytics.txt`

以下资源体积较大或属于本机运行产物，默认不纳入 Git，需要部署时单独准备：

- 模型权重：`weights/sam3.pt`（默认）或 `weights/sam3.1_multiplex.pt`
- 翻译模型安装包：`argos_models/translate-zh_en-1_9.argosmodel`
- 翻译模型解包目录：`argos-packages/translate-zh_en-1_9/`
- 本地依赖兜底：`_vendor/`（可选，仅在 conda 环境不可写或离线部署时使用）
- 运行目录：`uploads/`、`results/`、`.matplotlib/`、`.cache/`

## 安装依赖

Ascend OM 使用前面的容器安装命令。PT 后端在 `sam3` conda 环境中安装：

```bash
conda activate sam3
python -m pip install -r requirements-ultralytics.txt
```

`requirements-ultralytics.txt` 保留原 PT 依赖，不固定 CPU torch；CPU 部署需先从
PyTorch CPU 索引安装匹配的 torch/torchvision，CUDA 部署使用对应的 CUDA 构建。
已有 NVIDIA 环境请使用 PT 清单，避免默认 OM 清单将 torch 替换为 CPU 版。

如果环境 site-packages 不可写，需要先修复 pip 用户目录或使用可写的 conda 环境。服务启动前会检查 `ultralytics.models.sam.SAM3SemanticPredictor` 是否可导入。

如需完全离线部署，可以先在有网络的机器上把依赖安装到 `_vendor/`，再连同模型资源一起拷贝到目标机器：

```bash
cd /expdata/givap/research/sam3/deploy3.3
python -m pip install -r requirements-ultralytics.txt -t _vendor
python -m pip uninstall -y clip || true
python -m pip install git+https://github.com/ultralytics/CLIP.git -t _vendor
```

启动脚本会自动把 `_vendor/` 加入 `PYTHONPATH`。

## 准备模型和离线资源

以下命令均在部署目录执行：

```bash
cd /expdata/givap/research/sam3/deploy3.3
mkdir -p weights argos_models argos-packages uploads results
```

### SAM3 权重

`sam3.pt` 需要先在 Hugging Face 申请访问 `facebook/sam3`，审批通过并登录后下载。Ultralytics 文档也说明 SAM3 权重不会自动下载，必须手动下载 `sam3.pt` 后放到工作目录或显式指定路径。

```bash
python -m pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download facebook/sam3 sam3.pt --local-dir weights
```

下载完成后应存在：

```text
weights/sam3.pt
```

如果使用 SAM 3.1 Object Multiplex，可从 `facebook/sam3.1` 下载：

```bash
huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt --local-dir weights
```

下载完成后应存在：

```text
weights/sam3.1_multiplex.pt
```

也可以使用项目根目录已有的 ModelScope 下载脚本：

```bash
python ../../scripts/download_model.py
```

默认启动会优先使用 `weights/sam3.pt`；如果想指定其它权重：

```bash
export SAM3_CHECKPOINT_PATH=/abs/path/to/sam3.pt
```

### Argos 中文到英文翻译资源

中文 prompt 会先通过 Argos Translate 翻译成英文，再传给 SAM3。联网机器上可以直接通过 Argos 官方包索引下载 `zh -> en` 模型：

```bash
# OM 环境先使用 requirements.txt 安装依赖，确保 Stanza 使用 CPU torch。
export ARGOS_DEVICE_TYPE=cpu
python - <<'PY'
from pathlib import Path
import shutil
import argostranslate.package

from_code = "zh"
to_code = "en"
model_dir = Path("argos_models")
model_dir.mkdir(exist_ok=True)

argostranslate.package.update_package_index()
packages = argostranslate.package.get_available_packages()
package = next(
    item for item in packages
    if item.from_code == from_code and item.to_code == to_code
)
downloaded = Path(package.download())
target = model_dir / downloaded.name
shutil.copy2(downloaded, target)
print(target)
PY
```

然后把 `.argosmodel` 安装/解包到本项目的 `argos-packages/`：

```bash
ARGOS_DEVICE_TYPE=cpu ARGOS_PACKAGES_DIR="$PWD/argos-packages" python - <<'PY'
import argostranslate.package

argostranslate.package.install_from_path("argos_models/translate-zh_en-1_9.argosmodel")
PY
```

准备完成后应至少存在：

```text
argos_models/translate-zh_en-1_9.argosmodel
argos-packages/translate-zh_en-1_9/metadata.json
argos-packages/translate-zh_en-1_9/model/model.bin
argos-packages/translate-zh_en-1_9/stanza/resources.json
```

离线机器不能联网时，在有网络的机器上按上面步骤准备好 `argos_models/` 和 `argos-packages/`，再整体拷贝到部署目录：

```bash
rsync -av argos_models argos-packages user@server:/path/to/deploy3.3/
```

### 不需要手动下载的目录

这些目录是运行时自动生成或本机缓存，不需要上传 Git，也不需要提前准备：

```text
uploads/
results/
.matplotlib/
.cache/
```

启动脚本会创建 `uploads/` 和 `results/`；`.matplotlib/`、`.cache/` 会由相关 Python 库按需生成。

## 启动

```bash
cd /expdata/givap/research/sam3/deploy3.3
SAM3_BACKEND=ultralytics ./start_offline.sh
```

默认监听：`0.0.0.0:8006`

## 保持兼容的接口

- `POST /detect`
- `POST /v1/segmentations`
- `POST /v1/box-segmentations`
- `POST /v1/similar-object-segmentations`
- `POST /similar-detect`（兼容旧 multipart 单样例/同图接口）
- `POST /multi-similar-detect`（前端当前统一使用的样例识别接口）
- `GET /health`
- `GET /api-list.json`
- `GET /api-list`
- `GET /apis`
- `POST/GET/DELETE /ui/api-keys`
- `POST/GET/DELETE /v1/api-keys`

文本分割继续支持中文 prompt，服务会通过本地 Argos 模型翻译成英文后传入选定的 SAM3 后端。样例识别中，顶层 prompt 与样例类别同义时会复用到样例类别分组；类别不同时会保留独立的纯文本检测组。

## 关键环境变量

- `SAM3_BACKEND`：当前分支默认 `om`；另一个选项为 `ultralytics`
- `SAM3_OM_MODEL_DIR`：默认 `./om_weights`（相对于项目根目录）
- `SAM3_OM_DEVICE_ID`：默认 `0`，指定 ACL 使用的昇腾设备
- `SAM3_PYTHON`：启动脚本使用的 Python；OM 在存在时默认 `/opt/python310_venv/bin/python`，否则使用 `python`
- `ARGOS_DEVICE_TYPE`：OM 固定为 `cpu`，包括直接运行 `sam_app.py`；PT 保留原设备选择
- `SAM3_SERIALIZE_MODEL_ACCESS`：OM 始终串行访问，PT 可按原配置控制
- `SAM3_IDLE_MODEL_UNLOAD_SECONDS`：OM/PT 都支持；默认 `0`，关闭空闲卸载
- `SAM3_CHECKPOINT_PATH`：默认 `./weights/sam3.pt`
- `SAM3_DEVICE`：默认 CUDA 可用时 `cuda:0`，否则 `cpu`
- `SAM3_INFER_DTYPE`：`bfloat16`、`float16`、`float32`
- `SAM3_ULTRALYTICS_IMGSZ`：默认 `1036`（Ultralytics SAM3 stride 14 的倍数）
- `SAM3_ULTRALYTICS_IOU`：默认 `0.7`
- `SAM3_NMS_DEBUG=1`：仅用于排查样例标注重框，输出最终 NMS 的候选、抑制和保留明细；默认 `0`
- `SAM3_ARGOS_MODEL_DIR`：默认 `./argos_models`
- `ARGOS_PACKAGES_DIR`：默认 `./argos-packages`
- `SAM3_ARGOS_AUTO_INSTALL=0`
- `SAM3_ARGOS_FORCE_OFFLINE=1`

## 说明

`/v1/box-segmentations` 仍接收旧版 `[x, y, w, h]`，内部会转换成 Ultralytics 需要的 `[x1, y1, x2, y2]`，返回保持旧版格式。

`/v1/similar-object-segmentations`、`/similar-detect` 和 `/multi-similar-detect` 实现“样图 A + A 中目标框 -> 待识别图 B 中相似目标”的流程，返回 `pic_labels[].bnd_points` 和 `pic_labels[].polygon_points`。前端统一使用 `/multi-similar-detect`：普通单样例就是只提交一个正样本实例，多类别/多样例则继续追加标签、样例和实例框。

兼容 JSON/旧 multipart 接口仍支持两种模式：

- `similar_mode=feature_match`（默认）：跨图原生 visual prompt。先把样图框编码成 reference prompt embedding，再直接在待识别图上跑 SAM3 grounding。
- `similar_mode=same_image_prompt`：示例框和待找目标在同一张图，直接使用 SAM3 原生 visual prompt。
