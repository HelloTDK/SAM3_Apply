# Ultralytics SAM3 部署包

`deploy3.3/` 复刻 `deploy3.1/` 的 HTTP 接口、鉴权、前端页面和返回字段，模型推理替换为 Ultralytics 的 `SAM3SemanticPredictor`。

## 包含内容

- 服务代码：`sam_app.py`
- 前端静态页：`static/`
- 启动脚本：`start_offline.sh`、`start_server.sh`
- 依赖清单：`requirements.txt`

以下资源体积较大或属于本机运行产物，默认不纳入 Git，需要部署时单独准备：

- 模型权重：`weights/sam3.pt`（默认）或 `weights/sam3.1_multiplex.pt`
- 翻译模型安装包：`argos_models/translate-zh_en-1_9.argosmodel`
- 翻译模型解包目录：`argos-packages/translate-zh_en-1_9/`
- 本地依赖兜底：`_vendor/`（可选，仅在 conda 环境不可写或离线部署时使用）
- 运行目录：`uploads/`、`results/`、`.matplotlib/`、`.cache/`

## 安装依赖

在 `sam3` conda 环境中安装：

```bash
conda activate sam3
python -m pip install -r requirements.txt
```

如果环境 site-packages 不可写，需要先修复 pip 用户目录或使用可写的 conda 环境。服务启动前会检查 `ultralytics.models.sam.SAM3SemanticPredictor` 是否可导入。

如需完全离线部署，可以先在有网络的机器上把依赖安装到 `_vendor/`，再连同模型资源一起拷贝到目标机器：

```bash
cd /expdata/givap/research/sam3/deploy3.3
python -m pip install -r requirements.txt -t _vendor
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
python -m pip install -U argostranslate
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
ARGOS_PACKAGES_DIR="$PWD/argos-packages" python - <<'PY'
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
./start_offline.sh
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

文本分割继续支持中文 prompt，服务会通过本地 Argos 模型翻译成英文后传入 Ultralytics SAM3。

## 关键环境变量

- `SAM3_CHECKPOINT_PATH`：默认 `./weights/sam3.pt`
- `SAM3_DEVICE`：默认 CUDA 可用时 `cuda:0`，否则 `cpu`
- `SAM3_INFER_DTYPE`：`bfloat16`、`float16`、`float32`
- `SAM3_ULTRALYTICS_IMGSZ`：默认 `1036`（Ultralytics SAM3 stride 14 的倍数）
- `SAM3_ULTRALYTICS_IOU`：默认 `0.7`
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
