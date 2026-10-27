# SAM3 TensorRT 接口文档

服务地址示例：`http://192.168.100.25:8006`

本文档覆盖当前常用的以下能力：

- prompt 文本识别：按文本类别找目标。
- 画区域 / box 分割：用户给框，SAM3 分割框内目标。
- 示例目标相似识别：包括跨图特征匹配识别、同图拉框识别。
- 样例图标注：通过远程 `sample_url` 和 `data_url` 批量标注图片，避免传输大量 Base64。
- EISeg TCP 兼容标注：`sam_app.py` 同时监听 `6063`，与 HTTP 共享图像编码器，保留原有 `8000` 系列命令，见第 8 节。
- 原生正负点交互 Demo：旧调试工具保留，但 `start_offline.sh` 不启动 Gradio `7861`，历史说明见第 9 节。

当前服务的视觉编码、文字编码、几何 prompt 编码和 mask decoder 均由 TensorRT engine 执行；HTTP 路径、请求字段及响应字段保持兼容。

坐标约定：

- 所有业务入参/出参坐标均使用原始图片像素坐标。
- `bnd_points` 格式固定为 `[x, y, w, h]`。
- `polygon_points` 格式为 `[[x1, y1], [x2, y2], ...]`。
- 图片 base64 支持纯 base64 或 `data:image/jpeg;base64,...`。

## 鉴权

`/v1/*` 接口需要 API Key，二选一：

```http
Authorization: Bearer <api_key>
```

或：

```http
X-API-Key: <api_key>
```

`/detect`、`/similar-detect`、`/multi-similar-detect` 是 multipart 测试接口，当前代码未强制 API Key。前端相似识别统一使用 `/multi-similar-detect`，`/similar-detect` 仅作为旧客户端兼容入口保留。

## 1. Prompt 文本识别


### 1.1 JSON 接口

`POST /v1/segmentations`

请求头：

```http
Content-Type: application/json
Authorization: Bearer <api_key>
```

请求体：

```json
{
  "pic_id": "demo-001",
  "image_base64": "data:image/jpeg;base64,/9j/...",
  "prompt": "#人#安全帽#车辆",
  "confidence_threshold": 0.3,
  "polygon_simplify_epsilon": 2.0,
  "return_result_image": false
}
```

字段说明：

多框请求时，服务端会根据每个输入框与模型预测框的 IoU 做一对一匹配，再返回对应的分割结果；因此 `segmentations` 的顺序不依赖模型内部预测顺序。响应中的 `index` 表示匹配到的输入框下标（从 `0` 开始），`input_bnd_points` 保留实际参与匹配的输入框。

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pic_id` | 是 | string | 客户端图片 ID |
| `image_base64` | 是 | string | 待识别图片 |
| `prompt` | 是 | string | 类别文本，仅支持 `#` 分隔，例如 `#人#猫#桌子`；分隔符两侧允许换行或空白，故 `#人#猫\n#桌子` 也等价于三个类别；中文会自动翻译成英文 |
| `confidence_threshold` | 否 | number | 置信度阈值，默认 `0.3` |
| `polygon_simplify_epsilon` | 否 | number | 多边形简化参数，默认 `2.0` |
| `return_result_image` | 否 | boolean | 是否生成并保存可视化结果图，默认 `false`；需要调试图时显式传 `true` |

响应示例：

```json
{
  "model": "sam3-tensorrt",
  "pic_id": "demo-001",
  "success": true,
  "pic_labels": [
    {
      "category": "人",
      "translated_category": "person",
      "score": 0.812345,
      "bnd_points": [120.0, 80.0, 40.0, 110.0],
      "polygon_points": [[121.0, 80.0], [160.0, 82.0]],
      "mask_area": 2840
    }
  ],
  "num_detections": 1,
  "classes_detected": 1,
  "detection_details": {"人": 1},
  "prompt": "#人#安全帽#车辆",
  "translated_prompt": "#person#helmet#vehicle",
  "was_translated": true,
  "confidence_threshold": 0.3,
  "result_image": null,
  "created": 1779340000,
  "processing_time_ms": 850
}
```

### 1.2 Multipart 测试接口

`POST /detect`

表单字段：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `file` | 是 | file | 待识别图片 |
| `prompt` | 是 | string | 类别文本 |
| `confidence` | 否 | number | 默认 `0.3` |
| `polygon_simplify_epsilon` | 否 | number | 默认 `2.0` |
| `pic_id` | 否 | string | 图片 ID |

该页面测试接口会生成并保存可视化结果图；生产批量调用请使用 `/v1/segmentations` 并保持 `return_result_image=false`。

curl 示例：

```bash
curl -X POST 'http://192.168.100.25:8006/detect' \
  -F 'file=@/path/to/image.jpg' \
  -F 'prompt=#人#安全帽' \
  -F 'confidence=0.3'
```

## 2. 画区域 / Box 分割

用途：前端或客户端在图上画框后，把框传给 SAM3，返回框内目标的 mask 多边形和校正后的框。

`POST /v1/box-segmentations`

请求头：

```http
Content-Type: application/json
Authorization: Bearer <api_key>
```

单框请求：

```json
{
  "pic_id": "box-001",
  "image_base64": "data:image/jpeg;base64,/9j/...",
  "bnd_points": [120, 80, 260, 300],
  "polygon_simplify_epsilon": 2.0
}
```

多框请求：

```json
{
  "pic_id": "box-002",
  "image_base64": "data:image/jpeg;base64,/9j/...",
  "bnd_points_list": [
    [120, 80, 260, 300],
    [520, 220, 80, 160]
  ],
  "polygon_simplify_epsilon": 2.0
}
```

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pic_id` | 是 | string | 图片 ID |
| `image_base64` | 是 | string | 待分割图片 |
| `bnd_points` | 二选一 | array | 单框 `[x,y,w,h]`，也兼容 `[[x,y,w,h], ...]` |
| `bnd_points_list` | 二选一 | array | 多框列表 |
| `polygon_simplify_epsilon` | 否 | number | 多边形简化参数，默认 `2.0` |

响应示例：

```json
{
  "model": "sam3-tensorrt",
  "pic_id": "box-001",
  "success": true,
  "segmentations": [
    {
      "input_bnd_points": [120.0, 80.0, 260.0, 300.0],
      "bnd_points": [128.0, 92.0, 210.0, 265.0],
      "polygon_points": [[130.0, 95.0], [330.0, 100.0]],
      "score": 0.9321,
      "mask_area": 25100,
      "used_fallback": false,
      "index": 0
    }
  ],
  "num_segmentations": 1,
  "bnd_points": [128.0, 92.0, 210.0, 265.0],
  "polygon_points": [[130.0, 95.0], [330.0, 100.0]],
  "score": 0.9321,
  "mask_area": 25100,
  "used_fallback": false,
  "created": 1779340000,
  "processing_time_ms": 430
}
```

## 3. 示例目标相似识别

Base64 小请求使用 `POST /v1/similar-object-segmentations`，通过 `similar_mode` 切换模式。

大批量样例图标注不要使用 Base64。样例图 1-300 张、待标注图 1-5000 张时，使用后面的 URL/Manifest 接口：

- 单张目标图：`POST /v1/similar-object-segmentations/by-url`
- 批量目标图：`POST /v1/similar-object-segmentations/tasks`

支持模式：

| `similar_mode` | 说明 |
| --- | --- |
| `feature_match` | 跨图原生 visual prompt。先把参考框编码成 reference prompt，再直接在目标图上跑 SAM3 grounding。 |
| `same_image_prompt` | 同图拉框识别。示例框和待找目标在同一张图，直接使用 SAM3 原生 visual prompt。 |

### 3.1 跨图 Feature Match 识别

`POST /v1/similar-object-segmentations`

请求体：

```json
{
  "pic_id": "similar-001",
  "reference_image_base64": "data:image/jpeg;base64,/9j/...",
  "query_image_base64": "data:image/jpeg;base64,/9j/...",
  "reference_bnd_points": [120, 80, 260, 300],
  "prompt": "红色安全帽",
  "top_k": 5,
  "sam_threshold": 0.6,
  "similarity_threshold": 0.6,
  "polygon_simplify_epsilon": 2.0,
  "similar_mode": "feature_match"
}
```

多张目标图：

```json
{
  "pic_id": "similar-batch-001",
  "reference_image_base64": "data:image/jpeg;base64,/9j/...",
  "query_image_base64_list": [
    "data:image/jpeg;base64,/9j/...",
    "data:image/jpeg;base64,/9j/..."
  ],
  "reference_bnd_points": [120, 80, 260, 300],
  "top_k": 5,
  "sam_threshold": 0.6,
  "similarity_threshold": 0.6,
  "similar_mode": "feature_match"
}
```

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `reference_image_base64` | 是 | string | 示例图 A |
| `query_image_base64` | 是 | string | 目标图 B，单图 |
| `query_image_base64_list` | 否 | array | 目标图列表；和 `query_image_base64` 二选一 |
| `reference_bnd_points` | 是 | array | 示例图 A 中目标框 `[x,y,w,h]` |
| `prompt` | 否 | string | 可选相似目标文本描述；拼接模式填写后会和示例框一起传给 SAM3，中文会自动翻译成英文 |
| `top_k` | 否 | integer | 返回最多目标数，默认 `5`，范围 `1-20` |
| `sam_threshold` | 否 | number | SAM3 grounding 分数阈值，默认 `0.6` |
| `similarity_threshold` | 否 | number | 兼容旧客户端字段；当前不再执行余弦相似度过滤 |
| `polygon_simplify_epsilon` | 否 | number | 多边形简化参数，默认 `2.0` |
| `similar_mode` | 否 | string | `feature_match`、`same_image_prompt` |

响应关键字段：

```json
{
  "success": true,
  "similar_mode": "feature_match",
  "prompt": "红色安全帽",
  "translated_prompt": "red helmet",
  "box_text_prompt_enabled": true,
  "reference_bnd_points": [120.0, 80.0, 260.0, 300.0],
  "top_k": 5,
  "sam_threshold": 0.6,
  "similarity_threshold": 0.6,
  "num_candidates": 12,
  "num_matches": 3,
  "pic_labels": [
    {
      "category": "similar_object",
      "score": 0.71,
      "sam_score": 0.71,
      "similarity_score": 0.71,
      "combined_score": 0.71,
      "coarse_similarity": 0.71,
      "bnd_points": [520.0, 220.0, 80.0, 160.0],
      "polygon_points": [[521.0, 221.0], [600.0, 224.0]],
      "mask_area": 6700,
      "concat_scale": 1.0
    }
  ],
  "reference_result_image": "result_20260521_101000_000001.jpg",
  "result_image": "result_20260521_101000_000003.jpg",
  "processing_time_ms": 1800
}
```

图片字段访问方式：

```text
GET /results/{filename}
```

例如：

```text
http://192.168.100.25:8006/results/result_20260521_101000_000003.jpg
```

### 3.2 同图拉框识别

用途：示例目标和待搜索目标在同一张图上。只需要传一张图和一个示例框。

`POST /v1/similar-object-segmentations`

请求体：

```json
{
  "pic_id": "same-image-001",
  "reference_image_base64": "data:image/jpeg;base64,/9j/...",
  "reference_bnd_points": [120, 80, 260, 300],
  "top_k": 5,
  "sam_threshold": 0.6,
  "similarity_threshold": 0.6,
  "polygon_simplify_epsilon": 2.0,
  "similar_mode": "same_image_prompt"
}
```

说明：

- `same_image_prompt` 模式下不需要 `query_image_base64`。
- 返回的 `pic_labels[].bnd_points` 仍然是这张原图上的坐标。
- `pic_labels[].is_reference_overlap=true` 表示结果和示例框高度重叠，通常就是示例目标本身。

响应示例：

```json
{
  "success": true,
  "similar_mode": "same_image_prompt",
  "reference_bnd_points": [120.0, 80.0, 260.0, 300.0],
  "num_candidates": 8,
  "num_matches": 4,
  "pic_labels": [
    {
      "category": "similar_object",
      "score": 0.82,
      "sam_score": 0.82,
      "similarity_score": 0.82,
      "combined_score": 0.82,
      "bnd_points": [120.0, 80.0, 260.0, 300.0],
      "polygon_points": [[121.0, 81.0], [360.0, 90.0]],
      "mask_area": 32000,
      "is_reference_overlap": true
    }
  ],
  "reference_result_image": "result_20260521_101200_000001.jpg",
  "result_image": "result_20260521_101200_000002.jpg"
}
```

### 3.3 Feature Match 识别

请求和跨图识别一致，`similar_mode` 填 `feature_match`：

```json
{
  "pic_id": "feature-001",
  "reference_image_base64": "data:image/jpeg;base64,/9j/...",
  "query_image_base64": "data:image/jpeg;base64,/9j/...",
  "reference_bnd_points": [120, 80, 260, 300],
  "top_k": 5,
  "sam_threshold": 0.6,
  "similarity_threshold": 0.6,
  "similar_mode": "feature_match"
}
```

该模式会返回 `profile`，包含 reference prompt 编码、query grounding 和 NMS 后候选数量，便于排查慢请求。

说明：

- 当前 multi visual prompt 链路里，负样例会直接编码成 SAM3 底层 box prompt 的 negative label，并和正样例一起进入同一次 grounding。
- 为兼容旧响应，`profile` 中仍保留 `negative_grounding_forward_ms`、`negative_filter_candidates`、`suppressed_by_negative_samples` 等字段；在当前底层 negative box label 模式下，这些字段通常为 `0`，不再表示单独的负样例二次检索流程。

### 3.4 URL 样例图单张标注

用途：上层服务已经有远程图片路径，不希望把样例图和待标注图转成 Base64。该接口由 SAM3 服务直接下载 `sample_url` 和 `query_image_url`。如果没有正样例，也可以只传 `prompt` 走文本识别；此时 `sample_url` 可省略。

`POST /v1/similar-object-segmentations/by-url`

请求头：

```http
Content-Type: application/json
Authorization: Bearer <api_key>
```

请求体：

```json
{
  "pic_id": "image-001",
  "download_url": "http://192.168.100.118:8092",
  "sample_url": "/group1/default/path/sample.txt",
  "prompt": "#人#安全帽",
  "prompt_category_map": {"person": "人", "helmet": "安全帽"},
  "query_image_url": "/group1/default/path/image.jpg",
  "top_k": 5,
  "sam_threshold": 0.6,
  "similarity_threshold": 0.6,
  "nms_iou": 0.45,
  "polygon_simplify_epsilon": 2.0,
  "return_result_image": false
}
```

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pic_id` | 是 | string | 客户端图片 ID |
| `download_url` | 是 | string | 下载服务根地址。相对路径会拼接成 `{download_url}/{path}` |
| `sample_url` | 否 | string | 样例标注文件路径或完整 HTTP URL；有正样例时使用 |
| `prompt` | 否 | string | 顶层文本 prompt；当 `sample_url` 没有正样例或完全省略时必填，仅支持 `#` 分隔，例如 `#人#猫#桌子`；分隔符两侧允许换行或空白，中文会自动翻译。与样例 `label_id` 同类别（兼容中英文名称）时，会融合为该样例类别的文本条件；类别不同时，会额外创建独立的纯文本检测组并与样例结果一起返回。 |
| `prompt_category_map` | 否 | object | 业务类别到模型文字 prompt 的映射，例如 `{"person":"人"}`。样例模式推荐传入；服务端按键与 `label_id` 匹配，按值执行文字检测，避免翻译结果差异导致同一业务类别被拆成多个分组。 |
| `query_image_url` | 是 | string | 待标注图片路径或完整 HTTP URL |
| `top_k` | 否 | integer | 每个类别最多保留结果数，默认 `5`，范围 `1-50` |
| `sam_threshold` | 否 | number | SAM3 grounding 分数阈值，默认 `0.6` |
| `similarity_threshold` | 否 | number | 兼容字段；当前不执行余弦相似度过滤 |
| `nms_iou` | 否 | number | 同类别最终 NMS 阈值，默认 `0.45`；当较小框有至少 `90%` 被同类别较大框覆盖时，也会按重框抑制 |
| `polygon_simplify_epsilon` | 否 | number | 多边形简化参数，默认 `2.0` |
| `return_result_image` | 否 | boolean | 是否生成可视化结果图；批量场景建议保持 `false` |

响应示例：

```json
{
  "model": "sam3-tensorrt",
  "pic_id": "image-001",
  "success": true,
  "similar_mode": "multi_visual_prompt_url",
  "top_k": 5,
  "top_k_scope": "per_category",
  "sam_threshold": 0.6,
  "nms_iou": 0.45,
  "num_samples": 4,
  "num_positive_samples": 1,
  "num_negative_samples": 3,
  "num_groups": 1,
  "num_candidates": 12,
  "num_matches": 2,
  "category_counts": {
    "person": 2
  },
  "pic_labels": [
    {
      "category": "person",
      "sample_id": "person_0001",
      "sample_ids": ["person_0001"],
      "source_image_id": "0000000000000018",
      "source_image_ids": ["0000000000000018"],
      "score": 0.713421,
      "sam_score": 0.713421,
      "similarity_score": 0.713421,
      "combined_score": 0.713421,
      "coarse_similarity": 0.713421,
      "bnd_points": [520.0, 220.0, 80.0, 160.0],
      "polygon_points": [[521.0, 221.0], [600.0, 224.0]],
      "mask_area": 6700
    }
  ],
  "result_image": null,
  "created": 1779340000,
  "processing_time_ms": 1800,
  "profile": {
    "sample_cache_hit": false,
    "reference_feature_reuse_enabled": true,
    "reference_feature_extract_count": 1,
    "reference_prompt_encode_ms": 96,
    "grounding_forward_ms": 420
  }
}
```

### 3.5 URL 样例图批量任务

用途：一次任务包含多张样例图和大量待标注图片。该接口异步执行，避免一个 HTTP 请求长时间阻塞，也避免一次性传输大量 Base64。若没有正样例，也可以只传 `prompt`，此时 `sample_url` 可省略。

创建任务：

`POST /v1/similar-object-segmentations/tasks`

请求体：

```json
{
  "task_id": "BATCH-LM-001",
  "download_url": "http://192.168.100.118:8092",
  "data_type": 0,
  "data_url": "/group1/default/path/images.txt",
  "sample_url": "/group1/default/path/sample.txt",
  "prompt": "#人#安全帽",
  "prompt_category_map": {"person": "人", "helmet": "安全帽"},
  "infer_batch_size": 16,
  "frame_time": 25,
  "top_k": 5,
  "sam_threshold": 0.6,
  "similarity_threshold": 0.6,
  "nms_iou": 0.45,
  "polygon_simplify_epsilon": 2.0,
  "return_result_image": false,
  "result_ttl_seconds": 86400
}
```

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `task_id` | 是 | string | 任务 ID；重复 running 任务会返回错误 |
| `download_url` | 是 | string | 下载服务根地址 |
| `data_type` | 否 | integer | `0` 表示图片清单；非 `0` 表示视频清单 |
| `data_url` | 是 | string | 待标注图片/视频清单路径或完整 HTTP URL |
| `sample_url` | 否 | string | 样例标注文件路径或完整 HTTP URL；有正样例时使用 |
| `prompt` | 否 | string | 顶层文本 prompt；当 `sample_url` 没有正样例或完全省略时必填，仅支持 `#` 分隔，例如 `#人#猫#桌子`；分隔符两侧允许换行或空白，中文会自动翻译。与样例 `label_id` 同类别（兼容中英文名称）时，会融合为该样例类别的文本条件；类别不同时，会额外创建独立的纯文本检测组并与样例结果一起返回。 |
| `prompt_category_map` | 否 | object | 业务类别到模型文字 prompt 的映射，例如 `{"person":"人"}`。样例模式推荐传入；服务端按键与 `label_id` 匹配，按值执行文字检测，避免翻译结果差异导致同一业务类别被拆成多个分组。 |
| `infer_batch_size` | 否 | integer | 预留分批参数，默认 `16`，范围 `1-64` |
| `frame_time` | 否 | integer | 视频抽帧间隔，按帧数计；`0` 表示逐帧，默认 `1` |
| `top_k` | 否 | integer | 每个类别最多保留结果数，默认 `5` |
| `sam_threshold` | 否 | number | SAM3 grounding 分数阈值，默认 `0.6` |
| `similarity_threshold` | 否 | number | 兼容字段；当前不执行余弦相似度过滤 |
| `nms_iou` | 否 | number | 同类别最终 NMS 阈值，默认 `0.45`；当较小框有至少 `90%` 被同类别较大框覆盖时，也会按重框抑制 |
| `polygon_simplify_epsilon` | 否 | number | 多边形简化参数，默认 `2.0` |
| `return_result_image` | 否 | boolean | 是否生成可视化结果图；5000 张批量时建议 `false` |
| `result_ttl_seconds` | 否 | integer | 完成/失败/取消后结果保留时间，默认 `86400` |

创建任务响应：

```json
{
  "success": true,
  "task_id": "BATCH-LM-001",
  "status": "pending",
  "total": 0,
  "processed": 0,
  "success_count": 0,
  "fail_count": 0,
  "message": "pending",
  "created": 1779340000,
  "updated": 1779340000
}
```

查询任务状态：

`GET /v1/similar-object-segmentations/tasks/{task_id}`

响应示例：

```json
{
  "success": true,
  "task_id": "BATCH-LM-001",
  "status": "running",
  "data_type": 1,
  "total": 5000,
  "processed": 640,
  "success_count": 632,
  "fail_count": 8,
  "videos_total": 4,
  "videos_processed": 1,
  "current_video_id": "video-002",
  "current_video_name": "video-002.mp4",
  "current_frame_num": 325,
  "current_total_frames": 1200,
  "frame_interval": 25,
  "message": "running",
  "created": 1779340000,
  "updated": 1779340300
}
```

`status` 取值：

| status | 说明 |
| --- | --- |
| `pending` | 任务已创建，等待执行 |
| `running` | 任务执行中 |
| `completed` | 任务完成 |
| `failed` | 任务整体失败 |
| `cancelled` | 任务已取消 |

分页获取结果 / Long Polling：

`GET /v1/similar-object-segmentations/tasks/{task_id}/results?offset=0&limit=50`

也可以使用 long polling：

`GET /v1/similar-object-segmentations/tasks/{task_id}/results?offset=0&limit=50&wait_timeout=10`

字段说明：

| 参数 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `offset` | 否 | integer | 已消费结果数量，默认 `0` |
| `limit` | 否 | integer | 单页最多返回数量，默认 `50`，最大 `500` |
| `wait_timeout` | 否 | number | Long polling 等待秒数，默认 `0`；最大按服务端限制为 `60` 秒 |

Long polling 语义：

- 如果 `offset` 后已经有新结果，立即返回。
- 如果暂时没有新结果且任务仍是 `pending/running`，服务端最多等待 `wait_timeout` 秒。
- 等待期间一旦产生新结果，立即返回。
- 如果等待超时仍无新结果，返回 `items=[]`，并带上当前任务进度。
- 如果任务进入 `completed/failed/cancelled`，立即返回当前剩余结果和最终状态。

响应示例：

```json
{
  "success": true,
  "task_id": "BATCH-LM-001",
  "status": "running",
  "data_type": 0,
  "processed": 640,
  "success_count": 632,
  "fail_count": 8,
  "message": "running",
  "offset": 0,
  "limit": 50,
  "total": 5000,
  "result_total": 640,
  "items": [
    {
      "pic_id": "image-001",
      "status": 1,
      "message": "标注成功",
      "pic_labels": [
        {
          "category": "person",
          "sample_ids": ["person_0001"],
          "source_image_ids": ["0000000000000018"],
          "score": 0.713421,
          "bnd_points": [520.0, 220.0, 80.0, 160.0],
          "polygon_points": [[521.0, 221.0], [600.0, 224.0]]
        }
      ]
    }
  ]
}
```

当 `data_type != 0` 时，`items` 中每一项表示一个抽样帧，除 `pic_labels` 外还会附带：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `timestamp` | number | 当前帧时间戳，单位毫秒 |
| `frame_num` | integer | 帧序号，从 `0` 开始 |
| `fps` | number | 视频 FPS |
| `video_id` | string | 视频 ID |
| `video_name` | string | 视频文件名 |
| `frame_image_base64` | string | 当前帧 JPEG Base64；仅当该帧识别到目标时返回，供上层回传 `pic_url` 使用 |

上层服务轮询策略：

- 上层创建任务成功后，直接调用 long polling results 接口拉取进度和增量结果。
- 当前训练服务上层默认使用 `wait_timeout=10`，与训练服务心跳进度上报节奏保持一致。
- 上层字段 `sam_task_poll_interval` 可覆盖默认 long polling 等待时间；例如传 `10.0` 表示最多等待 10 秒。
- 每次响应都会包含 `processed/total/status`，上层可直接同步上报现有训练服务进度。
- 上层用自己维护的 `offset` 调用 `GET /tasks/{task_id}/results?offset=<已消费数量>&limit=<batch_size>&wait_timeout=10` 拉取新结果。
- `items=[]` 表示本次等待期间没有新结果，不是错误；上层收到响应后继续发起下一次 long polling。
- 当状态为 `completed`、`failed`、`cancelled` 时，上层会最后再拉一次 results，避免遗漏最后一批结果。

取消任务：

`DELETE /v1/similar-object-segmentations/tasks/{task_id}`

响应示例：

```json
{
  "success": true,
  "task_id": "BATCH-LM-001",
  "status": "cancelled",
  "message": "cancel requested"
}
```

### 3.6 `sample_url` 文件格式

`sample_url` 内容必须是 JSON 数组。样例图可以是相对路径，也可以是完整 HTTP URL。

```json
[
  {
    "label_id": "person",
    "label_sample_data": [
      {
        "image_url": "/group1/default/20251217/15/25/8/0000000000000018.jpg",
        "image_id": "sample-image-001",
        "image_mark": [
          {
            "mark_info": "{\"rotation\":0,\"x\":408.068,\"width\":56.644,\"y\":361.828,\"height\":236.98}",
            "sample_type": "1"
          },
          {
            "mark_info": [435.812, 339.864, 83.188, 201.144],
            "sample_type": "0"
          }
        ]
      }
    ]
  }
]
```

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `label_id` | 是 | string | 类别名称，会作为输出 `pic_labels[].category` |
| `label_sample_data[].image_url` | 是 | string | 样例图路径或完整 HTTP URL |
| `label_sample_data[].image_id` | 否 | string | 样例图 ID；缺省时从文件名生成 |
| `image_mark[].mark_info` | 是 | string/object/array | 样例框信息。推荐传包含 `x/y/width/height` 的 object 或 object-string，也兼容 `[x, y, width, height]` 数组 |
| `image_mark[].sample_type` | 否 | string/integer | `1/"1"` 表示正样本，`0/"0"` 表示负样本；也兼容 `positive/negative`；缺省按正样本处理 |

限制和规则：

- 最多支持 `300` 张样例图。
- 最多支持 `2000` 个样例实例。
- `sample_url` 可以全部是负样本；但此时请求级 `prompt` 必填。
- `sample_type=0/negative` 的样例会直接映射为 SAM3 底层 negative box label，与正样例一起编码成 visual prompt，而不是单独走一条负样例检索再做 IoU 过滤。
- `rotation` 当前接受但不参与计算；坐标仍按水平矩形 `[x,y,width,height]` 处理。
- `mark_info` 推荐使用 object 或 object-string；为兼容旧数据，当前也支持 `[x, y, width, height]` 数组格式。
- 样例图会按图片聚合，同一张样例图只提取一次特征。
- 样例 prompt embedding 会缓存；相同 `download_url + sample_url + sample_url内容 + top_k + prompt_category_map` 命中缓存时，不重复下载样例图和编码样例 prompt。
- 当 `prompt_category_map` 的同一业务类别映射多个文字条件（如 `{"bottle": "水杯，显示器"}`）时，服务会将其拆为独立的文字检测组；每个检测组都会复用该业务类别的全部正负样例，最终结果仍返回为原业务类别 `bottle`。

### 3.7 `data_url` 文件格式

`data_url` 支持 JSON 数组：

```json
[
  {
    "image_id": "image-001",
    "image_url": "/group1/default/path/image001.jpg"
  },
  {
    "image_id": "image-002",
    "image_url": "/group1/default/path/image002.jpg"
  }
]
```

也支持纯文本格式，每行一个 URL：

```text
/group1/default/path/image001.jpg
/group1/default/path/image002.jpg
```

兼容旧的 `image_id=remote_path` 格式：

```text
image-001=/group1/default/path/image001.jpg
image-002=/group1/default/path/image002.jpg
```

限制：

- URL 批量任务支持图片清单和视频清单；视频模式下会由服务端下载视频、按 `frame_time` 抽帧后再执行样例标注。
- 单任务最多 `5000` 张待标注图片。
- 每张图片大小受服务端 `SAM3_MAX_IMAGE_BYTES` 限制，默认 `20MB`。

## 4. Multipart 相似识别接口

前端页面统一使用 `POST /multi-similar-detect`。普通单样例就是 `sample_meta` 里只有一个正样本实例；继续添加标签、样例或实例即可扩展为多类别、多样例识别。现在也支持：

- 纯文本 prompt，无需上传 `sample_file`
- 文本 prompt + 仅负样例
- 正样例 + 可选文本 prompt + 可选负样例

### 4.1 跨图 Feature Match 识别

```bash
curl -X POST 'http://192.168.100.25:8006/multi-similar-detect' \
  -F 'sample_file=@/path/to/ref.jpg' \
  -F 'query_file=@/path/to/query.jpg' \
  -F 'sample_meta=[{"file_index":0,"sample_type":"positive","category":"安全帽","reference_bnd_points":[120,80,260,300],"prompt":"红色安全帽"}]' \
  -F 'top_k=5' \
  -F 'sam_threshold=0.6'
```

多张目标图可重复传 `query_file`：

```bash
curl -X POST 'http://192.168.100.25:8006/multi-similar-detect' \
  -F 'sample_file=@/path/to/ref.jpg' \
  -F 'query_file=@/path/to/query1.jpg' \
  -F 'query_file=@/path/to/query2.jpg' \
  -F 'sample_meta=[{"file_index":0,"sample_type":"positive","category":"目标","reference_bnd_points":[120,80,260,300]}]'
```

纯文本 prompt：

```bash
curl -X POST 'http://192.168.100.25:8006/multi-similar-detect' \
  -F 'query_file=@/path/to/query.jpg' \
  -F 'prompt=#人#安全帽' \
  -F 'sample_meta=[]'
```

### 4.2 兼容旧同图拉框接口

```bash
curl -X POST 'http://192.168.100.25:8006/similar-detect' \
  -F 'reference_file=@/path/to/image.jpg' \
  -F 'reference_bnd_points=120,80,260,300' \
  -F 'top_k=5' \
  -F 'sam_threshold=0.6' \
  -F 'similarity_threshold=0.6' \
  -F 'similar_mode=same_image_prompt'
```

该接口用于旧页面或旧脚本；新前端不再提供单独模式开关。

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `sample_file` | 条件必填 | file/list | 样例图；传了 `sample_meta` 样例实例时必填，可重复传多张 |
| `query_file` | 是 | file/list | 待识别图；可重复传多张 |
| `sample_meta` | 否 | string | JSON 数组，描述每个样例实例的 `file_index`、正负样本、类别和框；纯文本模式可传 `[]` 或省略 |
| `prompt` | 否 | string | 顶层文本 prompt；当没有正样例时必填，仅支持 `#` 分隔，例如 `#人#猫#桌子`；分隔符两侧允许换行或空白，中文会自动翻译 |
| `top_k` | 否 | integer | 默认 `5` |
| `sam_threshold` | 否 | number | SAM3 grounding 分数阈值，默认 `0.6` |
| `similarity_threshold` | 否 | number | 兼容旧客户端字段；当前不再执行余弦相似度过滤 |
| `polygon_simplify_epsilon` | 否 | number | 默认 `2.0` |
| `pic_id` | 否 | string | 图片 ID |

## 5. Base64 和 URL 接口选择

推荐：

- 小规模调试、旧客户端：使用 Base64 JSON 接口 `/v1/similar-object-segmentations` 或 multipart `/multi-similar-detect`。
- 上层服务已有远程图片路径，且单张待标注图：使用 `/v1/similar-object-segmentations/by-url`。
- 1-300 张样例图、1-5000 张待标注图：使用 `/v1/similar-object-segmentations/tasks`。
- 不建议把大量图片放入 `query_image_base64_list`，Base64 会膨胀请求体并增加内存峰值。

## 6. 常见错误

### 400

常见原因：

- 图片无法解析。
- `reference_bnd_points` 不是 4 个数字。
- 非 `same_image_prompt` 模式没有传 `query_image_base64` / `query_file`。
- 框宽高小于等于 0。
- URL 样例标注缺少 `download_url`、`query_image_url` 或 `data_url`。
- `sample_url` 不是合法 JSON 数组。
- 既没有正样例，也没有可用 `prompt`。
- `mark_info` 缺少 `x/y/width/height`，或框宽高小于等于 0。
- URL 批量任务 `data_type` 不是 `0`。
- `task_id` 对应的任务正在运行，重复创建同名任务。

### 401

`/v1/*` 接口未传 API Key，或 API Key 过期/无效。

### 404

常见原因：

- 查询或取消不存在的 URL 批量任务。

### 500

模型推理失败。建议先查看服务日志，重点检查 CUDA/显存、`SAM3_TRT_ENGINE_DIR`、engine 与当前 GPU compute capability/TensorRT 版本是否匹配。

## 7. 推荐调用选择

- 只按类别找目标：用 `/v1/segmentations`，也就是 prompt 文本识别。
- 用户已经画了一个框，只需要分割这个框内目标：用 `/v1/box-segmentations`。
- 示例图 A + 目标图 B 小规模调试：推荐使用 `/multi-similar-detect`。
- 远程样例文件 + 单张远程目标图：推荐使用 `/v1/similar-object-segmentations/by-url`。
- 远程样例文件 + 批量远程目标图：推荐使用 `/v1/similar-object-segmentations/tasks`。
- 希望统一走 multi-similar 链路但手上没有正样例：可直接传 `prompt`，负样例可选。
- 普通单样例 multipart 调试：`sample_meta` 只放一个正样本实例。
- 多类别/多样例 multipart 调试：继续追加标签、样例图和实例框。
- 同图拉框识别：旧接口仍可用 `similar_mode=same_image_prompt`。

## 8. EISeg TCP 兼容服务（6063）

生产入口为 `start_offline.sh` → `sam_app.py`，一个进程同时提供 HTTP `8006` 和 TCP `6063`，
不启动 Gradio `7861`。FastAPI 生命周期负责 TCP 启动和关闭。
HTTP runtime 持有一个 `shared-image-encoder.engine`，TCP 后端引用同一 encoder 对象、
execution context 和锁；该图包含一个视觉主干及检测/交互两个 neck，输出各通道需要的特征。
HTTP 继续使用文字/几何/检测解码头，TCP 额外使用首点和 refinement 解码头，
不会再加载旧 `vision-encoder.engine` 或 `interactive-image-encoder.engine`。
模型常驻显存共享，但各连接的特征和点击历史仍独立占用内存。
原 EISeg `6062` 服务可以独立运行，前端将目标端口改为 `6063` 或配置后的 TCP 端口。

正/负点击直接传给原生 point decoder，后续点击传入上一轮低分辨率 logits 做 refinement。
协议保持 EISeg 格式，实际模型为 SAM3，分割结果仍需用业务图片评估。

### 启动

在已有 SAM3 TensorRT 环境中执行，不安装或升级依赖：

```bash
cd SAM3_Apply
# 后台启动 HTTP + TCP
bash start_offline.sh --background
# 前台启动，并可覆盖 TCP 端口
SAM3_FOREGROUND=1 SAM3_TCP_PORT=6063 bash start_offline.sh
```

当前容器示例（默认前台运行）：

```bash
docker exec -it -e SAM3_FOREGROUND=1 -e SAM3_TCP_PORT=6063 \
  sam3_trt_dev_runtime bash /sam3/start_offline.sh
```

启动变量：`SAM3_TCP_HOST` 默认 `0.0.0.0`，`SAM3_TCP_PORT` 默认 `6063`，
HTTP 使用 `SAM3_HOST/SAM3_PORT`，默认 `0.0.0.0:8006`；TCP 端口也可直接修改
`start_offline.sh` 中的 `SAM3_TCP_PORT` 默认值。原生解码头目录使用 `SAM3_NATIVE_ENGINE_DIR`，
默认 `${SAM3_TRT_ENGINE_DIR}/native_interactive/`；共享图位于 `${SAM3_TRT_ENGINE_DIR}/shared-image-encoder.engine`。
使用 `--background` 时，脚本会覆盖继承的 `SAM3_FOREGROUND=1`，并等待 HTTP `/health` 和 TCP 端口均可连接后才返回；等待上限为 180 秒，可用 `SAM3_STARTUP_TIMEOUT_SECONDS` 调整。HTTP 与 TCP 在同一后台进程中运行，不单独启动第二个模型进程。
构建命令和 FP32/关闭 TF32 的精度要求见 `README_DEPLOY.md`。
需要沿用旧前端端口时，可设置 `SAM3_TCP_PORT=6062`，该端口必须空闲。
相对图片 URL 使用 `SAM3_TCP_IMAGE_BASE_URL`，默认保留旧地址 `http://192.168.100.25:8092/`。
TCP 沿用旧服务的无 API Key 协议；HTTP `/v1/*` 的鉴权规则不变。

### 浏览器 TCP 测试面板

启动后访问 `http://<服务地址>:8006/`，页面中的 **TCP 交互分割测试（SAM3 / EISeg）** 支持：

1. 设置 TCP 端口，默认 `6063`，点击“连接 TCP”。测试原 EISeg 时断开连接、把端口改为 `6062` 再连接；自定义端口同样可用。页面的端口仅选择测试目标，不修改服务监听配置。
2. 可查询模型 `8004` 或初始化模型 `8000`；连接后已默认选择槽位 0，无需额外初始化。SAM3 的两个槽位使用同一模型。
3. 选择本地图片，点击“载入图片（8001）”。页面仅在内存中读取 Base64，每次载入使用新的 `image_id`，避免原 EISeg 按 ID 命中旧图片缓存。
4. 在图片上左键添加当前选择的正/负点，右键固定发送负点；也可输入原图 `[x,y]` 后发送。缩放显示时自动转换为原图坐标。
5. 测试撤销 `3`、完成 `4`、重置 `5`；显示成功点击、多边形和外接框，以及最近一次请求、完整 JSON 响应和往返耗时。请求区省略图片 Base64 内容。

一次只发送一条请求，失败点击不会加入页面历史；完成后保留最终多边形预览，可开始下一个目标。
重新连接、初始化模型、切换图片后需要重新载入 TCP 图片。单条请求等待超过 120 秒时断开连接，避免把迟到响应用于下一条请求。

浏览器不支持直接连接原始 TCP，页面通过同一 HTTP 服务上的 WebSocket 桥接：

```text
ws://<HTTP服务地址>:8006/ui/tcp-test?port=6063
```

HTTPS 页面自动使用 `wss://`，反向代理需要转发 WebSocket Upgrade，并保留 Host。
桥接固定从 **SAM3 HTTP 服务所在的机器/容器** 连接 `127.0.0.1`；端口可调，目标主机不能由浏览器修改。
远端浏览器的 `127.0.0.1` 仍指服务端，不是浏览器机器。桥接路由与现有 `/ui/*` 页面接口一样无 API Key，浏览器请求检查同源 Origin；仅接受 `8000/8001/8002/8004` JSON 命令，自动追加/移除 `##@##`。
该调试页应沿用现有管理页面的网络访问限制。

桥接消息格式：

- 建连后：`{"type":"connected","host":"127.0.0.1","port":6063,"max_image_bytes":20971520}`。
- 浏览器发送原协议 JSON，例如 `{"cmd":8004,"data":{}}`，无需添加 TCP 分隔符。
- TCP 回复：`{"type":"response","response":{"cmd":9004,"data":[...]}}`，原始 `900x` 内容不改写。
- 连接、解析或传输错误：`{"type":"error","message":"..."}`，随后断开；业务 `status=0` 保留在原始响应中。

每个 WebSocket 对应一个到 loopback 端口的真实 TCP 连接，保留整套交互状态；页面断开时关闭该 TCP 连接。
连接超时 10 秒，TCP 无回复/空闲上限 600 秒；消息大小沿用 TCP 限制，`sam_app.py` 同步设置 uvicorn WebSocket 大小上限。
桥接本身不加载模型或保存图片，测试 SAM3 `6063` 继续复用共享模型；选择外部 EISeg `6062` 后，其模型及 `pics/` 图片缓存行为仍由原 EISeg 服务决定。

前端/桥接自检：

```bash
python -m unittest discover -s tests -p 'test_tcp_bridge.py' -v
node --test tests/test_tcp_frontend.cjs
```

2026-10-23：完整 Python 测试 35 项和前端状态/坐标测试 3 项通过。通过运行中的 HTTP `8006` 桥接
真实 SAM3 `6063`，`8000/8001/8002/8004`、五种点击操作和 17 MiB WebSocket 消息均通过，
`uploads/`、`results/` 未新增文件；切换 `6062` 成功查询原 EISeg 模型。此次未向原 EISeg 上传图片，
未重启已有服务；前端使用无依赖 DOM 模拟检查，尚未进行真实浏览器画面和鼠标操作验收。

### 消息与命令

每条消息为 UTF-8 JSON，结尾追加 ASCII 分隔符 `##@##`，无换行和长度头。
服务支持 TCP 拆包、粘包；连接建立后默认可用，无需先发模型初始化命令。

| 请求 → 响应 | 请求 `data` | 行为 |
| --- | --- | --- |
| `8000 → 9000` | `{"model":0,"use_gpu":false}` | 接受旧模型槽位 `0/1`，均使用同一 SAM3；重置本连接图片和点击状态，需要重新发 `8001`。`use_gpu` 仅兼容读取，实际始终使用 GPU。 |
| `8001 → 9001` | `{"image_id":"demo","image_url":"...","image_base64":"..."}` | URL 非空时优先使用 URL，否则使用 Base64；支持完整 HTTP(S) URL、相对 URL、纯 Base64 和 data URL。成功后清空点击历史。 |
| `8002 → 9002` | `{"image_id":"demo","click_type":1,"coordinate":[100,120]}` | 原图像素坐标 `[x,y]`。`1` 正点击，`2` 负点击，`3` 撤销，`4` 完成当前目标，`5` 重置当前目标；`3/4/5` 无需坐标。 |
| `8004 → 9004` | 可省略 | 返回旧格式模型列表，保留 `Normal model`、`Person model` 名称和 `model/model_name/use_gpu` 字段，两个槽位均指向 SAM3。 |

`9000` 返回 `status/message`，`9001` 返回 `image_id/status/message`，`9004.data` 是模型数组。
`9002` 继续保留旧协议拼写：

```json
{
  "cmd": 9002,
  "data": {
    "status": 1,
    "image_id": "demo",
    "click_type": 1,
    "ploygon": [[[90,100],[130,100],[130,150],[90,150]]],
    "bnd_boxs": [[90,100,41,51]],
    "min_boxs": [[85,105,50,40,90.0]],
    "message": "Click processed successfully"
  }
}
```

`ploygon` 为多边形列表，每个点是 `[x,y]`，孔洞通过往返桥接边并入外轮廓；
`bnd_boxs` 是 OpenCV `boundingRect` 的整数 `[x,y,w,h]`；
`min_boxs` 沿用旧实现的 `[center_x-width/2,center_y-height/2,width,height,angle]` 整数化和角度规则，
不是旋转矩形某个角点的坐标。三组列表按多边形一一对应。

TCP 输出轮廓在返回前使用 OpenCV `approxPolyDP` 做顶点简化，SAM 的原始 mask、logits 和点击细化不变。
默认 `SAM3_TCP_POLYGON_EPSILON=2.0`，单位为原图像素，和 HTTP polygon epsilon 默认值一致。
数值越大返回点越少，但边界偏差也可能增大；`0` 关闭该额外简化（原有 TC89 contour 压缩仍会生效）。建议从 `3` 或 `4` 开始比较，
细小边缘被削弱时调低。范围为 `0–50`，重启 `start_offline.sh` 后生效；可直接设置环境变量或编辑脚本默认值。

### 交互和运行限制

- 每个连接独立保存图片特征、轮廓和低分辨率 logits 历史；断开或空闲 600 秒后释放引用。每个共享 TensorRT engine 的执行通过锁保护。
- 首次不能负点击；撤销到空、完成或重置之后也必须从正点击开始。
- `4` 返回最后一次预览，并清空当前目标的点击历史；再次完成返回空列表。`5` 返回空列表并清空历史。图片特征仍可用于下一个目标。
- 首点从三个候选中按模型置信度选择，后续使用 refinement decoder；推理失败不写入点击和 logits 历史。撤销恢复上一轮轮廓及 logits，撤销到空后再次使用首点 decoder。
- 输出 mask 按 8 连通区域去除像素面积小于最大区域 1% 的碎片；包含任一正点击的区域始终保留，孔洞保留。TCP 轮廓和浏览器预览/二值 mask 使用同一清理结果，refinement 继续使用模型原始 logits。此规则只去除独立小区域，无法去除与主区域相连的毛刺。
- TCP 返回的多边形点数由 `SAM3_TCP_POLYGON_EPSILON` 控制；此参数只简化传输/显示用的坐标，不会改变模型 mask 或后续点击预测。
- 原生 decoder 每个目标最多 32 次成功点击。超过限制会返回 `status=0`，不静默丢弃旧点击。
- 图片载入失败会使当前图片失效，避免后续点击误标上一张图。图片只保存在内存，不读写原 EISeg 的 `pics/` 缓存。
- 图片二进制大小上限沿用 `SAM3_MAX_IMAGE_BYTES`（默认 20 MiB），消息上限在此基础上包含 Base64 膨胀和 64 KiB JSON 余量。URL 下载超时 30 秒。
- 业务错误使用对应 `900x` 响应和 `status=0/message`；点击失败的几何字段均为空。无效 JSON/UTF-8 返回 `cmd=0` 错误并继续读取；超大帧返回 `cmd=0` 后断开连接。
- 生产入口只有一个模型实例和一个共享图像编码器，HTTP/TCP 使用不同任务解码头；每个 TCP 会话特征约 20.25 MiB，连接数增加仍会增加显存。
- TCP 图片处理不调用 HTTP 上传/结果保存，也不触发文字翻译。HTTP 文字接口继续按原配置使用离线 Argos。
- TCP 端口占用时 HTTP 应用启动失败；停止 HTTP 同时关闭 TCP 监听。服务仅使用单个 uvicorn worker。
- `app_tensorrt.py`、`ann_tcp_6063.py` 保留为旧独立调试入口，生产部署不要同时启动它们。

无 GPU 协议自检：在 `SAM3_Apply` 下执行 `python -m unittest discover -s tests -p 'test_eiseg_tcp.py' -v`。

2026-10-23 合并服务验收：26 项单测通过；在 `sam3_trt_dev_runtime` 中通过 `start_offline.sh`
以临时 HTTP `18006` / TCP `16063` 启动，真实 `truck.jpg` 的文字检测、框分割、TCP 正/负点、
撤销后细化、完成、重置和 HTTP/TCP 同时请求均通过，`7861` 未监听，停止进程后两个端口均关闭。
该次检查只新增一个 GPU 进程，请求后的该进程显存为 `7032 MiB`；这是单次测试值，包含缓存和临时内存池，
不代表固定占用或并发上限。共享 encoder 与旧独立 FP32 交互 encoder 的 mask IoU：首点 `0.999824`、
追加负点 refinement `0.999967`，测试未修改依赖或覆盖两个旧图像 encoder。

此前原生 TCP 接入验收：在 `sam3_trt_dev_runtime` 容器中用 TensorRT 10.8 原生 engines 和 `truck.jpg`
完成真实 TCP 检查：`8000/8001/8002/8004`、正/负点、撤销后细化、完成、重置及下一目标均通过。
测试同时启动临时 Gradio `127.0.0.1:17861` 和随机本地 TCP 端口，共享后端，结束后关闭；
TCP 全流程禁止调用 PIL/OpenCV 图片保存函数。TCP 多边形栅格化与 Gradio 回调 mask 的
IoU 为 `0.9984`。该检查未使用实际 EISeg 前端，不代表业务图片质量验收。

## 9. 原生点击 Demo 与并发显存压测（7861）

本节记录旧 Gradio 调试工具及历史压测方法。当前生产入口 `start_offline.sh` 只启动 HTTP/TCP，
不导入 Gradio，也不监听 `7861`；共享 encoder 的部署说明见第 8 节和 `README_DEPLOY.md`。

`app_tensorrt.py` 同时提供 Gradio `7861` 和 EISeg TCP `6063`，使用 `engines/native_interactive/`
中的 image encoder、first-click decoder、refinement decoder；两种通道共享后端，均使用原生点击。
它不经过 `8006` HTTP 服务。`SAM3_NATIVE_ENGINE_DIR` 可指定 engine 目录，TCP 协议见第 8 节。

默认地址为 `http://127.0.0.1:7861`。浏览器上传图片后，左键正点、右键负点；
后续点击使用当前会话的上一轮低分辨率 logits refinement。单个目标最多 32 个点。

### Demo 分步骤耗时

`app_tensorrt.py` 和 `tools/sam3/demo/app_tensorrt.py` 均输出毫秒级计时。
终端日志格式为 `[SAM3][TensorRT] <步骤>耗时：<毫秒> ms`，具体包括：

- 首次使用时分别加载图像编码器、首点解码器、细化解码器 engine，以及后端初始化合计。
- 图片加载：RGB/缩放/归一化预处理、编码器输入传输与缓冲准备、编码器推理、图片编码合计。
- 点击：坐标和上一轮 mask 提示准备、首点或细化解码器输入传输与缓冲准备、解码器推理、输出回传 CPU、mask 后处理、点击推理合计、预览绘制。
- 加载图片、点击分割、撤销点击、清空点击、清除图片各自的总耗时。原有状态文本末尾追加 `，耗时：<毫秒> ms`，输出数量、点击数据和分割结果结构保持一致。

输入拷贝和 TensorRT 执行使用同一 CUDA stream，各阶段结束时等待该 stream 完成；
输出回传使用同步 `.get()`。记录的是该服务端步骤的实际墙钟时间，包含相应 Python/CUDA
调用开销，不是纯 GPU kernel 时间。总耗时包含其中的子步骤，不能将合计与子步骤重复相加。
合并入口 `app_tensorrt.py` 在启动 TCP/Gradio 前初始化 engines，启动耗时单独记录；
直接调用回调或使用 `tools/sam3/demo/app_tensorrt.py` 时，首次图片加载仍可能包含惰性初始化。
后续点击复用已缓存的图像特征。
计时不包含 Gradio 排队、HTTP 上传、网络传输和浏览器渲染。失败步骤也会打印已消耗的时间。

已有运行进程需要重新启动 demo 才会加载计时代码，无须重建 engines。无 GPU 检查命令
（在包含项目现有 Gradio 依赖的环境中，从 `SAM3_Apply` 目录运行）：

```bash
python -m unittest discover -s tests -p 'test_native_click_trt.py' -v
python -m unittest discover -s tests -p 'test_tensorrt_demo.py' -v
```

### 并发显存压测

压测脚本 `tools/benchmark_native_click.py` 通过 `/config` 发现 handler ID，
经 `/gradio_api/upload` 上传文件，使用 `/gradio_api/queue/join` 和
`/gradio_api/queue/data` 发起请求及读取 SSE 结果。点击的 `event_data` 为
`{"x": 原图像素x, "y": 原图像素y, "label": 1或0}`，`1` 为正点，`0` 为负点。
每位模拟用户使用独立的 `session_hash`，保留自己的 `gr.State`，不会将会话状态传到客户端。
脚本会检查状态文本、mask 和点击数，HTTP 200 中的推理失败也计为失败。

在服务所在宿主机或支持 `nvidia-smi` 的 GPU 容器中运行，不需要新增 Python 依赖：

```bash
cd /expdata/givap/givap-train-server-v2.2/src
python SAM3_Apply/tools/benchmark_native_click.py \
  --url http://127.0.0.1:7861 \
  --image tools/sam3/sam3/assets/images/truck.jpg \
  --users 1 4 8 16 32 \
  --clicks 4 --hold-seconds 5 --gpu 0
```

容器中若工作目录为 `/sam3`，使用 `python tools/benchmark_native_click.py`，
并将 `--image` 改为容器内可读的图片路径。远程 URL 可以压测，但显存采样始终来自
**运行脚本机器的本地 GPU**；远程客户端不能据此测量服务端显存。

`--users` 指逐级增加的**总在线会话数**，不是每一级重新新增该数量。
脚本先预热 encoder 和两个 decoder，再保持所有会话，新增用户并发上传/编码，
然后所有用户并发发起点击；每个用户内部的正负点请求按顺序执行。
每级默认每人 4 次点击，保留 5 秒。累计点数将超过 32 时先清空该用户的点击历史。
遇到加载、推理、会话状态或请求错误时停止继续加压，并保存部分结果。
结束后对脚本自己的会话调用 `clear_image`，不重启服务或清空全局显存池。

可选参数：

| 参数 | 用途 |
| --- | --- |
| `--gpu 0` | 本地 `nvidia-smi` 的 GPU index，也可使用 GPU UUID；应选服务实际使用的卡 |
| `--pid 12345` | 额外采样该服务进程的显存；PID 必须取自 `nvidia-smi`，可能与容器内 `ps` 不同 |
| `--interval 0.5` | 显存采样周期，单位秒 |
| `--timeout 180` | 单请求超时，包含排队；增加并发时可适当增加 |
| `--output /path/to/new-directory` | 报告目录；要求目录尚不存在，避免覆盖已有结果 |

默认报告写入当前目录的 `runs/native_click_benchmark_<时间>/`：

- `summary.json`：预热前/后、各级在线人数、总卡显存峰值/保持值、相对预热增量、
  点击 p50/p95 延迟、吞吐及失败数。提供 `--pid` 时还包括服务进程的显存峰值和增量。
- `gpu_memory.csv`：随时间采样的显存、GPU 利用率及阶段；提供 `--pid` 时含服务进程显存。
- `requests.csv`：每用户上传、加载、首点、refinement、清理的耗时和错误详情。

所有延迟均包含网络、排队、图像处理和序列化，不是单独的 TensorRT kernel 时间。
采样得到的显存峰值可能遗漏短于采样周期的峰值。没有提供 `--pid` 时，显存是整卡值，
其他进程释放或分配显存会影响结果，不能直接作为本服务的每用户显存增量。

显存解释：单进程共享 engines 和 execution contexts，推理通过相同的 Gradio
`concurrency_id` 串行排队；在线人数增加会提高排队延迟，而不会给每人加载一份模型。
每会话缓存的三组 FP32 特征合计为
`(256×72×72 + 32×288×288 + 64×144×144)×4 = 20.25 MiB`，
显存池对齐和临时缓冲还会产生额外开销。原图、完整分辨率 mask 历史和 logits 保存在
主机内存，亦会随图片大小、点击数和在线人数增长。`gr.State` TTL 为 1800 秒。
CuPy 显存池可能保留已释放数组的空间，因此清理会话后 `nvidia-smi` 不一定下降，
不能只用清理前后的整卡数字判断内存泄漏。

无 GPU 脚本自检：

```bash
python -m unittest discover -s SAM3_Apply/tests -p 'test_benchmark_native_click.py' -v
```
