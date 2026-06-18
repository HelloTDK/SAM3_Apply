# SAM3 接口文档

服务地址示例：`http://192.168.100.25:8006`

本文档只覆盖当前常用的三类能力：

- prompt 文本识别：按文本类别找目标。
- 画区域 / box 分割：用户给框，SAM3 分割框内目标。
- 示例目标相似识别：包括跨图特征匹配识别、同图拉框识别。

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
  "prompt": "人; 安全帽; 车辆",
  "confidence_threshold": 0.3,
  "polygon_simplify_epsilon": 2.0
}
```

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `pic_id` | 是 | string | 客户端图片 ID |
| `image_base64` | 是 | string | 待识别图片 |
| `prompt` | 是 | string | 类别文本，支持 `;` 或 `,` 分隔；中文会自动翻译成英文 |
| `confidence_threshold` | 否 | number | 置信度阈值，默认 `0.3` |
| `polygon_simplify_epsilon` | 否 | number | 多边形简化参数，默认 `2.0` |

响应示例：

```json
{
  "model": "ultralytics-sam3",
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
  "prompt": "人; 安全帽; 车辆",
  "translated_prompt": "person; helmet; vehicle",
  "was_translated": true,
  "confidence_threshold": 0.3,
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

curl 示例：

```bash
curl -X POST 'http://192.168.100.25:8006/detect' \
  -F 'file=@/path/to/image.jpg' \
  -F 'prompt=人;安全帽' \
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
  "model": "ultralytics-sam3",
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

统一使用 `POST /v1/similar-object-segmentations`，通过 `similar_mode` 切换模式。

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

## 4. Multipart 相似识别接口

前端页面统一使用 `POST /multi-similar-detect`。普通单样例就是 `sample_meta` 里只有一个正样本实例；继续添加标签、样例或实例即可扩展为多类别、多样例识别。

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
| `sample_file` | 是 | file/list | 样例图；可重复传多张 |
| `query_file` | 是 | file/list | 待识别图；可重复传多张 |
| `sample_meta` | 是 | string | JSON 数组，描述每个样例实例的 `file_index`、正负样本、类别和框 |
| `top_k` | 否 | integer | 默认 `5` |
| `sam_threshold` | 否 | number | SAM3 grounding 分数阈值，默认 `0.6` |
| `similarity_threshold` | 否 | number | 兼容旧客户端字段；当前不再执行余弦相似度过滤 |
| `polygon_simplify_epsilon` | 否 | number | 默认 `2.0` |
| `pic_id` | 否 | string | 图片 ID |

## 5. 常见错误

### 400

常见原因：

- 图片无法解析。
- `reference_bnd_points` 不是 4 个数字。
- 非 `same_image_prompt` 模式没有传 `query_image_base64` / `query_file`。
- 框宽高小于等于 0。

### 401

`/v1/*` 接口未传 API Key，或 API Key 过期/无效。

### 500

模型推理失败。建议先查看服务日志，重点看 CUDA、显存、checkpoint 路径和 Ultralytics 版本。

## 6. 推荐调用选择

- 只按类别找目标：用 `/v1/segmentations`，也就是 prompt 文本识别。
- 用户已经画了一个框，只需要分割这个框内目标：用 `/v1/box-segmentations`。
- 示例图 A + 目标图 B 找相似目标：推荐使用 `/multi-similar-detect`。
- 普通单样例：`sample_meta` 只放一个正样本实例。
- 多类别/多样例：继续追加标签、样例图和实例框。
- 同图拉框识别：旧接口仍可用 `similar_mode=same_image_prompt`。
