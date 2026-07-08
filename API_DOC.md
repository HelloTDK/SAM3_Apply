# SAM3 接口文档

服务地址示例：`http://192.168.100.25:8006`

本文档只覆盖当前常用的四类能力：

- prompt 文本识别：按文本类别找目标。
- 画区域 / box 分割：用户给框，SAM3 分割框内目标。
- 示例目标相似识别：包括跨图特征匹配识别、同图拉框识别。
- 样例图标注：通过远程 `sample_url` 和 `data_url` 批量标注图片，避免传输大量 Base64。

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
- 为兼容旧响应，`profile` 中仍保留 `negative_grounding_forward_ms`、`negative_filter_candidates`、`suppressed_by_negative_samples` 等字段；在底层 negative box label 模式下，这些字段通常为 `0`，不再表示单独的负样例二次检索流程。

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
  "prompt": "人; 安全帽",
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
| `prompt` | 否 | string | 顶层文本 prompt；当 `sample_url` 没有正样例或完全省略时必填，支持 `;` / `,` 分隔，中文会自动翻译 |
| `query_image_url` | 是 | string | 待标注图片路径或完整 HTTP URL |
| `top_k` | 否 | integer | 每个类别最多保留结果数，默认 `5`，范围 `1-50` |
| `sam_threshold` | 否 | number | SAM3 grounding 分数阈值，默认 `0.6` |
| `similarity_threshold` | 否 | number | 兼容字段；当前不执行余弦相似度过滤 |
| `nms_iou` | 否 | number | 最终按类别 NMS 阈值，默认 `0.45` |
| `polygon_simplify_epsilon` | 否 | number | 多边形简化参数，默认 `2.0` |
| `return_result_image` | 否 | boolean | 是否生成可视化结果图；批量场景建议保持 `false` |

响应示例：

```json
{
  "model": "ultralytics-sam3",
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
  "prompt": "人; 安全帽",
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
| `prompt` | 否 | string | 顶层文本 prompt；当 `sample_url` 没有正样例或完全省略时必填，支持 `;` / `,` 分隔，中文会自动翻译 |
| `infer_batch_size` | 否 | integer | 预留分批参数，默认 `16`，范围 `1-64` |
| `frame_time` | 否 | integer | 视频抽帧间隔，按帧数计；`0` 表示逐帧，默认 `1` |
| `top_k` | 否 | integer | 每个类别最多保留结果数，默认 `5` |
| `sam_threshold` | 否 | number | SAM3 grounding 分数阈值，默认 `0.6` |
| `similarity_threshold` | 否 | number | 兼容字段；当前不执行余弦相似度过滤 |
| `nms_iou` | 否 | number | 最终按类别 NMS 阈值，默认 `0.45` |
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
- 样例 prompt embedding 会缓存；相同 `download_url + sample_url + sample_url内容 + top_k` 命中缓存时，不重复下载样例图和编码样例 prompt。

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
  -F 'prompt=人;安全帽' \
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
| `prompt` | 否 | string | 顶层文本 prompt；当没有正样例时必填，支持 `;` / `,` 分隔，中文会自动翻译 |
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

模型推理失败。建议先查看服务日志，重点看 CUDA、显存、checkpoint 路径和 Ultralytics 版本。

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
