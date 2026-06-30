"""SAM3 模型运行时和推理 pipeline。

这个模块集中管理最重的模型生命周期和各类推理流程。配置、翻译和图像
工具放在更轻量的模块里，避免路由/auth 代码导入时意外触发模型加载。
"""

import asyncio
import contextlib
import gc
import os
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import (
    CHECKPOINT_PATH,
    CUDA_CLEANUP_AFTER_REQUEST,
    CUDA_CLEANUP_LOG,
    EMPTY_CUDA_CACHE_EACH_REQUEST,
    ENABLE_AUTOCAST,
    IDLE_MODEL_UNLOAD_SECONDS,
    INFER_DTYPE_STR,
    MAX_CONCURRENT_INFERENCES,
    MODEL_LABEL,
    MULTI_NEGATIVE_FILTER_IOU,
    SERIALIZE_MODEL_ACCESS,
    SIMILAR_MODES,
    ULTRALYTICS_IMGSZ,
    ULTRALYTICS_IOU,
    ULTRALYTICS_VERBOSE,
    VISUAL_PROMPT_MAX_CANDIDATES,
)
from .image_utils import (
    bbox_iou_xywh,
    bbox_to_xywh,
    bbox_xywh_to_polygon_points,
    bbox_xywh_to_xyxy,
    build_class_color,
    clip_bnd_points_to_image,
    clip_xyxy_to_image,
    mask_to_polygons,
    normalize_box_segmentation_inputs,
    pil_to_bgr_numpy,
    to_numpy,
    visualize_results,
)


def _resolve_infer_dtype(device_name: str, requested: str) -> torch.dtype:
    """把环境变量里的 dtype 字符串解析成 torch dtype，并处理 CPU 回退。"""
    normalized = requested.strip().lower()
    alias = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    kind = alias.get(normalized)
    if kind is None:
        raise ValueError(
            f"Unsupported SAM3_INFER_DTYPE='{requested}'. "
            "Use one of: bfloat16, float16, float32."
        )

    is_cuda = "cuda" in device_name and torch.cuda.is_available()
    if kind == "bfloat16":
        if is_cuda:
            return torch.bfloat16
        print("bfloat16 requested on non-CUDA device; fallback to float32")
        return torch.float32
    if kind == "float16":
        if is_cuda:
            return torch.float16
        print("float16 requested on non-CUDA device; fallback to float32")
        return torch.float32
    return torch.float32


def _inference_autocast_context():
    """根据设备和配置创建 autocast 上下文；不满足条件时返回空上下文。"""
    use_autocast = (
        ENABLE_AUTOCAST
        and torch.cuda.is_available()
        and "cuda" in device
        and MODEL_DTYPE in {torch.bfloat16, torch.float16}
    )
    if not use_autocast:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=MODEL_DTYPE)


def _patch_openai_clip_tokenizer() -> None:
    """兼容 PyPI openai-clip 和 Ultralytics SAM3 期望的 tokenizer 调用方式。"""
    try:
        import clip
    except Exception:
        return

    tokenizer_module = getattr(clip, "simple_tokenizer", None)
    tokenizer_cls = getattr(tokenizer_module, "SimpleTokenizer", None)
    if tokenizer_cls is None or callable(tokenizer_cls()):
        return

    def _call(self, texts: Any, context_length: int = 77) -> torch.Tensor:
        return clip.tokenize(texts, context_length=context_length, truncate=True)

    tokenizer_cls.__call__ = _call


device = os.getenv("SAM3_DEVICE", "") or ("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("ARGOS_DEVICE_TYPE", "cuda" if "cuda" in device else "cpu")
MODEL_DTYPE = _resolve_infer_dtype(device, INFER_DTYPE_STR)

print("Loading SAM3 model...")
print(f"Using device: {device}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Model label: {MODEL_LABEL}")
print(f"Inference dtype: {MODEL_DTYPE}")
print(f"Autocast enabled: {ENABLE_AUTOCAST}")
print(f"Ultralytics image size: {ULTRALYTICS_IMGSZ}")
print(f"Ultralytics NMS IoU: {ULTRALYTICS_IOU}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except Exception as exc:
    raise RuntimeError(
        "Ultralytics with SAM3 support is required. Install it in the sam3 conda environment, "
        "for example: python -m pip install -U 'ultralytics>=8.4.51'"
    ) from exc

_patch_openai_clip_tokenizer()

# predictor 持有已加载的 SAM3 权重，以及当前图片/特征缓存。
# Ultralytics 推理过程中会修改 predictor 内部状态，所以所有请求
# 后续都必须通过 semaphore 和 MODEL_LOCK 保护访问。
predictor = SAM3SemanticPredictor(
    overrides={
        "model": str(CHECKPOINT_PATH),
        "device": device,
        "conf": 0.01,
        "iou": ULTRALYTICS_IOU,
        "imgsz": ULTRALYTICS_IMGSZ,
        "half": MODEL_DTYPE == torch.float16,
        "verbose": ULTRALYTICS_VERBOSE,
        "save": False,
    }
)
predictor.setup_model()

print("Model loaded successfully")
print(f"CUDA cleanup after request: {CUDA_CLEANUP_AFTER_REQUEST}")
print(f"Idle model unload seconds: {IDLE_MODEL_UNLOAD_SECONDS}")


from .translation import (
    _normalize_prompt_label,
    prepare_single_text_prompt,
    split_prompt_classes,
    translate_to_english,
)


# 所有预测适配层统一返回这个数组结构。这样上层 pipeline 不需要关心
# 结果来自 Ultralytics Result、缓存特征推理，还是原生 SAM3 decoder。
PredictionArrays = Dict[str, np.ndarray]


def _empty_prediction_arrays() -> PredictionArrays:
    """返回和真实预测结果同结构的空结果，避免上层反复写空值分支。"""
    return {
        "masks": np.zeros((0, 0, 0), dtype=bool),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


def _normalize_pic_id(pic_id: Optional[str]) -> str:
    """优先使用调用方传入的图片 id；没有传入时生成一个短请求 id。"""
    return (pic_id or "").strip() or uuid.uuid4().hex[:16]


def _query_pic_id(batch_pic_id: str, index: int, total_count: int) -> str:
    """为 batch 内的每张查询图生成稳定 id，同时保持单图响应兼容。"""
    return batch_pic_id if total_count == 1 else f"{batch_pic_id}_{index + 1}"


def _attach_query_name(
    result: Dict[str, Any],
    query_names: Optional[List[str]],
    index: int,
) -> None:
    """把上传文件名挂到单个 query 结果上，便于调用方回溯。"""
    if query_names and index < len(query_names):
        result["query_name"] = query_names[index]


def _single_query_batch_response(
    query_results: List[Dict[str, Any]],
    processing_time_ms: int,
) -> Dict[str, Any]:
    """batch 只有一张图时，保留旧版单图接口的顶层响应形状。"""
    single = dict(query_results[0])
    single["batch_size"] = 1
    single["query_results"] = query_results
    single["processing_time_ms"] = processing_time_ms
    return single


def _elapsed_ms(start_time: float) -> int:
    """把 perf_counter 起点转换为接口 profile 使用的毫秒耗时。"""
    return int((time.perf_counter() - start_time) * 1000)


def _native_visual_prompt_score_threshold(sam_threshold: float) -> float:
    """原生 visual prompt 路径只使用 SAM3 grounding 分数过滤。"""
    return float(sam_threshold)


def _visual_prompt_candidate_limit(top_k: int) -> int:
    """限制进入高成本 mask 上采样/CPU 后处理的候选数。"""
    normalized_top_k = max(1, int(top_k))
    return max(normalized_top_k, min(VISUAL_PROMPT_MAX_CANDIDATES, normalized_top_k * 2))


def _multi_visual_prompt_candidate_limit(top_k: int, group_count: int) -> int:
    """多类别 visual prompt 按类别收缩候选预算，避免累计候选数过大。"""
    normalized_top_k = max(1, int(top_k))
    normalized_group_count = max(1, int(group_count))
    per_group_limit = _visual_prompt_candidate_limit(normalized_top_k)
    total_budget = max(normalized_top_k, min(VISUAL_PROMPT_MAX_CANDIDATES, normalized_top_k * 6))
    budget_per_group = max(1, total_budget // normalized_group_count)
    return max(1, min(per_group_limit, budget_per_group))


def _sam_grounding_score_fields(score: float) -> Dict[str, float]:
    """生成响应里的分数字段；similarity/combined 字段仅为兼容旧客户端。"""
    rounded_score = round(float(score), 6)
    return {
        "score": rounded_score,
        "sam_score": rounded_score,
        "similarity_score": rounded_score,
        "combined_score": rounded_score,
        "coarse_similarity": rounded_score,
    }


def _maybe_empty_cuda_cache() -> None:
    """按配置释放未使用的 CUDA cache；默认关闭，避免高并发下抖动。"""
    if EMPTY_CUDA_CACHE_EACH_REQUEST and torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def _model_inference_context() -> Iterator[None]:
    """保护所有共享 predictor 访问。

    Ultralytics predictor 会把“本次请求”的状态写到 predictor 对象上：
    当前图片、缓存特征、args、类别名和文本 embedding 都是共享可变状态。
    因此只要多个 HTTP 请求共用同一个 predictor，就必须串行访问模型。
    """
    lock_ctx = MODEL_LOCK if SERIALIZE_MODEL_ACCESS else contextlib.nullcontext()
    autocast_ctx = _inference_autocast_context()
    with torch.inference_mode(), lock_ctx, autocast_ctx:
        yield


def run_ultralytics_prediction(
    image: Image.Image,
    *,
    text: Optional[List[str]] = None,
    bboxes: Optional[List[List[float]]] = None,
    confidence_threshold: float = 0.3,
    reset_cached_image: bool = True,
) -> Any:
    """调用 Ultralytics predictor，并把图片/文本/框 prompt 统一传进去。"""
    np_image = pil_to_bgr_numpy(image)
    if reset_cached_image and hasattr(predictor, "reset_image"):
        predictor.reset_image()
    predictor.args.conf = confidence_threshold
    predictor.args.iou = ULTRALYTICS_IOU
    kwargs: Dict[str, Any] = {
        "source": np_image,
        "stream": False,
    }
    if text is not None:
        kwargs["text"] = text
    if bboxes is not None:
        kwargs["bboxes"] = bboxes
        kwargs["labels"] = [1] * len(bboxes)
    results = predictor(**kwargs)
    return results[0] if results else None


def extract_ultralytics_arrays(result: Any) -> PredictionArrays:
    """从 Ultralytics Result 对象中提取 mask、box、score、class 数组。"""
    if result is None or getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return _empty_prediction_arrays()

    masks_obj = result.masks
    boxes_obj = result.boxes
    masks_np = to_numpy(masks_obj.data)
    boxes_np = to_numpy(boxes_obj.xyxy)
    scores_np = to_numpy(boxes_obj.conf)
    classes_np = to_numpy(boxes_obj.cls).astype(np.int64)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if boxes_np.ndim == 1 and boxes_np.size == 4:
        boxes_np = boxes_np[None, ...]
    if scores_np.ndim == 0:
        scores_np = np.array([float(scores_np)])
    if classes_np.ndim == 0 and classes_np.size > 0:
        classes_np = np.array([int(classes_np)])

    return {
        "masks": masks_np,
        "boxes": boxes_np,
        "scores": scores_np,
        "classes": classes_np,
    }


def set_ultralytics_image_features(image: Image.Image) -> Dict[str, Any]:
    """设置当前图片并读取 SAM3 backbone 特征，供后续缓存推理复用。"""
    predictor.set_image(pil_to_bgr_numpy(image))
    features = predictor.features
    if not isinstance(features, dict):
        raise ValueError("Ultralytics SAM3 did not return dictionary features")
    backbone_fpn = features.get("backbone_fpn")
    if not isinstance(backbone_fpn, list) or not backbone_fpn:
        raise ValueError("Ultralytics SAM3 features missing backbone_fpn")
    feature_map = backbone_fpn[-1]
    if not torch.is_tensor(feature_map) or feature_map.ndim != 4 or feature_map.shape[0] < 1:
        raise ValueError("Unexpected Ultralytics SAM3 feature map shape")
    return features


def extract_ultralytics_feature_arrays(
    masks: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor],
) -> PredictionArrays:
    """把 predictor.inference_features 的原始张量输出转成统一数组结构。"""
    if masks is None or boxes is None:
        return _empty_prediction_arrays()

    masks_np = to_numpy(masks)
    boxes_np = to_numpy(boxes)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if boxes_np.ndim == 1 and boxes_np.size >= 6:
        boxes_np = boxes_np[None, ...]
    if boxes_np.ndim != 2 or boxes_np.shape[1] < 6:
        return _empty_prediction_arrays()

    return {
        "masks": masks_np,
        "boxes": boxes_np[:, :4],
        "scores": boxes_np[:, 4],
        "classes": boxes_np[:, 5].astype(np.int64),
    }


def run_ultralytics_cached_feature_prediction(
    image: Image.Image,
    *,
    text: Optional[List[str]] = None,
    confidence_threshold: float = 0.3,
) -> PredictionArrays:
    """基于已缓存的图片特征执行文本 prompt 推理，减少重复 backbone 计算。"""
    features = predictor.features
    if not isinstance(features, dict):
        raise ValueError("Ultralytics SAM3 cached image features are not available")

    predictor.args.conf = confidence_threshold
    predictor.args.iou = ULTRALYTICS_IOU
    if hasattr(predictor, "inference_features"):
        masks, boxes = predictor.inference_features(
            dict(features),
            (image.height, image.width),
            text=text,
        )
        return extract_ultralytics_feature_arrays(masks, boxes)

    result = run_ultralytics_prediction(
        image,
        text=text,
        confidence_threshold=confidence_threshold,
        reset_cached_image=False,
    )
    return extract_ultralytics_arrays(result)


def _prepare_detection_classes(prompt: str) -> Tuple[List[Dict[str, Any]], str, bool]:
    """拆分类别 prompt，必要时翻译成英文，并为可视化分配颜色。"""
    original_classes = split_prompt_classes(prompt)
    if not original_classes:
        raise ValueError("Prompt is empty after parsing. Use ';' or ',' to separate classes.")

    classes_info: List[Dict[str, Any]] = []
    translated_classes: List[str] = []
    for class_index, one_class in enumerate(original_classes):
        translated = translate_to_english(one_class)
        translated = translated.strip() if translated else one_class
        if not translated:
            translated = one_class

        translated_classes.append(translated)
        classes_info.append(
            {
                "class_name": translated,
                "original_class_name": one_class,
                "color": build_class_color(class_index),
            }
        )

    translated_prompt = "; ".join(translated_classes)
    original_prompt = "; ".join(original_classes)
    return classes_info, translated_prompt, translated_prompt != original_prompt


def _count_labels_by_category(labels: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计每个原始类别的检测数量，用于 response.detection_details。"""
    counts: Dict[str, int] = {}
    for label in labels:
        category = label["category"]
        counts[category] = counts.get(category, 0) + 1
    return counts


def _visualization_groups_to_detections(visualization_groups: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把按类别累积的可视化数据转成 visualize_results 需要的结构。"""
    detections: List[Dict[str, Any]] = []
    for group in visualization_groups.values():
        if not group["masks"]:
            continue
        detections.append(
            {
                "class_name": group["class_name"],
                "original_class_name": group["original_class_name"],
                "masks": np.asarray(group["masks"]),
                "boxes": np.asarray(group["boxes"]),
                "scores": np.asarray(group["scores"]),
                "color": group["color"],
            }
        )
    return detections


def _visualize_single_detection_group(
    image: Image.Image,
    *,
    class_name: str,
    masks: List[np.ndarray],
    boxes: List[List[float]],
    scores: List[float],
    color_index: int = 0,
    original_class_name: Optional[str] = None,
) -> Optional[str]:
    """只有一个可视化类别时的便捷封装，返回生成的结果图文件名。"""
    if not masks or not boxes or not scores:
        return None

    detections = [
        {
            "class_name": class_name,
            "original_class_name": original_class_name or class_name,
            "masks": np.asarray(masks),
            "boxes": np.asarray(boxes),
            "scores": np.asarray(scores),
            "color": build_class_color(color_index),
        }
    ]
    return visualize_results(image, detections)


def _extract_primary_component_mask(
    mask_2d: np.ndarray,
    candidate_bnd_points: List[float],
    min_area_ratio: float = 0.0001,
) -> np.ndarray:
    """从预测 mask 中挑一个主要连通域，减少一张 mask 粘连多个目标的问题。"""
    mask_np = np.asarray(mask_2d)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return np.zeros((1, 1), dtype=np.uint8)

    binary = (mask_np > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return binary

    h_img, w_img = binary.shape
    x, y, w, h = [float(v) for v in candidate_bnd_points]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    cx = int(round(x + w / 2.0))
    cy = int(round(y + h / 2.0))
    cx = max(0, min(cx, w_img - 1))
    cy = max(0, min(cy, h_img - 1))
    label_at_center = int(labels[cy, cx])

    candidate_labels: List[Tuple[int, int]] = []
    total_pixels = max(1, h_img * w_img)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area / total_pixels < min_area_ratio:
            continue
        candidate_labels.append((label_id, area))

    if not candidate_labels:
        return np.zeros_like(binary)

    # 框中心所在连通域通常是用户/候选框真正指向的目标；如果中心不在
    # 有效连通域里，再退回到面积最大的连通域。
    if label_at_center > 0 and any(label_id == label_at_center for label_id, _ in candidate_labels):
        chosen_label = label_at_center
    else:
        chosen_label = max(candidate_labels, key=lambda item: item[1])[0]
    return (labels == chosen_label).astype(np.uint8)


# 并发分两层控制：
# 1. INFERENCE_SEMAPHORE 限制进入后台线程池的推理请求数量，避免请求无限堆积。
# 2. MODEL_LOCK 串行化真正的 predictor 访问，防止共享模型状态被并发请求互相覆盖。
MODEL_LOCK = threading.Lock()
INFERENCE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
INFERENCE_STATE_LOCK = threading.Lock()
ACTIVE_INFERENCE_COUNT = 0
LAST_INFERENCE_FINISHED_AT = 0.0
IDLE_MODEL_UNLOADED = False


def get_cuda_memory_stats() -> Optional[Dict[str, int]]:
    """读取当前 CUDA 设备的显存统计；CPU 模式下返回 None。"""
    if not torch.cuda.is_available() or "cuda" not in device:
        return None

    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return {
        "device_index": int(device_index),
        "allocated_mb": int(torch.cuda.memory_allocated(device_index) / 1024 / 1024),
        "reserved_mb": int(torch.cuda.memory_reserved(device_index) / 1024 / 1024),
        "max_allocated_mb": int(torch.cuda.max_memory_allocated(device_index) / 1024 / 1024),
        "max_reserved_mb": int(torch.cuda.max_memory_reserved(device_index) / 1024 / 1024),
    }


def _reset_predictor_runtime_state() -> None:
    """清理 predictor 上一次请求残留的图片、结果和视频写入器状态。"""
    if hasattr(predictor, "reset_image"):
        predictor.reset_image()

    for attr_name in (
        "results",
        "batch",
        "dataset",
        "source_type",
        "plotted_img",
        "transforms",
    ):
        if hasattr(predictor, attr_name):
            setattr(predictor, attr_name, None)

    vid_writer = getattr(predictor, "vid_writer", None)
    if isinstance(vid_writer, dict):
        for writer in list(vid_writer.values()):
            if hasattr(writer, "release"):
                with contextlib.suppress(Exception):
                    writer.release()
        predictor.vid_writer = {}


def cleanup_cuda_runtime_state(*, unload_model: bool = False, reason: str = "request") -> None:
    """在没有活跃推理时清理 CUDA 缓存，可选地把模型卸到 CPU/置空。"""
    if not torch.cuda.is_available() or "cuda" not in device:
        return

    global IDLE_MODEL_UNLOADED
    with INFERENCE_STATE_LOCK:
        # 有请求正在推理或等待模型锁时不能清 predictor 状态。
        if ACTIVE_INFERENCE_COUNT > 0:
            return

    before = get_cuda_memory_stats() if CUDA_CLEANUP_LOG else None
    with MODEL_LOCK:
        _reset_predictor_runtime_state()

        if unload_model and getattr(predictor, "model", None) is not None:
            with contextlib.suppress(Exception):
                predictor.model.to("cpu")
            predictor.model = None
            predictor.mean = None
            predictor.std = None
            predictor.done_warmup = False
            IDLE_MODEL_UNLOADED = True

    gc.collect()
    torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()

    if CUDA_CLEANUP_LOG:
        after = get_cuda_memory_stats()
        print(f"CUDA cleanup ({reason}): before={before} after={after} unload_model={unload_model}")


async def run_inference_in_thread(func: Any, *args: Any) -> Any:
    """把同步推理函数放到线程池执行，并维护活跃推理计数。"""
    global ACTIVE_INFERENCE_COUNT, LAST_INFERENCE_FINISHED_AT, IDLE_MODEL_UNLOADED
    with INFERENCE_STATE_LOCK:
        ACTIVE_INFERENCE_COUNT += 1
        IDLE_MODEL_UNLOADED = False

    try:
        return await asyncio.to_thread(func, *args)
    finally:
        should_cleanup = False
        with INFERENCE_STATE_LOCK:
            ACTIVE_INFERENCE_COUNT = max(0, ACTIVE_INFERENCE_COUNT - 1)
            if ACTIVE_INFERENCE_COUNT == 0:
                LAST_INFERENCE_FINISHED_AT = time.monotonic()
                should_cleanup = CUDA_CLEANUP_AFTER_REQUEST or EMPTY_CUDA_CACHE_EACH_REQUEST

        if should_cleanup:
            await asyncio.to_thread(cleanup_cuda_runtime_state, reason="request")


def _idle_model_unload_worker() -> None:
    """后台空闲检查线程；超过配置时间后尝试释放模型占用的显存。"""
    if IDLE_MODEL_UNLOAD_SECONDS <= 0:
        return

    while True:
        time.sleep(min(max(IDLE_MODEL_UNLOAD_SECONDS // 2, 5), 60))
        with INFERENCE_STATE_LOCK:
            active_count = ACTIVE_INFERENCE_COUNT
            last_finished_at = LAST_INFERENCE_FINISHED_AT
            already_unloaded = IDLE_MODEL_UNLOADED

        if active_count > 0 or already_unloaded or last_finished_at <= 0:
            continue

        idle_seconds = time.monotonic() - last_finished_at
        if idle_seconds >= IDLE_MODEL_UNLOAD_SECONDS:
            cleanup_cuda_runtime_state(unload_model=True, reason=f"idle_{int(idle_seconds)}s")


if IDLE_MODEL_UNLOAD_SECONDS > 0:
    # 只有显式配置时才启动后台线程；默认不做 idle unload。
    threading.Thread(target=_idle_model_unload_worker, name="sam3-idle-model-unload", daemon=True).start()


def run_detection_pipeline(
    image: Image.Image,
    prompt: str,
    confidence_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
) -> Dict[str, Any]:
    """文本开放词表检测入口：按 prompt 类别逐个推理并合并响应。"""
    image = image.convert("RGB")
    start_time = time.perf_counter()
    normalized_pic_id = _normalize_pic_id(pic_id)
    classes_info, translated_prompt, was_translated = _prepare_detection_classes(prompt)

    pic_labels: List[Dict[str, Any]] = []
    visualization_groups: Dict[int, Dict[str, Any]] = {}

    with _model_inference_context():
        set_ultralytics_image_features(image)

        # 先对整图计算一次特征，再按类别复用 cached features 做文本 grounding。
        for class_index, class_info in enumerate(classes_info):
            arrays = run_ultralytics_cached_feature_prediction(
                image,
                text=[class_info["class_name"]],
                confidence_threshold=confidence_threshold,
            )
            masks_np = arrays["masks"]
            boxes_np = arrays["boxes"]
            scores_np = arrays["scores"]

            for mask, box, score in zip(masks_np, boxes_np, scores_np):
                if float(score) < confidence_threshold:
                    continue

                bnd_points = bbox_to_xywh(box)
                polygons = mask_to_polygons(mask, epsilon=polygon_simplify_epsilon)
                polygon_points = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(bnd_points)

                pic_labels.append(
                    {
                        "category": class_info["original_class_name"],
                        "translated_category": class_info["class_name"],
                        "score": round(float(score), 6),
                        "bnd_points": bnd_points,
                        "polygon_points": polygon_points,
                        "mask_area": int(np.count_nonzero(np.asarray(mask) > 0.5)),
                    }
                )
                visualization_group = visualization_groups.setdefault(
                    class_index,
                    {
                        "class_name": class_info["class_name"],
                        "original_class_name": class_info["original_class_name"],
                        "masks": [],
                        "boxes": [],
                        "scores": [],
                        "color": class_info["color"],
                    },
                )
                visualization_group["masks"].append(np.asarray(mask, dtype=np.float32))
                visualization_group["boxes"].append(np.asarray(box, dtype=np.float32))
                visualization_group["scores"].append(float(score))

    _maybe_empty_cuda_cache()
    detection_details = _count_labels_by_category(pic_labels)
    processing_time_ms = _elapsed_ms(start_time)
    detection_for_viz = _visualization_groups_to_detections(visualization_groups)
    result_image = visualize_results(image, detection_for_viz) if detection_for_viz else None

    response: Dict[str, Any] = {
        "model": MODEL_LABEL,
        "pic_id": normalized_pic_id,
        "success": True,
        "pic_labels": pic_labels,
        "num_detections": len(pic_labels),
        "classes_detected": len(detection_details),
        "detection_details": detection_details,
        "prompt": prompt,
        "translated_prompt": translated_prompt if was_translated else None,
        "was_translated": was_translated,
        "confidence_threshold": confidence_threshold,
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
    }

    return response


def _build_box_segmentation_result(
    arrays: Dict[str, np.ndarray],
    index: int,
    requested_bnd_points: List[float],
    polygon_simplify_epsilon: float,
) -> Dict[str, Any]:
    """把一次 box prompt 的模型输出转成接口需要的单个框选分割结果。"""
    masks_np = arrays["masks"]
    boxes_np = arrays["boxes"]
    scores_np = arrays["scores"]

    used_fallback = False
    score = 0.0
    mask_area = 0
    selected_bnd_points = requested_bnd_points
    polygon_points: List[List[float]] = bbox_xywh_to_polygon_points(requested_bnd_points)

    has_predictions = (
        isinstance(scores_np, np.ndarray)
        and scores_np.size > 0
        and isinstance(masks_np, np.ndarray)
        and masks_np.shape[0] > 0
    )
    if has_predictions:
        # Ultralytics 返回顺序通常和输入框顺序一致；防御性地限制 index 边界。
        best_index = min(index, scores_np.shape[0] - 1)
        score = round(float(scores_np[best_index]), 6)

        best_mask = np.asarray(masks_np[best_index])
        if best_mask.ndim > 2:
            best_mask = np.squeeze(best_mask)
        mask_area = int(np.count_nonzero(best_mask > 0.5))

        if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > best_index:
            selected_bnd_points = bbox_to_xywh(boxes_np[best_index])

        polygons = mask_to_polygons(best_mask, epsilon=polygon_simplify_epsilon)
        if polygons:
            polygon_points = polygons[0]["points"]
        else:
            used_fallback = True
    else:
        used_fallback = True

    return {
        "input_bnd_points": requested_bnd_points,
        "bnd_points": selected_bnd_points,
        "polygon_points": polygon_points,
        "score": score,
        "mask_area": mask_area,
        "used_fallback": used_fallback,
    }


def run_box_segmentation_pipeline(
    image: Image.Image,
    bnd_points: Any,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
) -> Dict[str, Any]:
    """框选分割入口：把用户 xywh 框转成 SAM3 几何 prompt 后返回 mask/polygon。"""
    image = image.convert("RGB")
    start_time = time.perf_counter()
    normalized_pic_id = _normalize_pic_id(pic_id)

    raw_boxes = normalize_box_segmentation_inputs(bnd_points)

    clipped_boxes: List[List[float]] = []
    xyxy_boxes: List[List[float]] = []
    for index, one_bnd_points in enumerate(raw_boxes):
        try:
            clipped_bnd_points = clip_bnd_points_to_image(
                bnd_points=one_bnd_points,
                image_width=image.width,
                image_height=image.height,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid bnd_points_list[{index}]: {exc}") from exc

        clipped_boxes.append(clipped_bnd_points)
        xyxy_boxes.append(bbox_xywh_to_xyxy(clipped_bnd_points))

    segmentations: List[Dict[str, Any]] = []
    with _model_inference_context():
        result = run_ultralytics_prediction(
            image,
            bboxes=xyxy_boxes,
            confidence_threshold=0.0,
        )
        arrays = extract_ultralytics_arrays(result)
        for index, clipped_box in enumerate(clipped_boxes):
            one_result = _build_box_segmentation_result(
                arrays=arrays,
                index=index,
                requested_bnd_points=clipped_box,
                polygon_simplify_epsilon=polygon_simplify_epsilon,
            )
            one_result["index"] = index
            segmentations.append(one_result)

    _maybe_empty_cuda_cache()

    processing_time_ms = _elapsed_ms(start_time)
    response: Dict[str, Any] = {
        "model": MODEL_LABEL,
        "pic_id": normalized_pic_id,
        "success": True,
        "segmentations": segmentations,
        "num_segmentations": len(segmentations),
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
    }

    # 兼容旧客户端：单框请求时继续在顶层返回 bnd_points/polygon_points 等字段。
    if len(segmentations) == 1:
        first = segmentations[0]
        response.update(
            {
                "bnd_points": first["bnd_points"],
                "polygon_points": first["polygon_points"],
                "score": first["score"],
                "mask_area": first["mask_area"],
                "used_fallback": first["used_fallback"],
            }
        )

    return response


def _prepare_similar_reference_context(
    reference_image: Image.Image,
    reference_bnd_points: Optional[List[float]],
    top_k: int,
) -> Dict[str, Any]:
    """预处理单个相似目标参考框；兼容旧单样例入口。"""
    reference_image = reference_image.convert("RGB")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if reference_bnd_points is None:
        raise ValueError("reference_bnd_points is required, format: [x, y, w, h]")

    reference_bnd_points = clip_bnd_points_to_image(
        bnd_points=reference_bnd_points,
        image_width=reference_image.width,
        image_height=reference_image.height,
    )
    reference_xyxy = bbox_xywh_to_xyxy(reference_bnd_points)

    reference_features = set_ultralytics_image_features(reference_image)
    result = run_ultralytics_prediction(
        reference_image,
        bboxes=[reference_xyxy],
        confidence_threshold=0.0,
        reset_cached_image=False,
    )
    arrays = extract_ultralytics_arrays(result)
    return _build_similar_reference_context_from_arrays(
        reference_image=reference_image,
        reference_bnd_points=reference_bnd_points,
        arrays=arrays,
        result_index=0,
        reference_features=reference_features,
    )


def _build_similar_reference_context_from_arrays(
    reference_image: Image.Image,
    reference_bnd_points: List[float],
    arrays: PredictionArrays,
    result_index: int,
    reference_features: Optional[Dict[str, Any]] = None,
    build_reference_result_image: bool = True,
) -> Dict[str, Any]:
    """把批量 box prompt 输出拆成单个样本上下文，避免同图多实例重复提特征。"""
    ref_masks_np = arrays["masks"]
    ref_boxes_np = arrays["boxes"]
    ref_scores_np = arrays["scores"]
    reference_xyxy = bbox_xywh_to_xyxy(reference_bnd_points)

    if ref_scores_np.size == 0 or ref_masks_np.shape[0] == 0:
        raise ValueError("reference image SAM+box did not produce valid masks")

    ref_best_idx = min(max(0, int(result_index)), int(ref_scores_np.shape[0]) - 1)
    ref_best_mask = np.asarray(ref_masks_np[ref_best_idx])
    ref_sam_score = float(ref_scores_np[ref_best_idx])
    if isinstance(ref_boxes_np, np.ndarray) and ref_boxes_np.shape[0] > ref_best_idx:
        ref_best_xyxy = clip_xyxy_to_image([float(v) for v in ref_boxes_np[ref_best_idx]], reference_image.width, reference_image.height)
        ref_best_xywh = bbox_to_xywh(np.asarray(ref_best_xyxy))
    else:
        ref_best_xywh = reference_bnd_points
        ref_best_xyxy = reference_xyxy

    ref_primary_mask_u8 = _extract_primary_component_mask(ref_best_mask, ref_best_xywh)
    if ref_primary_mask_u8.max() == 0:
        raise ValueError("reference mask is empty after postprocess")

    reference_result_image = None
    if build_reference_result_image:
        ref_detection_for_viz = [
            {
                "class_name": "reference_object",
                "original_class_name": "reference_object",
                "masks": np.asarray([ref_primary_mask_u8.astype(np.float32)]),
                "boxes": np.asarray([ref_best_xyxy]),
                "scores": np.asarray([max(0.0, min(1.0, ref_sam_score))]),
                "color": build_class_color(1),
            }
        ]
        reference_result_image = visualize_results(reference_image, ref_detection_for_viz)

    return {
        "reference_image": reference_image,
        "reference_bnd_points": reference_bnd_points,
        "reference_mask": ref_primary_mask_u8,
        "ref_best_xywh": ref_best_xywh,
        "ref_best_xyxy": ref_best_xyxy,
        "reference_score": max(0.0, min(1.0, ref_sam_score)),
        "reference_features": reference_features,
        "reference_result_image": reference_result_image,
    }


def _prediction_box_xywh_from_arrays(
    arrays: PredictionArrays,
    prediction_index: int,
    image_width: int,
    image_height: int,
) -> Optional[List[float]]:
    boxes_np = arrays["boxes"]
    if not isinstance(boxes_np, np.ndarray):
        return None
    if boxes_np.ndim == 1 and boxes_np.size == 4:
        boxes_np = boxes_np[None, ...]
    if boxes_np.ndim != 2 or boxes_np.shape[0] <= prediction_index:
        return None
    box_xyxy = clip_xyxy_to_image(
        [float(v) for v in boxes_np[prediction_index]],
        image_width,
        image_height,
    )
    return bbox_to_xywh(np.asarray(box_xyxy))


def _match_reference_prediction_indices(
    reference_boxes_xywh: List[List[float]],
    arrays: PredictionArrays,
    image_width: int,
    image_height: int,
) -> List[int]:
    """把输入框匹配到 SAM 返回框，避免多框返回顺序变化导致正负样本串位。"""
    scores_np = arrays["scores"]
    prediction_count = 0
    if isinstance(scores_np, np.ndarray):
        prediction_count = 1 if scores_np.ndim == 0 and scores_np.size == 1 else int(scores_np.shape[0])
    if prediction_count <= 0:
        return [0 for _ in reference_boxes_xywh]

    prediction_boxes = [
        _prediction_box_xywh_from_arrays(arrays, prediction_index, image_width, image_height)
        for prediction_index in range(prediction_count)
    ]
    pairs: List[Tuple[float, int, int, int]] = []
    for reference_index, reference_box in enumerate(reference_boxes_xywh):
        for prediction_index, prediction_box in enumerate(prediction_boxes):
            if prediction_box is None:
                continue
            same_order_bonus = 1 if reference_index == prediction_index else 0
            pairs.append(
                (
                    bbox_iou_xywh(reference_box, prediction_box),
                    same_order_bonus,
                    reference_index,
                    prediction_index,
                )
            )

    assigned_reference_indices = set()
    assigned_prediction_indices = set()
    matched_indices: List[Optional[int]] = [None for _ in reference_boxes_xywh]
    for iou, _same_order_bonus, reference_index, prediction_index in sorted(pairs, reverse=True):
        if iou <= 0:
            continue
        if reference_index in assigned_reference_indices or prediction_index in assigned_prediction_indices:
            continue
        matched_indices[reference_index] = prediction_index
        assigned_reference_indices.add(reference_index)
        assigned_prediction_indices.add(prediction_index)

    for reference_index, matched_index in enumerate(matched_indices):
        if matched_index is not None:
            continue
        if reference_index < prediction_count and reference_index not in assigned_prediction_indices:
            matched_indices[reference_index] = reference_index
            assigned_prediction_indices.add(reference_index)
            continue
        fallback_pairs = [
            (iou, prediction_index)
            for iou, _same_order_bonus, pair_reference_index, prediction_index in pairs
            if pair_reference_index == reference_index and prediction_index not in assigned_prediction_indices
        ]
        if fallback_pairs:
            _iou, prediction_index = max(fallback_pairs)
            matched_indices[reference_index] = prediction_index
            assigned_prediction_indices.add(prediction_index)
        else:
            matched_indices[reference_index] = min(reference_index, prediction_count - 1)

    return [int(index or 0) for index in matched_indices]


def _visualize_multi_reference_contexts(
    reference_image: Image.Image,
    reference_contexts: List[Dict[str, Any]],
) -> Optional[str]:
    """同一张样例图只保存一张参考结果图，并标出所有正/负样本框。"""
    detections_for_viz: List[Dict[str, Any]] = []
    for sample_ctx in reference_contexts:
        reference_mask = sample_ctx.get("reference_mask")
        reference_box = sample_ctx.get("ref_best_xyxy")
        if reference_mask is None or reference_box is None:
            continue
        is_negative = sample_ctx.get("sample_type") == "negative" or sample_ctx.get("is_negative") is True
        label_prefix = "负样本" if is_negative else "正样本"
        category = sample_ctx.get("category") or "reference_object"
        detections_for_viz.append(
            {
                "class_name": f"{label_prefix}:{category}",
                "original_class_name": f"{label_prefix}:{category}",
                "masks": np.asarray([np.asarray(reference_mask, dtype=np.float32)]),
                "boxes": np.asarray([reference_box]),
                "scores": np.asarray([float(sample_ctx.get("reference_score", 0.0))]),
                "color": np.asarray([220, 38, 38] if is_negative else [15, 118, 110], dtype=np.float32),
            }
        )
    if not detections_for_viz:
        return None
    return visualize_results(reference_image, detections_for_viz)


def _build_sam3_geometric_prompt_from_boxes(
    boxes_xywh: List[List[float]],
    image_width: int,
    image_height: int,
) -> Any:
    """把原图像素坐标里的 xywh 框转换成 SAM3 内部几何 prompt。"""
    xyxy_boxes = [
        bbox_xywh_to_xyxy(clip_bnd_points_to_image(one_box, image_width, image_height))
        for one_box in boxes_xywh
    ]
    bboxes, labels = predictor._prepare_geometric_prompts((image_height, image_width), xyxy_boxes, None)
    geometric_prompt = predictor._get_dummy_prompt(num_prompts=1)
    if bboxes is not None:
        for index in range(len(bboxes)):
            geometric_prompt.append_boxes(bboxes[[index]], labels[[index]])
    return geometric_prompt


def _build_sam3_geometric_prompt_from_box(box_xywh: List[float], image_width: int, image_height: int) -> Any:
    """单框版本的 SAM3 几何 prompt 构造。"""
    return _build_sam3_geometric_prompt_from_boxes([box_xywh], image_width, image_height)


def _prepare_sam3_backbone_features(backbone_out: Dict[str, Any], batch: int = 1) -> Tuple[Dict[str, Any], List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
    """按 Ultralytics SAM 内部格式展开 backbone 特征，供 encoder/decoder 复用。"""
    if batch > 1:
        backbone_out = {
            **backbone_out,
            "backbone_fpn": [feat.expand(batch, -1, -1, -1) for feat in backbone_out["backbone_fpn"]],
            "vision_pos_enc": [pos.expand(batch, -1, -1, -1) for pos in backbone_out["vision_pos_enc"]],
        }

    assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
    assert len(backbone_out["backbone_fpn"]) >= predictor.model.num_feature_levels

    feature_maps = backbone_out["backbone_fpn"][-predictor.model.num_feature_levels :]
    vision_pos_embeds = backbone_out["vision_pos_enc"][-predictor.model.num_feature_levels :]
    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
    vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
    vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
    return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


def _extract_sam3_semantic_arrays_from_raw_outputs(
    raw_outputs: Dict[str, Any],
    image_height: int,
    image_width: int,
    confidence_threshold: float,
    max_candidates: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """把 SAM3 原始 grounding 输出后处理成统一的 mask/box/score 数组。"""
    import torchvision

    pred_boxes = raw_outputs["pred_boxes"]
    pred_logits = raw_outputs["pred_logits"]
    pred_masks = raw_outputs["pred_masks"]
    pred_scores = pred_logits.sigmoid()
    presence_logits = raw_outputs.get("presence_logit_dec")
    if presence_logits is not None:
        pred_scores = pred_scores * presence_logits.sigmoid().unsqueeze(1)
    pred_scores = pred_scores.squeeze(-1)
    raw_candidate_count = int(pred_scores.numel())

    pred_cls = torch.arange(
        pred_scores.shape[0],
        dtype=pred_scores.dtype,
        device=pred_scores.device,
    )[:, None].expand_as(pred_scores)
    pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

    # 先按置信度过滤，再做 NMS。这里返回 raw/kept 数量用于接口 profile。
    keep = pred_scores > confidence_threshold
    pred_masks = pred_masks[keep]
    pred_boxes = pred_boxes[keep]
    if pred_boxes.numel() == 0 or pred_masks.numel() == 0:
        return _empty_prediction_arrays(), raw_candidate_count, 0

    pred_boxes = pred_boxes.clone()
    xywh = pred_boxes[:, :4]
    cx, cy, w, h = xywh.unbind(-1)
    pred_boxes[:, 0] = cx - w / 2.0
    pred_boxes[:, 1] = cy - h / 2.0
    pred_boxes[:, 2] = cx + w / 2.0
    pred_boxes[:, 3] = cy + h / 2.0

    class_offsets = pred_boxes[:, 5:6] * (0 if predictor.args.agnostic_nms else 7680)
    keep = torchvision.ops.nms(pred_boxes[:, :4] + class_offsets, pred_boxes[:, 4], predictor.args.iou)
    pred_boxes = pred_boxes[keep]
    pred_masks = pred_masks[keep]
    order = torch.argsort(pred_boxes[:, 4], descending=True)
    if max_candidates is not None:
        order = order[: int(max_candidates)]
    pred_boxes = pred_boxes[order]
    pred_masks = pred_masks[order]
    kept_candidate_count = int(pred_boxes.shape[0])

    pred_masks = F.interpolate(
        pred_masks.float()[None],
        (image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )[0] > 0.5
    pred_boxes[:, [0, 2]] *= float(image_width)
    pred_boxes[:, [1, 3]] *= float(image_height)

    return (
        {
            "masks": to_numpy(pred_masks),
            "boxes": to_numpy(pred_boxes[:, :4]),
            "scores": to_numpy(pred_boxes[:, 4]),
            "classes": to_numpy(pred_boxes[:, 5]).astype(np.int64),
        },
        raw_candidate_count,
        kept_candidate_count,
    )


def _encode_reference_visual_prompt_from_boxes(
    reference_image: Image.Image,
    reference_boxes_xywh: List[List[float]],
    reference_features: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """把参考图上的一个或多个框编码成 SAM3 visual prompt token。"""
    reference_image = reference_image.convert("RGB")
    encode_start = time.perf_counter()
    if reference_features is None:
        reference_features = set_ultralytics_image_features(reference_image)
    _, reference_img_feats, reference_img_pos_embeds, reference_feat_sizes = _prepare_sam3_backbone_features(
        reference_features,
        batch=1,
    )
    reference_prompt = _build_sam3_geometric_prompt_from_boxes(
        reference_boxes_xywh,
        reference_image.width,
        reference_image.height,
    )
    reference_visual_prompt_embed, reference_visual_prompt_mask = predictor.model._encode_prompt(
        reference_img_feats,
        reference_img_pos_embeds,
        reference_feat_sizes,
        reference_prompt,
    )
    return reference_visual_prompt_embed, reference_visual_prompt_mask, {
        "reference_prompt_encode_ms": _elapsed_ms(encode_start)
    }


def _run_sam3_query_grounding_with_visual_prompt_embeddings(
    query_image: Image.Image,
    query_features: Dict[str, Any],
    visual_prompt_embed: torch.Tensor,
    visual_prompt_mask: torch.Tensor,
    text_prompt: Optional[List[str]] = None,
    confidence_threshold: float = 0.0,
    max_candidates: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """在查询图上使用已编码的 visual prompt token 做 SAM3 grounding。"""
    prompt_batch = text_prompt if text_prompt is not None else ["visual"]
    if predictor.model.names != prompt_batch:
        predictor.model.set_classes(text=prompt_batch)

    grounding_start = time.perf_counter()
    text_ids = torch.arange(len(prompt_batch), device=predictor.device, dtype=torch.long)
    query_backbone_out, query_img_feats, query_img_pos_embeds, query_feat_sizes = _prepare_sam3_backbone_features(
        dict(query_features),
        batch=len(prompt_batch),
    )
    query_backbone_out.update({key: value for key, value in predictor.model.text_embeddings.items()})
    text_features = query_backbone_out["language_features"][:, text_ids]
    text_masks = query_backbone_out["language_mask"][text_ids]
    prompt_embed = torch.cat([text_features, visual_prompt_embed], dim=0)
    prompt_mask = torch.cat([text_masks, visual_prompt_mask], dim=1)

    encoder_out = predictor.model._run_encoder(
        query_img_feats,
        query_img_pos_embeds,
        query_feat_sizes,
        prompt_embed,
        prompt_mask,
    )
    raw_outputs: Dict[str, Any] = {"backbone_out": query_backbone_out}
    raw_outputs, hs = predictor.model._run_decoder(
        memory=encoder_out["encoder_hidden_states"],
        pos_embed=encoder_out["pos_embed"],
        src_mask=encoder_out["padding_mask"],
        out=raw_outputs,
        prompt=prompt_embed,
        prompt_mask=prompt_mask,
        encoder_out=encoder_out,
    )
    predictor.model._run_segmentation_heads(
        out=raw_outputs,
        backbone_out=query_backbone_out,
        encoder_hidden_states=encoder_out["encoder_hidden_states"],
        prompt=prompt_embed,
        prompt_mask=prompt_mask,
        hs=hs,
    )
    grounding_forward_ms = _elapsed_ms(grounding_start)

    arrays, raw_candidate_count, kept_candidate_count = _extract_sam3_semantic_arrays_from_raw_outputs(
        raw_outputs,
        query_image.height,
        query_image.width,
        confidence_threshold,
        max_candidates,
    )
    return arrays, {
        "grounding_forward_ms": grounding_forward_ms,
        "raw_candidate_count": raw_candidate_count,
        "kept_candidate_count": kept_candidate_count,
    }


def _iter_primary_mask_candidates(
    arrays: PredictionArrays,
    image: Image.Image,
    *,
    min_area_ratio: float = 0.0002,
    score_threshold: float = 0.0,
    fallback_box_xyxy: Optional[List[float]] = None,
) -> Iterator[Dict[str, Any]]:
    """遍历预测结果，并只产出通过面积/分数过滤的主连通域候选。"""
    masks_np = arrays["masks"]
    boxes_np = arrays["boxes"]
    scores_np = arrays["scores"]

    for index in range(scores_np.shape[0]):
        if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > index:
            one_box_xyxy = [float(v) for v in boxes_np[index]]
        elif fallback_box_xyxy is not None:
            one_box_xyxy = [float(v) for v in fallback_box_xyxy]
        else:
            continue

        one_score = float(scores_np[index])
        if one_score < score_threshold:
            continue
        one_mask = np.asarray(masks_np[index])
        one_box_xyxy = clip_xyxy_to_image(one_box_xyxy, image.width, image.height)
        one_box_xywh = bbox_to_xywh(np.asarray(one_box_xyxy))
        primary_u8 = _extract_primary_component_mask(one_mask, one_box_xywh)
        if primary_u8.size <= 1 or primary_u8.max() == 0:
            continue

        primary_box_xyxy = mask_to_xyxy(primary_u8)
        if primary_box_xyxy is not None:
            one_box_xyxy = clip_xyxy_to_image(primary_box_xyxy, image.width, image.height)
            one_box_xywh = bbox_to_xywh(np.asarray(one_box_xyxy))

        primary_area = int(np.count_nonzero(primary_u8))
        area_ratio = primary_area / max(1, image.width * image.height)
        if area_ratio <= min_area_ratio:
            continue

        yield {
            "index": index,
            "mask": one_mask,
            "primary_mask": primary_u8,
            "box_xyxy": one_box_xyxy,
            "bnd_points": one_box_xywh,
            "score": one_score,
            "mask_area": primary_area,
        }


def _run_similar_query_with_reference_context(
    reference_ctx: Dict[str, Any],
    query_image: Image.Image,
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
    prompt_text: Optional[str] = None,
) -> Dict[str, Any]:
    """使用已准备好的参考上下文，在单张查询图上查找相似目标。"""
    query_image = query_image.convert("RGB")
    query_start_time = time.perf_counter()
    reference_image = reference_ctx["reference_image"]
    reference_bnd_points = reference_ctx["reference_bnd_points"]
    ref_best_xywh = reference_ctx["ref_best_xywh"]
    text_prompt, original_prompt, translated_prompt, was_translated = prepare_single_text_prompt(prompt_text)
    native_score_threshold = _native_visual_prompt_score_threshold(sam_threshold)
    max_visual_prompt_candidates = _visual_prompt_candidate_limit(top_k)
    reference_prompt_encode_start = time.perf_counter()
    reference_features = set_ultralytics_image_features(reference_image)
    _, reference_img_feats, reference_img_pos_embeds, reference_feat_sizes = _prepare_sam3_backbone_features(
        reference_features,
        batch=1,
    )
    reference_prompt = _build_sam3_geometric_prompt_from_box(
        ref_best_xywh,
        reference_image.width,
        reference_image.height,
    )
    reference_visual_prompt_embed, reference_visual_prompt_mask = predictor.model._encode_prompt(
        reference_img_feats,
        reference_img_pos_embeds,
        reference_feat_sizes,
        reference_prompt,
    )
    reference_prompt_encode_ms = _elapsed_ms(reference_prompt_encode_start)

    set_query_start = time.perf_counter()
    query_features = set_ultralytics_image_features(query_image)
    set_query_ms = _elapsed_ms(set_query_start)

    prompt_start = time.perf_counter()
    arrays, native_profile = _run_sam3_query_grounding_with_visual_prompt_embeddings(
        query_image=query_image,
        query_features=query_features,
        visual_prompt_embed=reference_visual_prompt_embed,
        visual_prompt_mask=reference_visual_prompt_mask,
        text_prompt=text_prompt,
        confidence_threshold=native_score_threshold,
        max_candidates=max_visual_prompt_candidates,
    )
    prompt_forward_ms = _elapsed_ms(prompt_start)

    matched_labels: List[Dict[str, Any]] = []
    matched_masks: List[np.ndarray] = []
    matched_boxes_xyxy: List[List[float]] = []
    matched_scores: List[float] = []
    candidate_loop_start = time.perf_counter()
    for candidate in _iter_primary_mask_candidates(
        arrays,
        query_image,
        score_threshold=native_score_threshold,
    ):
        one_box_xywh = candidate["bnd_points"]
        one_score = candidate["score"]
        if any(bbox_iou_xywh(existing["bnd_points"], one_box_xywh) > 0.75 for existing in matched_labels):
            continue

        primary_u8 = candidate["primary_mask"]
        matched_labels.append(
            {
                "category": original_prompt or "similar_object",
                "translated_category": translated_prompt if was_translated else None,
                **_sam_grounding_score_fields(one_score),
                "bnd_points": one_box_xywh,
                "mask_area": candidate["mask_area"],
            }
        )
        matched_masks.append(primary_u8.astype(np.float32))
        matched_boxes_xyxy.append(candidate["box_xyxy"])
        matched_scores.append(max(0.0, min(1.0, one_score)))

    candidate_loop_ms = _elapsed_ms(candidate_loop_start)
    order = sorted(range(len(matched_labels)), key=lambda i: matched_labels[i]["combined_score"], reverse=True)[:top_k]
    matched_labels = [matched_labels[i] for i in order]
    matched_masks = [matched_masks[i] for i in order]
    matched_boxes_xyxy = [matched_boxes_xyxy[i] for i in order]
    matched_scores = [matched_scores[i] for i in order]
    for label, mask in zip(matched_labels, matched_masks):
        polygons = mask_to_polygons(mask, epsilon=polygon_simplify_epsilon)
        label["polygon_points"] = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(label["bnd_points"])

    result_image = _visualize_single_detection_group(
        query_image,
        class_name="similar_object",
        original_class_name=original_prompt or "similar_object",
        masks=matched_masks,
        boxes=matched_boxes_xyxy,
        scores=matched_scores,
    )

    processing_time_ms = _elapsed_ms(query_start_time)
    return {
        "model": MODEL_LABEL,
        "pic_id": pic_id,
        "success": True,
        "similar_mode": "feature_match",
        "prompt": original_prompt,
        "translated_prompt": translated_prompt if was_translated else None,
        "was_translated": was_translated,
        "box_text_prompt_enabled": text_prompt is not None,
        "reference_bnd_points": [round(float(v), 3) for v in reference_bnd_points],
        "reference_box_auto_generated": False,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "num_candidates": native_profile["raw_candidate_count"],
        "num_matches": len(matched_labels),
        "pic_labels": matched_labels,
        "reference_result_image": reference_ctx["reference_result_image"],
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
        "profile": {
            "prompt_forward_ms": prompt_forward_ms,
            "reference_prompt_encode_ms": reference_prompt_encode_ms,
            "set_query_ms": set_query_ms,
            "grounding_forward_ms": native_profile["grounding_forward_ms"],
            "candidate_loop_ms": candidate_loop_ms,
            "raw_visual_prompt_candidates": native_profile["raw_candidate_count"],
            "post_nms_candidates": native_profile["kept_candidate_count"],
            "max_visual_prompt_candidates": max_visual_prompt_candidates,
            "native_cross_image_visual_prompt": True,
            "native_score_threshold": round(native_score_threshold, 6),
            "similarity_threshold_applied": False,
            "score_type": "sam_grounding_score",
        },
    }


def mask_to_xyxy(mask_2d: np.ndarray) -> Optional[List[float]]:
    """从二值 mask 计算最小外接 xyxy 框；空 mask 返回 None。"""
    mask_np = np.asarray(mask_2d)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return None
    ys, xs = np.where(mask_np > 0.5)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [x1, y1, x2, y2]


def _nms_multi_similar_records(
    records: List[Dict[str, Any]],
    iou_threshold: float,
    top_k_per_category: int,
) -> List[Dict[str, Any]]:
    """multi prompt 结果按类别分别做 NMS，并限制每类 top-k。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["label"]["category"], []).append(record)

    kept_records: List[Dict[str, Any]] = []
    for category in sorted(grouped):
        category_records = sorted(
            grouped[category],
            key=lambda item: float(item["label"].get("combined_score", 0.0)),
            reverse=True,
        )
        kept_for_category: List[Dict[str, Any]] = []
        for record in category_records:
            if any(
                bbox_iou_xywh(existing["label"]["bnd_points"], record["label"]["bnd_points"]) > iou_threshold
                for existing in kept_for_category
            ):
                continue
            kept_for_category.append(record)
            if len(kept_for_category) >= top_k_per_category:
                break
        kept_records.extend(kept_for_category)

    kept_records.sort(key=lambda item: float(item["label"].get("combined_score", 0.0)), reverse=True)
    return kept_records


def _normalize_sample_type(sample: Dict[str, Any]) -> str:
    """把 sample_type/role/is_negative 统一归一成 positive 或 negative。"""
    raw_sample_type = _normalize_prompt_label(str(sample.get("sample_type", sample.get("role", "")) or "")).lower()
    if raw_sample_type in {"negative", "neg"} or sample.get("is_negative") is True:
        return "negative"
    return "positive"


def _normalize_multi_sample_inputs(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """校验 multi-similar 输入样本，并归一化字段类型和默认值。"""
    normalized_samples: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        category = _normalize_prompt_label(str(sample.get("category", "")))
        if not category:
            raise ValueError(f"samples[{index - 1}].category is required")
        reference_image = sample.get("reference_image")
        if not isinstance(reference_image, Image.Image):
            raise ValueError(f"samples[{index - 1}].reference_image is required")
        reference_bnd_points = sample.get("reference_bnd_points")
        if not isinstance(reference_bnd_points, list) or len(reference_bnd_points) != 4:
            raise ValueError(f"samples[{index - 1}].reference_bnd_points must be [x, y, w, h]")
        sample_id = _normalize_prompt_label(str(sample.get("sample_id", "") or "")) or f"sample_{index}"
        paste_bnd_points = sample.get("paste_bnd_points")
        if paste_bnd_points is not None:
            if not isinstance(paste_bnd_points, list) or len(paste_bnd_points) != 4:
                raise ValueError(f"samples[{index - 1}].paste_bnd_points must be [x, y, w, h]")
            paste_bnd_points = [float(v) for v in paste_bnd_points]
        normalized_samples.append(
            {
                "sample_id": sample_id,
                "source_image_id": _normalize_prompt_label(str(sample.get("source_image_id", "") or "")) or sample_id,
                "source_file_index": sample.get("source_file_index"),
                "category": category,
                "sample_type": _normalize_sample_type(sample),
                "reference_image": reference_image.convert("RGB"),
                "reference_bnd_points": [float(v) for v in reference_bnd_points],
                "paste_bnd_points": paste_bnd_points,
                "prompt": _normalize_prompt_label(str(sample.get("prompt", "") or "")) or None,
            }
        )
    return normalized_samples


def _run_same_image_prompt_query(
    image: Image.Image,
    reference_bnd_points: List[float],
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
) -> Dict[str, Any]:
    """同图模式：参考框和待搜索目标在同一张图内，只按 SAM 分数过滤。"""
    image = image.convert("RGB")
    query_start_time = time.perf_counter()
    reference_bnd_points = clip_bnd_points_to_image(reference_bnd_points, image.width, image.height)
    reference_xyxy = bbox_xywh_to_xyxy(reference_bnd_points)

    set_image_start = time.perf_counter()
    set_ultralytics_image_features(image)
    set_image_ms = _elapsed_ms(set_image_start)

    prompt_start = time.perf_counter()
    result = run_ultralytics_prediction(
        image,
        bboxes=[reference_xyxy],
        confidence_threshold=0.0,
        reset_cached_image=False,
    )
    prompt_forward_ms = _elapsed_ms(prompt_start)
    arrays = extract_ultralytics_arrays(result)
    masks_np = arrays["masks"]
    boxes_np = arrays["boxes"]
    scores_np = arrays["scores"]

    matched_labels: List[Dict[str, Any]] = []
    matched_masks: List[np.ndarray] = []
    matched_boxes_xyxy: List[List[float]] = []
    matched_scores: List[float] = []
    reference_result_image = None

    if scores_np.size > 0 and masks_np.shape[0] > 0:
        best_ref_idx = 0
        best_ref_overlap = -1.0
        for idx in range(masks_np.shape[0]):
            one_box_xyxy = (
                [float(v) for v in boxes_np[idx]]
                if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > idx
                else reference_xyxy
            )
            one_box_xywh = bbox_to_xywh(np.asarray(clip_xyxy_to_image(one_box_xyxy, image.width, image.height)))
            overlap = bbox_iou_xywh(reference_bnd_points, one_box_xywh)
            if overlap > best_ref_overlap:
                best_ref_overlap = overlap
                best_ref_idx = idx

        ref_mask = np.asarray(masks_np[best_ref_idx])
        ref_primary_u8 = _extract_primary_component_mask(ref_mask, reference_bnd_points)
        if ref_primary_u8.max() > 0:
            ref_box_xyxy = mask_to_xyxy(ref_primary_u8)
            if ref_box_xyxy is None:
                ref_box_xyxy = reference_xyxy
            ref_detection_for_viz = [
                {
                    "class_name": "reference_object",
                    "original_class_name": "reference_object",
                    "masks": np.asarray([ref_primary_u8.astype(np.float32)]),
                    "boxes": np.asarray([clip_xyxy_to_image(ref_box_xyxy, image.width, image.height)]),
                    "scores": np.asarray([max(0.0, min(1.0, float(scores_np[best_ref_idx])))]),
                    "color": build_class_color(1),
                }
            ]
            reference_result_image = visualize_results(image, ref_detection_for_viz)

    for candidate in _iter_primary_mask_candidates(arrays, image, fallback_box_xyxy=reference_xyxy):
        primary_u8 = candidate["primary_mask"]
        one_box_xywh = candidate["bnd_points"]
        sam_score = candidate["score"]
        if sam_score < sam_threshold:
            continue
        if any(bbox_iou_xywh(existing["bnd_points"], one_box_xywh) > 0.75 for existing in matched_labels):
            continue

        matched_labels.append(
            {
                "category": "similar_object",
                **_sam_grounding_score_fields(sam_score),
                "bnd_points": one_box_xywh,
                "mask_area": candidate["mask_area"],
                "is_reference_overlap": bbox_iou_xywh(reference_bnd_points, one_box_xywh) > 0.5,
            }
        )
        matched_masks.append(primary_u8.astype(np.float32))
        matched_boxes_xyxy.append(candidate["box_xyxy"])
        matched_scores.append(max(0.0, min(1.0, sam_score)))

    order = sorted(range(len(matched_labels)), key=lambda i: matched_labels[i]["combined_score"], reverse=True)[:top_k]
    matched_labels = [matched_labels[i] for i in order]
    matched_masks = [matched_masks[i] for i in order]
    matched_boxes_xyxy = [matched_boxes_xyxy[i] for i in order]
    matched_scores = [matched_scores[i] for i in order]
    for label, mask in zip(matched_labels, matched_masks):
        polygons = mask_to_polygons(mask, epsilon=polygon_simplify_epsilon)
        label["polygon_points"] = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(label["bnd_points"])

    result_image = _visualize_single_detection_group(
        image,
        class_name="similar_object",
        masks=matched_masks,
        boxes=matched_boxes_xyxy,
        scores=matched_scores,
    )

    processing_time_ms = _elapsed_ms(query_start_time)
    return {
        "model": MODEL_LABEL,
        "pic_id": pic_id,
        "success": True,
        "similar_mode": "same_image_prompt",
        "reference_bnd_points": [round(float(v), 3) for v in reference_bnd_points],
        "reference_box_auto_generated": False,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "num_candidates": int(scores_np.size),
        "num_matches": len(matched_labels),
        "pic_labels": matched_labels,
        "reference_result_image": reference_result_image,
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
        "profile": {
            "set_image_ms": set_image_ms,
            "prompt_forward_ms": prompt_forward_ms,
            "raw_visual_prompt_candidates": int(scores_np.size),
            "native_ultralytics_sam3_visual_prompt": True,
            "similarity_threshold_applied": False,
            "score_type": "sam_grounding_score",
        },
    }


def run_similar_object_pipeline(
    reference_image: Image.Image,
    query_image: Image.Image,
    reference_bnd_points: Optional[List[float]],
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
    similar_mode: str = "feature_match",
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """单参考图、单查询图的相似目标检测入口。"""
    start_time = time.perf_counter()
    normalized_pic_id = _normalize_pic_id(pic_id)
    with _model_inference_context():
        if similar_mode == "same_image_prompt":
            result = _run_same_image_prompt_query(
                reference_image,
                reference_bnd_points,
                top_k,
                similarity_threshold,
                sam_threshold,
                polygon_simplify_epsilon,
                normalized_pic_id,
            )
        else:
            reference_ctx = _prepare_similar_reference_context(reference_image, reference_bnd_points, top_k)
            result = _run_similar_query_with_reference_context(
                reference_ctx,
                query_image,
                top_k,
                similarity_threshold,
                sam_threshold,
                polygon_simplify_epsilon,
                normalized_pic_id,
                prompt,
            )

    _maybe_empty_cuda_cache()
    result["processing_time_ms"] = _elapsed_ms(start_time)
    return result


def run_similar_object_batch_pipeline(
    reference_image: Image.Image,
    query_images: List[Image.Image],
    reference_bnd_points: Optional[List[float]],
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
    query_names: Optional[List[str]] = None,
    similar_mode: str = "feature_match",
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """单参考目标、多查询图的 batch 相似目标检测入口。"""
    if similar_mode not in SIMILAR_MODES:
        raise ValueError(f"similar_mode must be one of: {', '.join(sorted(SIMILAR_MODES))}")

    if similar_mode == "same_image_prompt":
        start_time = time.perf_counter()
        normalized_pic_id = _normalize_pic_id(pic_id)
        with _model_inference_context():
            result = _run_same_image_prompt_query(
                reference_image,
                reference_bnd_points,
                top_k,
                similarity_threshold,
                sam_threshold,
                polygon_simplify_epsilon,
                normalized_pic_id,
            )
        _attach_query_name(result, query_names, 0)
        _maybe_empty_cuda_cache()
        total_processing_ms = _elapsed_ms(start_time)
        return _single_query_batch_response([dict(result)], total_processing_ms)

    if not query_images:
        raise ValueError("At least one query image is required")

    start_time = time.perf_counter()
    normalized_pic_id = _normalize_pic_id(pic_id)
    query_results: List[Dict[str, Any]] = []
    with _model_inference_context():
        reference_ctx = _prepare_similar_reference_context(reference_image, reference_bnd_points, top_k)
        for index, query_image in enumerate(query_images):
            query_pic_id = _query_pic_id(normalized_pic_id, index, len(query_images))
            one_result = _run_similar_query_with_reference_context(
                reference_ctx,
                query_image,
                top_k,
                similarity_threshold,
                sam_threshold,
                polygon_simplify_epsilon,
                query_pic_id,
                prompt,
            )
            _attach_query_name(one_result, query_names, index)
            query_results.append(one_result)

    _maybe_empty_cuda_cache()

    total_processing_ms = _elapsed_ms(start_time)
    if len(query_results) == 1:
        return _single_query_batch_response(query_results, total_processing_ms)

    return {
        "model": MODEL_LABEL,
        "pic_id": normalized_pic_id,
        "success": True,
        "batch_size": len(query_results),
        "query_results": query_results,
        "reference_bnd_points": query_results[0].get("reference_bnd_points") if query_results else None,
        "reference_box_auto_generated": False,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "similar_mode": similar_mode,
        "total_num_matches": sum(int(item.get("num_matches", 0)) for item in query_results),
        "total_num_candidates": sum(int(item.get("num_candidates", 0)) for item in query_results),
        "reference_result_image": query_results[0].get("reference_result_image") if query_results else None,
        "created": int(time.time()),
        "processing_time_ms": total_processing_ms,
    }


def _prepare_multi_similar_reference_contexts(
    samples: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """为 multi-similar 的每个样本准备参考 mask、特征、prompt 和元数据。"""
    normalized_samples = _normalize_multi_sample_inputs(samples)
    grouped_samples: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    ordered_group_keys: List[Tuple[str, Any]] = []
    for sample in normalized_samples:
        group_key = _reference_source_group_key(sample)
        if group_key not in grouped_samples:
            grouped_samples[group_key] = []
            ordered_group_keys.append(group_key)
        grouped_samples[group_key].append(sample)

    sample_contexts: List[Dict[str, Any]] = []
    for group_key in ordered_group_keys:
        group_samples = grouped_samples[group_key]
        reference_contexts = _prepare_multi_reference_contexts_for_one_image(group_samples, top_k)
        sample_contexts.extend(reference_contexts)
    if not any(sample_ctx.get("sample_type") != "negative" for sample_ctx in sample_contexts):
        raise ValueError("At least one positive sample is required")
    return sample_contexts


def _reference_source_group_key(sample: Dict[str, Any]) -> Tuple[str, Any]:
    """优先按上传文件索引聚合同源样例，避免同图多实例被 source_image_id 拆开。"""
    source_file_index = sample.get("source_file_index")
    if source_file_index is not None:
        return ("file", source_file_index)
    return ("source", sample.get("source_image_id", sample.get("sample_id", "sample")))


def _prepare_multi_reference_contexts_for_one_image(
    samples: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """同一张样例图只提一次特征，再用多个框批量生成参考上下文。"""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not samples:
        return []

    reference_image = samples[0]["reference_image"].convert("RGB")
    clipped_boxes: List[List[float]] = []
    xyxy_boxes: List[List[float]] = []
    for sample in samples:
        reference_bnd_points = clip_bnd_points_to_image(
            bnd_points=sample["reference_bnd_points"],
            image_width=reference_image.width,
            image_height=reference_image.height,
        )
        clipped_boxes.append(reference_bnd_points)
        xyxy_boxes.append(bbox_xywh_to_xyxy(reference_bnd_points))

    reference_features = set_ultralytics_image_features(reference_image)
    result = run_ultralytics_prediction(
        reference_image,
        bboxes=xyxy_boxes,
        confidence_threshold=0.0,
        reset_cached_image=False,
    )
    arrays = extract_ultralytics_arrays(result)
    matched_result_indices = _match_reference_prediction_indices(
        clipped_boxes,
        arrays,
        reference_image.width,
        reference_image.height,
    )

    reference_contexts: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        reference_ctx = _build_similar_reference_context_from_arrays(
            reference_image=reference_image,
            reference_bnd_points=clipped_boxes[index],
            arrays=arrays,
            result_index=matched_result_indices[index],
            reference_features=reference_features,
            build_reference_result_image=False,
        )
        text_prompt, original_prompt, translated_prompt, was_translated = prepare_single_text_prompt(sample.get("prompt"))
        reference_ctx.update(
            {
                "sample_id": sample["sample_id"],
                "source_image_id": sample.get("source_image_id", sample["sample_id"]),
                "source_file_index": sample.get("source_file_index"),
                "category": sample["category"],
                "sample_type": sample["sample_type"],
                "is_negative": sample["sample_type"] == "negative",
                "paste_bnd_points": sample.get("paste_bnd_points"),
                "effective_paste_bnd_points": sample.get("paste_bnd_points"),
                "reference_preprocess_group_size": len(samples),
                "prompt": original_prompt,
                "translated_prompt": translated_prompt if was_translated else None,
                "was_translated": was_translated,
                "text_prompt": text_prompt,
            }
        )
        reference_contexts.append(reference_ctx)
    reference_result_image = _visualize_multi_reference_contexts(reference_image, reference_contexts)
    for reference_ctx in reference_contexts:
        reference_ctx["reference_result_image"] = reference_result_image
    return reference_contexts


def _format_multi_prompt_source_label(values: List[str]) -> str:
    """把多个 sample/source id 压缩成适合展示的短标签。"""
    unique_values: List[str] = []
    for one_value in values:
        normalized = str(one_value or "").strip()
        if not normalized or normalized in unique_values:
            continue
        unique_values.append(normalized)
    if not unique_values:
        return "-"
    if len(unique_values) <= 3:
        return " | ".join(unique_values)
    return " | ".join(unique_values[:3]) + f" | ...(+{len(unique_values) - 3})"


def _resolve_multi_group_text_prompt(
    group_contexts: List[Dict[str, Any]],
) -> Tuple[Optional[List[str]], Optional[str], Optional[str], bool]:
    """为同一类别的一组正样本决定是否可以共用一个文本 prompt。"""
    prompt_pairs: List[Tuple[str, str]] = []
    for sample_ctx in group_contexts:
        original_prompt = _normalize_prompt_label(sample_ctx.get("prompt") or "")
        if not original_prompt:
            continue
        translated_prompt = _normalize_prompt_label(sample_ctx.get("translated_prompt") or original_prompt).lower()
        if not translated_prompt:
            translated_prompt = original_prompt
        prompt_pairs.append((original_prompt, translated_prompt))

    if not prompt_pairs:
        return None, None, None, False

    unique_original: List[str] = []
    unique_translated: List[str] = []
    for original_prompt, translated_prompt in prompt_pairs:
        if original_prompt not in unique_original:
            unique_original.append(original_prompt)
        if translated_prompt not in unique_translated:
            unique_translated.append(translated_prompt)

    if len(unique_translated) == 1:
        translated_prompt = unique_translated[0]
        original_prompt = unique_original[0]
        return [translated_prompt], original_prompt, translated_prompt, translated_prompt != original_prompt

    original_prompt_joined = "; ".join(unique_original)
    translated_prompt_joined = "; ".join(unique_translated)
    return None, original_prompt_joined, translated_prompt_joined, translated_prompt_joined != original_prompt_joined


def _group_multi_sample_contexts_for_native_prompt(sample_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按类别和正负样本分组，并按参考图片聚合 visual prompt 框。"""
    grouped: Dict[str, Dict[str, Any]] = {}
    ordered_groups: List[Dict[str, Any]] = []

    for sample_ctx in sample_contexts:
        category = sample_ctx["category"]
        group = grouped.get(category)
        if group is None:
            group = {
                "category": category,
                "sample_contexts": [],
                "positive_sample_contexts": [],
                "negative_sample_contexts": [],
                "sample_ids": [],
                "source_image_ids": [],
                "source_groups": {},
                "negative_sample_ids": [],
                "negative_source_image_ids": [],
                "negative_source_groups": {},
            }
            grouped[category] = group
            ordered_groups.append(group)

        group["sample_contexts"].append(sample_ctx)
        source_image_id = sample_ctx.get("source_image_id", sample_ctx["sample_id"])
        source_group_key = _reference_source_group_key(sample_ctx)
        is_negative = sample_ctx.get("sample_type") == "negative" or sample_ctx.get("is_negative") is True
        if is_negative:
            group["negative_sample_contexts"].append(sample_ctx)
            group["negative_sample_ids"].append(sample_ctx["sample_id"])
            group["negative_source_image_ids"].append(source_image_id)
            source_groups = group["negative_source_groups"]
        else:
            group["positive_sample_contexts"].append(sample_ctx)
            group["sample_ids"].append(sample_ctx["sample_id"])
            group["source_image_ids"].append(source_image_id)
            source_groups = group["source_groups"]
        source_group = source_groups.get(source_group_key)
        if source_group is None:
            source_group = {
                "source_image_id": source_image_id,
                "source_group_key": source_group_key,
                "reference_image": sample_ctx["reference_image"],
                "reference_features": sample_ctx.get("reference_features"),
                "reference_boxes": [],
                "sample_ids": [],
            }
            source_groups[source_group_key] = source_group
        source_group["reference_boxes"].append(sample_ctx["ref_best_xywh"])
        source_group["sample_ids"].append(sample_ctx["sample_id"])

    finalized_groups: List[Dict[str, Any]] = []
    for group in ordered_groups:
        if not group["positive_sample_contexts"]:
            continue
        text_prompt, original_prompt, translated_prompt, was_translated = _resolve_multi_group_text_prompt(
            group["positive_sample_contexts"]
        )
        finalized_groups.append(
            {
                "category": group["category"],
                "sample_contexts": group["positive_sample_contexts"],
                "negative_sample_contexts": group["negative_sample_contexts"],
                "sample_ids": group["sample_ids"],
                "source_image_ids": group["source_image_ids"],
                "sample_id_display": _format_multi_prompt_source_label(group["sample_ids"]),
                "source_image_id_display": _format_multi_prompt_source_label(group["source_image_ids"]),
                "source_groups": list(group["source_groups"].values()),
                "negative_sample_ids": group["negative_sample_ids"],
                "negative_source_image_ids": group["negative_source_image_ids"],
                "negative_source_groups": list(group["negative_source_groups"].values()),
                "text_prompt": text_prompt,
                "prompt": original_prompt,
                "translated_prompt": translated_prompt if was_translated else None,
                "was_translated": was_translated,
            }
        )
    return finalized_groups


def _collect_negative_visual_prompt_boxes(
    group: Dict[str, Any],
    query_image: Image.Image,
    query_features: Dict[str, Any],
    native_score_threshold: float,
    max_visual_prompt_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """用负样本 visual prompt 找出需要从正样本结果中抑制的区域。"""
    negative_source_groups = group.get("negative_source_groups") or []
    profile = {
        "negative_reference_prompt_encode_ms": 0,
        "negative_grounding_forward_ms": 0,
        "negative_raw_candidates": 0,
        "negative_post_nms_candidates": 0,
        "negative_filter_candidates": 0,
    }
    if not negative_source_groups:
        return [], profile

    visual_prompt_embeds: List[torch.Tensor] = []
    visual_prompt_masks: List[torch.Tensor] = []
    for source_group in negative_source_groups:
        prompt_embed, prompt_mask, prompt_profile = _encode_reference_visual_prompt_from_boxes(
            source_group["reference_image"],
            source_group["reference_boxes"],
            source_group.get("reference_features"),
        )
        profile["negative_reference_prompt_encode_ms"] += int(prompt_profile["reference_prompt_encode_ms"])
        visual_prompt_embeds.append(prompt_embed)
        visual_prompt_masks.append(prompt_mask)

    if not visual_prompt_embeds:
        return [], profile

    merged_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
    merged_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
    arrays, negative_profile = _run_sam3_query_grounding_with_visual_prompt_embeddings(
        query_image=query_image,
        query_features=query_features,
        visual_prompt_embed=merged_prompt_embed,
        visual_prompt_mask=merged_prompt_mask,
        text_prompt=group.get("text_prompt"),
        confidence_threshold=native_score_threshold,
        max_candidates=max_visual_prompt_candidates,
    )
    profile["negative_grounding_forward_ms"] += int(negative_profile["grounding_forward_ms"])
    profile["negative_raw_candidates"] += int(negative_profile["raw_candidate_count"])
    profile["negative_post_nms_candidates"] += int(negative_profile["kept_candidate_count"])

    negative_records: List[Dict[str, Any]] = []
    for candidate in _iter_primary_mask_candidates(
        arrays,
        query_image,
        score_threshold=native_score_threshold,
    ):
        negative_records.append(
            {
                "bnd_points": candidate["bnd_points"],
                "box_xyxy": candidate["box_xyxy"],
                "score": max(0.0, min(1.0, candidate["score"])),
            }
        )

    profile["negative_filter_candidates"] = len(negative_records)
    return negative_records, profile


def _collect_negative_visual_prompt_boxes_from_group_state(
    group: Dict[str, Any],
    query_image: Image.Image,
    query_features: Dict[str, Any],
    native_score_threshold: float,
    max_visual_prompt_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Use pre-encoded negative visual prompts for one query image."""
    profile = {
        "negative_reference_prompt_encode_ms": 0,
        "negative_grounding_forward_ms": 0,
        "negative_raw_candidates": 0,
        "negative_post_nms_candidates": 0,
        "negative_filter_candidates": 0,
    }
    visual_prompt_embed = group.get("negative_visual_prompt_embed")
    visual_prompt_mask = group.get("negative_visual_prompt_mask")
    if visual_prompt_embed is None or visual_prompt_mask is None:
        return [], profile

    arrays, negative_profile = _run_sam3_query_grounding_with_visual_prompt_embeddings(
        query_image=query_image,
        query_features=query_features,
        visual_prompt_embed=visual_prompt_embed,
        visual_prompt_mask=visual_prompt_mask,
        text_prompt=group.get("text_prompt"),
        confidence_threshold=native_score_threshold,
        max_candidates=max_visual_prompt_candidates,
    )
    profile["negative_grounding_forward_ms"] += int(negative_profile["grounding_forward_ms"])
    profile["negative_raw_candidates"] += int(negative_profile["raw_candidate_count"])
    profile["negative_post_nms_candidates"] += int(negative_profile["kept_candidate_count"])

    negative_records: List[Dict[str, Any]] = []
    for candidate in _iter_primary_mask_candidates(
        arrays,
        query_image,
        score_threshold=native_score_threshold,
    ):
        negative_records.append(
            {
                "bnd_points": candidate["bnd_points"],
                "box_xyxy": candidate["box_xyxy"],
                "score": max(0.0, min(1.0, candidate["score"])),
            }
        )

    profile["negative_filter_candidates"] = len(negative_records)
    return negative_records, profile


def _is_suppressed_by_negative_sample(
    candidate_bnd_points: List[float],
    negative_records: List[Dict[str, Any]],
    iou_threshold: float,
) -> bool:
    """判断候选框是否和负样本区域重叠过高，需要过滤。"""
    return any(
        bbox_iou_xywh(candidate_bnd_points, negative_record["bnd_points"]) >= iou_threshold
        for negative_record in negative_records
    )


def prepare_multi_visual_prompt_state(samples: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    """Prepare reusable visual prompt embeddings for URL/task based sample annotation."""
    start_time = time.perf_counter()
    with _model_inference_context():
        sample_contexts = _prepare_multi_similar_reference_contexts(samples, top_k)
        grouped_prompts = _group_multi_sample_contexts_for_native_prompt(sample_contexts)
        prepared_groups: List[Dict[str, Any]] = []
        total_reference_prompt_encode_ms = 0
        total_negative_reference_prompt_encode_ms = 0

        for group in grouped_prompts:
            visual_prompt_embeds: List[torch.Tensor] = []
            visual_prompt_masks: List[torch.Tensor] = []
            group_prompt_encode_ms = 0
            for source_group in group["source_groups"]:
                prompt_embed, prompt_mask, prompt_profile = _encode_reference_visual_prompt_from_boxes(
                    source_group["reference_image"],
                    source_group["reference_boxes"],
                    source_group.get("reference_features"),
                )
                group_prompt_encode_ms += int(prompt_profile["reference_prompt_encode_ms"])
                visual_prompt_embeds.append(prompt_embed)
                visual_prompt_masks.append(prompt_mask)
            if not visual_prompt_embeds:
                continue

            prepared_group = dict(group)
            prepared_group["visual_prompt_embed"] = torch.cat(visual_prompt_embeds, dim=0)
            prepared_group["visual_prompt_mask"] = torch.cat(visual_prompt_masks, dim=1)
            prepared_group["reference_prompt_encode_ms"] = group_prompt_encode_ms
            total_reference_prompt_encode_ms += group_prompt_encode_ms

            negative_prompt_embeds: List[torch.Tensor] = []
            negative_prompt_masks: List[torch.Tensor] = []
            negative_prompt_encode_ms = 0
            for source_group in group.get("negative_source_groups") or []:
                prompt_embed, prompt_mask, prompt_profile = _encode_reference_visual_prompt_from_boxes(
                    source_group["reference_image"],
                    source_group["reference_boxes"],
                    source_group.get("reference_features"),
                )
                negative_prompt_encode_ms += int(prompt_profile["reference_prompt_encode_ms"])
                negative_prompt_embeds.append(prompt_embed)
                negative_prompt_masks.append(prompt_mask)
            if negative_prompt_embeds:
                prepared_group["negative_visual_prompt_embed"] = torch.cat(negative_prompt_embeds, dim=0)
                prepared_group["negative_visual_prompt_mask"] = torch.cat(negative_prompt_masks, dim=1)
            prepared_group["negative_reference_prompt_encode_ms"] = negative_prompt_encode_ms
            total_negative_reference_prompt_encode_ms += negative_prompt_encode_ms
            prepared_groups.append(prepared_group)

    positive_sample_count = sum(1 for sample_ctx in sample_contexts if sample_ctx.get("sample_type") != "negative")
    negative_sample_count = sum(1 for sample_ctx in sample_contexts if sample_ctx.get("sample_type") == "negative")
    reference_source_keys = {_reference_source_group_key(sample_ctx) for sample_ctx in sample_contexts}
    for sample_ctx in sample_contexts:
        sample_ctx.pop("reference_image", None)
        sample_ctx.pop("reference_mask", None)
        sample_ctx.pop("reference_features", None)
    for group in prepared_groups:
        for source_group in group.get("source_groups") or []:
            source_group.pop("reference_image", None)
            source_group.pop("reference_features", None)
        for source_group in group.get("negative_source_groups") or []:
            source_group.pop("reference_image", None)
            source_group.pop("reference_features", None)
        for sample_ctx in group.get("sample_contexts") or []:
            sample_ctx.pop("reference_image", None)
            sample_ctx.pop("reference_mask", None)
            sample_ctx.pop("reference_features", None)
        for sample_ctx in group.get("negative_sample_contexts") or []:
            sample_ctx.pop("reference_image", None)
            sample_ctx.pop("reference_mask", None)
            sample_ctx.pop("reference_features", None)
    return {
        "sample_contexts": sample_contexts,
        "prepared_groups": prepared_groups,
        "num_samples": len(sample_contexts),
        "num_positive_samples": positive_sample_count,
        "num_negative_samples": negative_sample_count,
        "num_groups": len(prepared_groups),
        "reference_source_count": len(reference_source_keys),
        "reference_prompt_encode_ms": total_reference_prompt_encode_ms,
        "negative_reference_prompt_encode_ms": total_negative_reference_prompt_encode_ms,
        "prepare_time_ms": _elapsed_ms(start_time),
    }


def _run_multi_native_visual_prompt_query(
    sample_contexts: List[Dict[str, Any]],
    query_image: Image.Image,
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    nms_iou: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
) -> Dict[str, Any]:
    """multi-similar 核心查询：按类别合并多个 visual prompt，并应用负样本过滤。"""
    query_image = query_image.convert("RGB")
    query_start_time = time.perf_counter()
    grouped_prompts = _group_multi_sample_contexts_for_native_prompt(sample_contexts)
    native_score_threshold = _native_visual_prompt_score_threshold(sam_threshold)
    max_visual_prompt_candidates = _multi_visual_prompt_candidate_limit(top_k, len(grouped_prompts))

    set_query_start = time.perf_counter()
    query_features = set_ultralytics_image_features(query_image)
    set_query_ms = _elapsed_ms(set_query_start)

    candidate_records: List[Dict[str, Any]] = []
    raw_candidate_count = 0
    post_nms_candidate_count = 0
    total_reference_prompt_encode_ms = 0
    total_grounding_forward_ms = 0
    total_candidate_loop_ms = 0
    total_negative_reference_prompt_encode_ms = 0
    total_negative_grounding_forward_ms = 0
    total_negative_raw_candidate_count = 0
    total_negative_post_nms_candidate_count = 0
    total_negative_filter_candidate_count = 0
    total_suppressed_by_negative_count = 0
    group_profiles: List[Dict[str, Any]] = []

    for group in grouped_prompts:
        negative_records, negative_profile = _collect_negative_visual_prompt_boxes(
            group,
            query_image,
            query_features,
            native_score_threshold,
            max_visual_prompt_candidates,
        )
        total_negative_reference_prompt_encode_ms += int(negative_profile["negative_reference_prompt_encode_ms"])
        total_negative_grounding_forward_ms += int(negative_profile["negative_grounding_forward_ms"])
        total_negative_raw_candidate_count += int(negative_profile["negative_raw_candidates"])
        total_negative_post_nms_candidate_count += int(negative_profile["negative_post_nms_candidates"])
        total_negative_filter_candidate_count += int(negative_profile["negative_filter_candidates"])
        suppressed_by_negative_count = 0

        visual_prompt_embeds: List[torch.Tensor] = []
        visual_prompt_masks: List[torch.Tensor] = []
        group_prompt_encode_ms = 0
        for source_group in group["source_groups"]:
            prompt_embed, prompt_mask, prompt_profile = _encode_reference_visual_prompt_from_boxes(
                source_group["reference_image"],
                source_group["reference_boxes"],
                source_group.get("reference_features"),
            )
            group_prompt_encode_ms += int(prompt_profile["reference_prompt_encode_ms"])
            visual_prompt_embeds.append(prompt_embed)
            visual_prompt_masks.append(prompt_mask)
        if not visual_prompt_embeds:
            continue

        merged_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
        merged_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
        arrays, group_profile = _run_sam3_query_grounding_with_visual_prompt_embeddings(
            query_image=query_image,
            query_features=query_features,
            visual_prompt_embed=merged_prompt_embed,
            visual_prompt_mask=merged_prompt_mask,
            text_prompt=group["text_prompt"],
            confidence_threshold=native_score_threshold,
            max_candidates=max_visual_prompt_candidates,
        )
        group_candidate_loop_start = time.perf_counter()

        raw_candidate_count += int(group_profile["raw_candidate_count"])
        post_nms_candidate_count += int(group_profile["kept_candidate_count"])
        total_reference_prompt_encode_ms += group_prompt_encode_ms
        total_grounding_forward_ms += int(group_profile["grounding_forward_ms"])

        for candidate in _iter_primary_mask_candidates(
            arrays,
            query_image,
            score_threshold=native_score_threshold,
        ):
            one_box_xywh = candidate["bnd_points"]
            one_score = candidate["score"]
            if _is_suppressed_by_negative_sample(one_box_xywh, negative_records, MULTI_NEGATIVE_FILTER_IOU):
                suppressed_by_negative_count += 1
                total_suppressed_by_negative_count += 1
                continue

            primary_u8 = candidate["primary_mask"]
            label = {
                "category": group["category"],
                "sample_id": group["sample_id_display"],
                "sample_ids": group["sample_ids"],
                "source_image_id": group["source_image_id_display"],
                "source_image_ids": group["source_image_ids"],
                "prompt": group.get("prompt"),
                "translated_prompt": group.get("translated_prompt"),
                **_sam_grounding_score_fields(one_score),
                "bnd_points": one_box_xywh,
                "mask_area": candidate["mask_area"],
            }
            candidate_records.append(
                {
                    "label": label,
                    "mask": primary_u8.astype(np.float32),
                    "box_xyxy": candidate["box_xyxy"],
                    "score": max(0.0, min(1.0, one_score)),
                }
            )

        group_candidate_loop_ms = _elapsed_ms(group_candidate_loop_start)
        total_candidate_loop_ms += group_candidate_loop_ms
        group_profiles.append(
            {
                "category": group["category"],
                "num_samples": len(group["sample_contexts"]),
                "num_negative_samples": len(group.get("negative_sample_contexts") or []),
                "num_source_images": len(group["source_groups"]),
                "num_negative_source_images": len(group.get("negative_source_groups") or []),
                "reference_prompt_encode_ms": group_prompt_encode_ms,
                "grounding_forward_ms": int(group_profile["grounding_forward_ms"]),
                "raw_candidates": int(group_profile["raw_candidate_count"]),
                "post_nms_candidates": int(group_profile["kept_candidate_count"]),
                "negative_reference_prompt_encode_ms": int(negative_profile["negative_reference_prompt_encode_ms"]),
                "negative_grounding_forward_ms": int(negative_profile["negative_grounding_forward_ms"]),
                "negative_raw_candidates": int(negative_profile["negative_raw_candidates"]),
                "negative_post_nms_candidates": int(negative_profile["negative_post_nms_candidates"]),
                "negative_filter_candidates": int(negative_profile["negative_filter_candidates"]),
                "suppressed_by_negative_samples": suppressed_by_negative_count,
                "candidate_loop_ms": group_candidate_loop_ms,
                "max_visual_prompt_candidates": max_visual_prompt_candidates,
            }
        )

    kept_records = _nms_multi_similar_records(candidate_records, nms_iou, top_k)
    for record in kept_records:
        label = record["label"]
        polygons = mask_to_polygons(record["mask"], epsilon=polygon_simplify_epsilon)
        label["polygon_points"] = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(label["bnd_points"])
    matched_labels = [record["label"] for record in kept_records]

    result_image = None
    if kept_records:
        category_order: Dict[str, int] = {}
        for record in kept_records:
            category = record["label"]["category"]
            if category not in category_order:
                category_order[category] = len(category_order)
        detections_for_viz: List[Dict[str, Any]] = []
        for category, color_index in category_order.items():
            category_records = [record for record in kept_records if record["label"]["category"] == category]
            detections_for_viz.append(
                {
                    "class_name": category,
                    "original_class_name": category,
                    "masks": np.asarray([record["mask"] for record in category_records]),
                    "boxes": np.asarray([record["box_xyxy"] for record in category_records]),
                    "scores": np.asarray([record["score"] for record in category_records]),
                    "color": build_class_color(color_index),
                }
            )
        result_image = visualize_results(query_image, detections_for_viz)

    sample_instance_results = [
        {
            "sample_id": sample_ctx["sample_id"],
            "source_image_id": sample_ctx.get("source_image_id", sample_ctx["sample_id"]),
            "source_file_index": sample_ctx.get("source_file_index"),
            "category": sample_ctx["category"],
            "sample_type": sample_ctx.get("sample_type", "positive"),
            "is_negative": sample_ctx.get("sample_type") == "negative",
            "reference_bnd_points": [round(float(v), 3) for v in sample_ctx["reference_bnd_points"]],
            "paste_bnd_points": sample_ctx.get("effective_paste_bnd_points"),
            "reference_result_image": sample_ctx["reference_result_image"],
            "prompt": sample_ctx.get("prompt"),
            "translated_prompt": sample_ctx.get("translated_prompt"),
        }
        for sample_ctx in sample_contexts
    ]
    sample_result_images: List[Dict[str, Any]] = []
    seen_reference_sources = set()
    for sample_ctx in sample_contexts:
        source_key = _reference_source_group_key(sample_ctx)
        if source_key in seen_reference_sources:
            continue
        seen_reference_sources.add(source_key)
        source_contexts = [
            one_ctx
            for one_ctx in sample_contexts
            if _reference_source_group_key(one_ctx) == source_key
        ]
        sample_result_images.append(
            {
                "source_image_id": sample_ctx.get("source_image_id", sample_ctx["sample_id"]),
                "source_file_index": sample_ctx.get("source_file_index"),
                "reference_result_image": sample_ctx["reference_result_image"],
                "num_instances": len(source_contexts),
                "num_positive_samples": sum(1 for one_ctx in source_contexts if one_ctx.get("sample_type") != "negative"),
                "num_negative_samples": sum(1 for one_ctx in source_contexts if one_ctx.get("sample_type") == "negative"),
                "sample_ids": [one_ctx["sample_id"] for one_ctx in source_contexts],
            }
        )
    processing_time_ms = _elapsed_ms(query_start_time)
    category_counts: Dict[str, int] = {}
    for label in matched_labels:
        category_counts[label["category"]] = category_counts.get(label["category"], 0) + 1
    positive_sample_count = sum(1 for sample_ctx in sample_contexts if sample_ctx.get("sample_type") != "negative")
    negative_sample_count = sum(1 for sample_ctx in sample_contexts if sample_ctx.get("sample_type") == "negative")
    reference_source_keys = {_reference_source_group_key(sample_ctx) for sample_ctx in sample_contexts}

    return {
        "model": MODEL_LABEL,
        "pic_id": pic_id,
        "success": True,
        "similar_mode": "multi_visual_prompt",
        "top_k": int(top_k),
        "top_k_scope": "per_category",
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "nms_iou": float(nms_iou),
        "num_samples": len(sample_contexts),
        "num_positive_samples": positive_sample_count,
        "num_negative_samples": negative_sample_count,
        "num_groups": len(grouped_prompts),
        "num_candidates": raw_candidate_count,
        "num_matches": len(matched_labels),
        "category_counts": category_counts,
        "pic_labels": matched_labels,
        "sample_result_images": sample_result_images,
        "sample_instance_results": sample_instance_results,
        "reference_result_image": sample_result_images[0]["reference_result_image"] if sample_result_images else None,
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
        "profile": {
            "set_query_ms": set_query_ms,
            "reference_feature_reuse_enabled": True,
            "reference_feature_extract_count": len(reference_source_keys),
            "reference_sample_context_count": len(sample_contexts),
            "reference_prompt_encode_ms": total_reference_prompt_encode_ms,
            "grounding_forward_ms": total_grounding_forward_ms,
            "candidate_loop_ms": total_candidate_loop_ms,
            "negative_reference_prompt_encode_ms": total_negative_reference_prompt_encode_ms,
            "negative_grounding_forward_ms": total_negative_grounding_forward_ms,
            "raw_visual_prompt_candidates": raw_candidate_count,
            "post_nms_candidates": post_nms_candidate_count,
            "max_visual_prompt_candidates": max_visual_prompt_candidates,
            "negative_raw_visual_prompt_candidates": total_negative_raw_candidate_count,
            "negative_post_nms_candidates": total_negative_post_nms_candidate_count,
            "negative_filter_candidates": total_negative_filter_candidate_count,
            "suppressed_by_negative_samples": total_suppressed_by_negative_count,
            "negative_filter_iou": round(MULTI_NEGATIVE_FILTER_IOU, 6),
            "final_nms_iou": float(nms_iou),
            "native_multi_visual_prompt": True,
            "native_score_threshold": round(native_score_threshold, 6),
            "similarity_threshold_applied": False,
            "score_type": "sam_grounding_score",
            "groups": group_profiles,
        },
    }


def run_multi_visual_prompt_query_with_state(
    prompt_state: Dict[str, Any],
    query_image: Image.Image,
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    nms_iou: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
    *,
    return_result_image: bool = False,
) -> Dict[str, Any]:
    """Run one query image with pre-encoded multi visual prompt state."""
    query_image = query_image.convert("RGB")
    query_start_time = time.perf_counter()
    grouped_prompts = prompt_state.get("prepared_groups") or []
    if not grouped_prompts:
        raise ValueError("No prepared visual prompt groups available")

    native_score_threshold = _native_visual_prompt_score_threshold(sam_threshold)
    max_visual_prompt_candidates = _multi_visual_prompt_candidate_limit(top_k, len(grouped_prompts))

    candidate_records: List[Dict[str, Any]] = []
    raw_candidate_count = 0
    post_nms_candidate_count = 0
    total_grounding_forward_ms = 0
    total_candidate_loop_ms = 0
    total_negative_grounding_forward_ms = 0
    total_negative_raw_candidate_count = 0
    total_negative_post_nms_candidate_count = 0
    total_negative_filter_candidate_count = 0
    total_suppressed_by_negative_count = 0
    group_profiles: List[Dict[str, Any]] = []

    with _model_inference_context():
        set_query_start = time.perf_counter()
        query_features = set_ultralytics_image_features(query_image)
        set_query_ms = _elapsed_ms(set_query_start)

        for group in grouped_prompts:
            negative_records, negative_profile = _collect_negative_visual_prompt_boxes_from_group_state(
                group,
                query_image,
                query_features,
                native_score_threshold,
                max_visual_prompt_candidates,
            )
            total_negative_grounding_forward_ms += int(negative_profile["negative_grounding_forward_ms"])
            total_negative_raw_candidate_count += int(negative_profile["negative_raw_candidates"])
            total_negative_post_nms_candidate_count += int(negative_profile["negative_post_nms_candidates"])
            total_negative_filter_candidate_count += int(negative_profile["negative_filter_candidates"])
            suppressed_by_negative_count = 0

            arrays, group_profile = _run_sam3_query_grounding_with_visual_prompt_embeddings(
                query_image=query_image,
                query_features=query_features,
                visual_prompt_embed=group["visual_prompt_embed"],
                visual_prompt_mask=group["visual_prompt_mask"],
                text_prompt=group["text_prompt"],
                confidence_threshold=native_score_threshold,
                max_candidates=max_visual_prompt_candidates,
            )
            group_candidate_loop_start = time.perf_counter()
            raw_candidate_count += int(group_profile["raw_candidate_count"])
            post_nms_candidate_count += int(group_profile["kept_candidate_count"])
            total_grounding_forward_ms += int(group_profile["grounding_forward_ms"])

            for candidate in _iter_primary_mask_candidates(
                arrays,
                query_image,
                score_threshold=native_score_threshold,
            ):
                one_box_xywh = candidate["bnd_points"]
                one_score = candidate["score"]
                if _is_suppressed_by_negative_sample(one_box_xywh, negative_records, MULTI_NEGATIVE_FILTER_IOU):
                    suppressed_by_negative_count += 1
                    total_suppressed_by_negative_count += 1
                    continue

                primary_u8 = candidate["primary_mask"]
                label = {
                    "category": group["category"],
                    "sample_id": group["sample_id_display"],
                    "sample_ids": group["sample_ids"],
                    "source_image_id": group["source_image_id_display"],
                    "source_image_ids": group["source_image_ids"],
                    "prompt": group.get("prompt"),
                    "translated_prompt": group.get("translated_prompt"),
                    **_sam_grounding_score_fields(one_score),
                    "bnd_points": one_box_xywh,
                    "mask_area": candidate["mask_area"],
                }
                candidate_records.append(
                    {
                        "label": label,
                        "mask": primary_u8.astype(np.float32),
                        "box_xyxy": candidate["box_xyxy"],
                        "score": max(0.0, min(1.0, one_score)),
                    }
                )

            group_candidate_loop_ms = _elapsed_ms(group_candidate_loop_start)
            total_candidate_loop_ms += group_candidate_loop_ms
            group_profiles.append(
                {
                    "category": group["category"],
                    "num_samples": len(group["sample_contexts"]),
                    "num_negative_samples": len(group.get("negative_sample_contexts") or []),
                    "num_source_images": len(group["source_groups"]),
                    "num_negative_source_images": len(group.get("negative_source_groups") or []),
                    "reference_prompt_encode_ms": int(group.get("reference_prompt_encode_ms", 0)),
                    "grounding_forward_ms": int(group_profile["grounding_forward_ms"]),
                    "raw_candidates": int(group_profile["raw_candidate_count"]),
                    "post_nms_candidates": int(group_profile["kept_candidate_count"]),
                    "negative_reference_prompt_encode_ms": int(group.get("negative_reference_prompt_encode_ms", 0)),
                    "negative_grounding_forward_ms": int(negative_profile["negative_grounding_forward_ms"]),
                    "negative_raw_candidates": int(negative_profile["negative_raw_candidates"]),
                    "negative_post_nms_candidates": int(negative_profile["negative_post_nms_candidates"]),
                    "negative_filter_candidates": int(negative_profile["negative_filter_candidates"]),
                    "suppressed_by_negative_samples": suppressed_by_negative_count,
                    "candidate_loop_ms": group_candidate_loop_ms,
                    "max_visual_prompt_candidates": max_visual_prompt_candidates,
                }
            )

    kept_records = _nms_multi_similar_records(candidate_records, nms_iou, top_k)
    for record in kept_records:
        label = record["label"]
        polygons = mask_to_polygons(record["mask"], epsilon=polygon_simplify_epsilon)
        label["polygon_points"] = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(label["bnd_points"])
    matched_labels = [record["label"] for record in kept_records]

    result_image = None
    if return_result_image and kept_records:
        category_order: Dict[str, int] = {}
        for record in kept_records:
            category = record["label"]["category"]
            if category not in category_order:
                category_order[category] = len(category_order)
        detections_for_viz: List[Dict[str, Any]] = []
        for category, color_index in category_order.items():
            category_records = [record for record in kept_records if record["label"]["category"] == category]
            detections_for_viz.append(
                {
                    "class_name": category,
                    "original_class_name": category,
                    "masks": np.asarray([record["mask"] for record in category_records]),
                    "boxes": np.asarray([record["box_xyxy"] for record in category_records]),
                    "scores": np.asarray([record["score"] for record in category_records]),
                    "color": build_class_color(color_index),
                }
            )
        result_image = visualize_results(query_image, detections_for_viz)

    sample_contexts = prompt_state.get("sample_contexts") or []
    category_counts: Dict[str, int] = {}
    for label in matched_labels:
        category_counts[label["category"]] = category_counts.get(label["category"], 0) + 1

    processing_time_ms = _elapsed_ms(query_start_time)
    return {
        "model": MODEL_LABEL,
        "pic_id": pic_id,
        "success": True,
        "similar_mode": "multi_visual_prompt_url",
        "top_k": int(top_k),
        "top_k_scope": "per_category",
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "nms_iou": float(nms_iou),
        "num_samples": int(prompt_state.get("num_samples", len(sample_contexts))),
        "num_positive_samples": int(prompt_state.get("num_positive_samples", 0)),
        "num_negative_samples": int(prompt_state.get("num_negative_samples", 0)),
        "num_groups": len(grouped_prompts),
        "num_candidates": raw_candidate_count,
        "num_matches": len(matched_labels),
        "category_counts": category_counts,
        "pic_labels": matched_labels,
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
        "profile": {
            "set_query_ms": set_query_ms,
            "sample_cache_hit": bool(prompt_state.get("cache_hit", False)),
            "reference_feature_reuse_enabled": True,
            "reference_feature_extract_count": int(prompt_state.get("reference_source_count", 0)),
            "reference_sample_context_count": int(prompt_state.get("num_samples", len(sample_contexts))),
            "reference_prompt_encode_ms": int(prompt_state.get("reference_prompt_encode_ms", 0)),
            "negative_reference_prompt_encode_ms": int(prompt_state.get("negative_reference_prompt_encode_ms", 0)),
            "grounding_forward_ms": total_grounding_forward_ms,
            "candidate_loop_ms": total_candidate_loop_ms,
            "negative_grounding_forward_ms": total_negative_grounding_forward_ms,
            "raw_visual_prompt_candidates": raw_candidate_count,
            "post_nms_candidates": post_nms_candidate_count,
            "max_visual_prompt_candidates": max_visual_prompt_candidates,
            "negative_raw_visual_prompt_candidates": total_negative_raw_candidate_count,
            "negative_post_nms_candidates": total_negative_post_nms_candidate_count,
            "negative_filter_candidates": total_negative_filter_candidate_count,
            "suppressed_by_negative_samples": total_suppressed_by_negative_count,
            "negative_filter_iou": round(MULTI_NEGATIVE_FILTER_IOU, 6),
            "final_nms_iou": float(nms_iou),
            "native_multi_visual_prompt": True,
            "native_score_threshold": round(native_score_threshold, 6),
            "similarity_threshold_applied": False,
            "score_type": "sam_grounding_score",
            "groups": group_profiles,
        },
    }


def run_multi_similar_object_batch_pipeline(
    samples: List[Dict[str, Any]],
    query_images: List[Image.Image],
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    nms_iou: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
    query_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """多类别、多参考样本、多查询图的 batch visual prompt 检测入口。"""
    if not query_images:
        raise ValueError("At least one query image is required")

    start_time = time.perf_counter()
    normalized_pic_id = _normalize_pic_id(pic_id)
    query_results: List[Dict[str, Any]] = []
    with _model_inference_context():
        sample_contexts = _prepare_multi_similar_reference_contexts(samples, top_k)
        for index, query_image in enumerate(query_images):
            query_pic_id = _query_pic_id(normalized_pic_id, index, len(query_images))
            one_result = _run_multi_native_visual_prompt_query(
                sample_contexts,
                query_image,
                top_k,
                similarity_threshold,
                sam_threshold,
                nms_iou,
                polygon_simplify_epsilon,
                query_pic_id,
            )
            _attach_query_name(one_result, query_names, index)
            query_results.append(one_result)

    _maybe_empty_cuda_cache()

    total_processing_ms = _elapsed_ms(start_time)
    if len(query_results) == 1:
        return _single_query_batch_response(query_results, total_processing_ms)

    return {
        "model": MODEL_LABEL,
        "pic_id": normalized_pic_id,
        "success": True,
        "similar_mode": "multi_visual_prompt",
        "batch_size": len(query_results),
        "query_results": query_results,
        "num_samples": len(samples),
        "top_k": int(top_k),
        "top_k_scope": "per_category",
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "nms_iou": float(nms_iou),
        "total_num_matches": sum(int(item.get("num_matches", 0)) for item in query_results),
        "total_num_candidates": sum(int(item.get("num_candidates", 0)) for item in query_results),
        "sample_result_images": query_results[0].get("sample_result_images") if query_results else [],
        "reference_result_image": query_results[0].get("reference_result_image") if query_results else None,
        "created": int(time.time()),
        "processing_time_ms": total_processing_ms,
    }
