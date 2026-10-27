"""SAM3 TensorRT 运行时和业务推理 pipeline。"""

import asyncio
import contextlib
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import cupy as cp
import numpy as np
import tensorrt as trt
from PIL import Image
from tokenizers import Tokenizer

from .config import (
    CUDA_CLEANUP_AFTER_REQUEST,
    MAX_CONCURRENT_INFERENCES,
    MODEL_LABEL,
    NATIVE_ENGINE_DIR,
    SIMILAR_MODES,
    TRT_ENGINE_DIR,
    TRT_NMS_IOU,
    VISUAL_PROMPT_MAX_CANDIDATES,
)
from .image_utils import (
    bbox_iou_xywh,
    bbox_overlap_over_smaller_area_xywh,
    bbox_to_xywh,
    bbox_xywh_to_polygon_points,
    build_class_color,
    clip_bnd_points_to_image,
    clip_xyxy_to_image,
    mask_to_polygons,
    normalize_box_segmentation_inputs,
    visualize_results,
)
from .native_click_trt import NativeInteractiveBackend


LOGGER = logging.getLogger(__name__)
PredictionArrays = Dict[str, np.ndarray]
device = "cuda:0 (TensorRT)"


class TRTModule:
    """使用 CuPy 显存执行一个 TensorRT engine。"""

    def __init__(self, engine_path: Path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        with engine_path.open("rb") as engine_file, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT engine 加载失败: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"TensorRT execution context 创建失败: {engine_path}")
        # 使用默认 blocking stream 语义，确保 CuPy 输入拷贝先于 TRT 执行完成。
        self.stream = cp.cuda.Stream()
        self.lock = threading.Lock()
        self.io_info: Dict[str, Dict[str, Any]] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.io_info[name] = {
                "mode": self.engine.get_tensor_mode(name),
                "dtype": self.engine.get_tensor_dtype(name),
                "shape": tuple(self.engine.get_tensor_shape(name)),
            }

    @staticmethod
    def _cupy_dtype(dtype: Any) -> Any:
        mapping = {
            trt.float32: cp.float32,
            trt.float16: cp.float16,
            trt.int32: cp.int32,
            trt.int64: cp.int64,
            trt.bool: cp.bool_,
        }
        if dtype not in mapping:
            raise TypeError(f"不支持的 TensorRT dtype: {dtype}")
        return mapping[dtype]

    def get_tensor_shape(self, name: str) -> Tuple[int, ...]:
        if name not in self.io_info:
            raise KeyError(f"engine 中不存在 tensor: {name}")
        return self.io_info[name]["shape"]

    def __call__(self, **inputs: Any) -> Dict[str, cp.ndarray]:
        # URL 缓存预编码也会直接调用 engine，因此锁必须位于 context 本身。
        with self.lock:
            return self._execute(inputs)

    def _execute(self, inputs: Dict[str, Any]) -> Dict[str, cp.ndarray]:
        expected_inputs = {
            name for name, info in self.io_info.items() if info["mode"] == trt.TensorIOMode.INPUT
        }
        missing_inputs = expected_inputs.difference(inputs)
        if missing_inputs:
            raise ValueError(f"TensorRT 缺少输入: {', '.join(sorted(missing_inputs))}")

        # 输入和输出数组必须存活到异步执行完成，因此都保留在局部字典中。
        gpu_inputs: Dict[str, cp.ndarray] = {}
        gpu_outputs: Dict[str, cp.ndarray] = {}
        for name in expected_inputs:
            data = inputs[name]
            dtype = self._cupy_dtype(self.io_info[name]["dtype"])
            tensor = cp.asarray(data, dtype=dtype) if isinstance(data, np.ndarray) else cp.ascontiguousarray(data, dtype=dtype)
            gpu_inputs[name] = tensor
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"TensorRT 输入 shape 不在 profile 范围内: {name}={tuple(tensor.shape)}")
            self.context.set_tensor_address(name, int(tensor.data.ptr))

        for name, info in self.io_info.items():
            if info["mode"] != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"TensorRT 无法解析输出 shape: {name}={shape}")
            output = cp.empty(shape, dtype=self._cupy_dtype(info["dtype"]))
            gpu_outputs[name] = output
            self.context.set_tensor_address(name, int(output.data.ptr))

        if not self.context.execute_async_v3(stream_handle=self.stream.ptr):
            raise RuntimeError("TensorRT execute_async_v3 执行失败")
        self.stream.synchronize()
        return gpu_outputs


class Sam3TensorRTRuntime:
    """HTTP/TCP 共享图像编码器，分别使用检测和原生点击解码头。"""

    MAX_GEOMETRY_BOXES = 20
    MAX_DECODER_GEOMETRY_TOKENS = 28

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir.resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"TensorRT engine 目录不存在: {self.model_dir}")
        compute_capability = cp.cuda.Device(0).compute_capability
        suffix = f"_sm{compute_capability}_trt{trt.__version__}"
        # One dual-neck encoder replaces both separate image encoders in GPU memory.
        self.vision_encoder = TRTModule(self._find_engine("shared-image-encoder", suffix))
        self.text_encoder = TRTModule(self._find_engine("text-encoder", suffix))
        self.geometry_encoder = TRTModule(self._find_engine("geometry-encoder", suffix))
        self.decoder = TRTModule(self._find_engine("decoder", suffix))
        self.interactive = NativeInteractiveBackend(NATIVE_ENGINE_DIR, encoder=self.vision_encoder)

        input_shape = self.vision_encoder.get_tensor_shape("images")
        self.target_height = int(input_shape[2])
        self.target_width = int(input_shape[3])
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"TensorRT tokenizer 不存在: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)
        self.model = self

    def _find_engine(self, name: str, suffix: str) -> Path:
        specific_path = self.model_dir / f"{name}{suffix}.engine"
        generic_path = self.model_dir / f"{name}.engine"
        engine_path = specific_path if specific_path.is_file() else generic_path
        if not engine_path.is_file():
            raise FileNotFoundError(f"缺少 TensorRT engine: {generic_path}")
        return engine_path

    def encode_image(self, image: Image.Image) -> Dict[str, cp.ndarray]:
        """按导出模型的 [-1, 1] RGB 规则预处理图片并提取 FPN 特征。"""
        rgb = np.asarray(image.convert("RGB").resize((self.target_width, self.target_height), Image.Resampling.BILINEAR))
        pixel_values = (rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        output = self.vision_encoder(images=pixel_values)
        return {name: output[name] for name in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2")}

    def encode_text(self, text: Optional[str]) -> Tuple[cp.ndarray, cp.ndarray]:
        if text:
            encoding = self.tokenizer.encode(text)
            input_ids = np.asarray([encoding.ids], dtype=np.int64)
            attention_mask = np.asarray([encoding.attention_mask], dtype=np.int64)
        else:
            # 纯框 prompt 仍需要一个有效的占位 token，与 TRT demo 保持一致。
            input_ids = np.full((1, 32), 49407, dtype=np.int64)
            attention_mask = np.zeros((1, 32), dtype=np.int64)
            attention_mask[0, 0] = 1
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output["text_features"], output["text_mask"]

    def encode_geometry(
        self,
        image_features: Dict[str, cp.ndarray],
        boxes_xywh: Sequence[Sequence[float]],
        labels: Sequence[int],
        image_size: Tuple[int, int],
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        if not boxes_xywh or len(boxes_xywh) != len(labels):
            raise ValueError("几何 prompt 的 boxes 和 labels 必须非空且数量一致")
        if len(boxes_xywh) > self.MAX_GEOMETRY_BOXES:
            raise ValueError(f"单次 geometry engine 最多支持 {self.MAX_GEOMETRY_BOXES} 个框")
        image_width, image_height = image_size
        normalized_boxes = []
        for raw_box in boxes_xywh:
            x, y, width, height = [float(value) for value in raw_box]
            normalized_boxes.append(
                [(x + width / 2) / image_width, (y + height / 2) / image_height, width / image_width, height / image_height]
            )
        output = self.geometry_encoder(
            input_boxes=np.asarray([normalized_boxes], dtype=np.float32),
            input_boxes_labels=np.asarray([labels], dtype=np.int64),
            fpn_feat_2=image_features["fpn_feat_2"],
            fpn_pos_2=image_features["fpn_pos_2"],
        )
        return output["geometry_features"], output["geometry_mask"]

    def decode(
        self,
        image_features: Dict[str, cp.ndarray],
        image_size: Tuple[int, int],
        *,
        text: Optional[str],
        geometry: Optional[Tuple[cp.ndarray, cp.ndarray]] = None,
        confidence_threshold: float,
        nms_iou: float = TRT_NMS_IOU,
        max_candidates: int = 300,
    ) -> PredictionArrays:
        prompt_features, prompt_mask = self.encode_text(text)
        if geometry is not None:
            geometry_features, geometry_mask = geometry
            prompt_features = cp.concatenate([prompt_features, geometry_features], axis=1)
            prompt_mask = cp.concatenate([prompt_mask, geometry_mask], axis=1)
        if prompt_features.shape[1] > 60:
            raise ValueError(f"decoder prompt token 数量超过 engine 上限 60: {prompt_features.shape[1]}")

        output = self.decoder(
            fpn_feat_0=image_features["fpn_feat_0"],
            fpn_feat_1=image_features["fpn_feat_1"],
            fpn_feat_2=image_features["fpn_feat_2"],
            fpn_pos_2=image_features["fpn_pos_2"],
            prompt_features=prompt_features,
            prompt_mask=prompt_mask,
        )
        logits = output["pred_logits"][0].get().astype(np.float32)
        presence = float(np.asarray(output["presence_logits"][0, 0].get()))
        boxes = output["pred_boxes"][0].get().astype(np.float32)
        scores = _sigmoid(logits) * float(_sigmoid(np.asarray(presence)))
        selected = np.flatnonzero(scores >= float(confidence_threshold))
        if selected.size == 0:
            return _empty_prediction_arrays()

        image_width, image_height = image_size
        boxes[:, [0, 2]] *= image_width
        boxes[:, [1, 3]] *= image_height
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)
        selected = selected[_nms_indices(boxes[selected], scores[selected], nms_iou, max_candidates)]
        masks = output["pred_masks"][0, cp.asarray(selected)].get()
        resized_masks = []
        for mask in masks:
            mask_2d = np.squeeze(mask)
            resized = cv2.resize(mask_2d, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(resized > 0)
        return {
            "masks": np.asarray(resized_masks, dtype=bool),
            "boxes": boxes[selected],
            "scores": scores[selected],
            "classes": np.zeros(selected.size, dtype=np.int64),
        }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, limit: int) -> np.ndarray:
    """使用 NumPy 做稳定 NMS，避免为后处理重新引入 torchvision。"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    kept: List[int] = []
    while order.size and len(kept) < max(1, int(limit)):
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        inter_x1 = np.maximum(x1[current], x1[remaining])
        inter_y1 = np.maximum(y1[current], y1[remaining])
        inter_x2 = np.minimum(x2[current], x2[remaining])
        inter_y2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= float(iou_threshold)]
    return np.asarray(kept, dtype=np.int64)


def _empty_prediction_arrays() -> PredictionArrays:
    return {
        "masks": np.zeros((0, 0, 0), dtype=bool),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


print(f"正在加载 SAM3 TensorRT engines: {TRT_ENGINE_DIR}")
predictor = Sam3TensorRTRuntime(TRT_ENGINE_DIR)
print(f"SAM3 TensorRT engines 加载完成，输入尺寸: {predictor.target_width}x{predictor.target_height}")

MODEL_LOCK = threading.Lock()
INFERENCE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
INFERENCE_STATE_LOCK = threading.Lock()
ACTIVE_INFERENCE_COUNT = 0
LAST_INFERENCE_FINISHED_AT = 0.0
IDLE_MODEL_UNLOADED = False


@contextlib.contextmanager
def _model_inference_context() -> Iterator[None]:
    """TensorRT context 不是线程安全对象，所有共享 engine 访问必须串行。"""
    with MODEL_LOCK:
        yield


async def run_inference_in_thread(func: Any, *args: Any) -> Any:
    global ACTIVE_INFERENCE_COUNT, LAST_INFERENCE_FINISHED_AT
    with INFERENCE_STATE_LOCK:
        ACTIVE_INFERENCE_COUNT += 1
    try:
        return await asyncio.to_thread(func, *args)
    finally:
        with INFERENCE_STATE_LOCK:
            ACTIVE_INFERENCE_COUNT -= 1
            LAST_INFERENCE_FINISHED_AT = time.monotonic()
        if CUDA_CLEANUP_AFTER_REQUEST:
            cp.get_default_memory_pool().free_all_blocks()


def get_cuda_memory_stats() -> Dict[str, int]:
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
        "memory_pool_used_bytes": int(cp.get_default_memory_pool().used_bytes()),
    }


def _elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _normalize_pic_id(pic_id: Optional[str]) -> str:
    return (pic_id or "").strip() or uuid.uuid4().hex[:16]


def _prepare_detection_classes(prompt: str) -> Tuple[List[Dict[str, Any]], str, bool]:
    # Box-only TCP inference must not initialize or install Argos translation resources.
    from .translation import split_prompt_classes, translate_to_english

    original_classes = split_prompt_classes(prompt)
    if not original_classes:
        raise ValueError("Prompt is empty after parsing. Use '#' to separate classes.")
    classes = []
    translated_classes = []
    for index, original in enumerate(original_classes):
        translated = (translate_to_english(original) or original).strip() or original
        translated_classes.append(translated)
        classes.append(
            {
                "class_name": translated,
                "original_class_name": original,
                "color": build_class_color(index),
            }
        )
    translated_prompt = "#".join(translated_classes)
    return classes, translated_prompt, translated_prompt != "#".join(original_classes)


def _visualize(image: Image.Image, records: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not records:
        return None
    grouped: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        label = record["label"]
        category = str(label["category"])
        group = grouped.setdefault(
            category,
            {
                "class_name": label.get("translated_category") or category,
                "original_class_name": category,
                "masks": [],
                "boxes": [],
                "scores": [],
                "color": build_class_color(len(grouped)),
            },
        )
        group["masks"].append(record["mask"])
        group["boxes"].append(record["box"])
        group["scores"].append(label.get("score", 0.0))
    detections = []
    for group in grouped.values():
        group["masks"] = np.asarray(group["masks"])
        group["boxes"] = np.asarray(group["boxes"])
        group["scores"] = np.asarray(group["scores"])
        detections.append(group)
    return visualize_results(image, detections)


def _records_from_arrays(
    arrays: PredictionArrays,
    image: Image.Image,
    category: str,
    translated_category: Optional[str],
    polygon_simplify_epsilon: float,
) -> List[Dict[str, Any]]:
    records = []
    for mask, box, score in zip(arrays["masks"], arrays["boxes"], arrays["scores"]):
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        mask_area = int(np.count_nonzero(mask_u8))
        if mask_area == 0:
            continue
        clipped_box = clip_xyxy_to_image([float(value) for value in box], image.width, image.height)
        bnd_points = bbox_to_xywh(np.asarray(clipped_box))
        polygons = mask_to_polygons(mask_u8, epsilon=polygon_simplify_epsilon)
        label = {
            "category": category,
            "translated_category": translated_category,
            "score": round(float(score), 6),
            "bnd_points": bnd_points,
            "polygon_points": polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(bnd_points),
            "mask_area": mask_area,
        }
        records.append({"label": label, "mask": mask_u8.astype(np.float32), "box": clipped_box})
    return records


def run_detection_pipeline(
    image: Image.Image,
    prompt: str,
    confidence_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
    return_result_image: bool = False,
) -> Dict[str, Any]:
    image = image.convert("RGB")
    start_time = time.perf_counter()
    classes, translated_prompt, was_translated = _prepare_detection_classes(prompt)
    records: List[Dict[str, Any]] = []
    with _model_inference_context():
        image_features = predictor.encode_image(image)
        for class_info in classes:
            arrays = predictor.decode(
                image_features,
                (image.width, image.height),
                text=class_info["class_name"],
                confidence_threshold=confidence_threshold,
            )
            records.extend(
                _records_from_arrays(
                    arrays,
                    image,
                    class_info["original_class_name"],
                    class_info["class_name"],
                    polygon_simplify_epsilon,
                )
            )
    labels = [record["label"] for record in records]
    detection_details: Dict[str, int] = {}
    for label in labels:
        detection_details[label["category"]] = detection_details.get(label["category"], 0) + 1
    return {
        "model": MODEL_LABEL,
        "pic_id": _normalize_pic_id(pic_id),
        "success": True,
        "pic_labels": labels,
        "num_detections": len(labels),
        "classes_detected": len(detection_details),
        "detection_details": detection_details,
        "prompt": prompt,
        "translated_prompt": translated_prompt if was_translated else None,
        "was_translated": was_translated,
        "confidence_threshold": confidence_threshold,
        # 批量标注只消费结构化标签，关闭结果图可避免每张图都执行 Matplotlib 绘图和磁盘写入。
        "result_image": _visualize(image, records) if return_result_image else None,
        "created": int(time.time()),
        "processing_time_ms": _elapsed_ms(start_time),
    }


def _merge_arrays(items: Sequence[PredictionArrays]) -> PredictionArrays:
    non_empty = [item for item in items if item["scores"].size]
    if not non_empty:
        return _empty_prediction_arrays()
    return {
        key: np.concatenate([item[key] for item in non_empty], axis=0)
        for key in ("masks", "boxes", "scores", "classes")
    }


def _match_requested_boxes(requested_boxes: Sequence[Sequence[float]], predicted_boxes: np.ndarray) -> List[Optional[int]]:
    pairs = []
    for request_index, request_box in enumerate(requested_boxes):
        for prediction_index, prediction_box in enumerate(predicted_boxes):
            predicted_xywh = bbox_to_xywh(prediction_box)
            pairs.append((bbox_iou_xywh(list(request_box), predicted_xywh), request_index, prediction_index))
    matches: List[Optional[int]] = [None] * len(requested_boxes)
    used = set()
    for score, request_index, prediction_index in sorted(pairs, reverse=True):
        if score > 0 and matches[request_index] is None and prediction_index not in used:
            matches[request_index] = prediction_index
            used.add(prediction_index)
    return matches


def run_box_segmentation_pipeline(
    image: Image.Image,
    bnd_points: Any,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
) -> Dict[str, Any]:
    image = image.convert("RGB")
    start_time = time.perf_counter()
    raw_boxes = normalize_box_segmentation_inputs(bnd_points)
    clipped_boxes = [clip_bnd_points_to_image(box, image.width, image.height) for box in raw_boxes]
    arrays_list = []
    with _model_inference_context():
        image_features = predictor.encode_image(image)
        for offset in range(0, len(clipped_boxes), predictor.MAX_GEOMETRY_BOXES):
            box_chunk = clipped_boxes[offset : offset + predictor.MAX_GEOMETRY_BOXES]
            geometry = predictor.encode_geometry(
                image_features,
                box_chunk,
                [1] * len(box_chunk),
                (image.width, image.height),
            )
            arrays_list.append(
                predictor.decode(
                    image_features,
                    (image.width, image.height),
                    text=None,
                    geometry=geometry,
                    confidence_threshold=0.0,
                )
            )
    arrays = _merge_arrays(arrays_list)
    matches = _match_requested_boxes(clipped_boxes, arrays["boxes"])
    segmentations = []
    for index, requested_box in enumerate(clipped_boxes):
        prediction_index = matches[index]
        if prediction_index is None:
            result = {
                "input_bnd_points": requested_box,
                "bnd_points": requested_box,
                "polygon_points": bbox_xywh_to_polygon_points(requested_box),
                "score": 0.0,
                "mask_area": 0,
                "used_fallback": True,
                "index": index,
            }
        else:
            mask = arrays["masks"][prediction_index]
            predicted_box = bbox_to_xywh(arrays["boxes"][prediction_index])
            polygons = mask_to_polygons(mask, epsilon=polygon_simplify_epsilon)
            result = {
                "input_bnd_points": requested_box,
                "bnd_points": predicted_box,
                "polygon_points": polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(predicted_box),
                "score": round(float(arrays["scores"][prediction_index]), 6),
                "mask_area": int(np.count_nonzero(mask)),
                "used_fallback": not bool(polygons),
                "index": index,
            }
        segmentations.append(result)
    response: Dict[str, Any] = {
        "model": MODEL_LABEL,
        "pic_id": _normalize_pic_id(pic_id),
        "success": True,
        "segmentations": segmentations,
        "num_segmentations": len(segmentations),
        "created": int(time.time()),
        "processing_time_ms": _elapsed_ms(start_time),
    }
    if len(segmentations) == 1:
        response.update({key: segmentations[0][key] for key in ("bnd_points", "polygon_points", "score", "mask_area", "used_fallback")})
    return response


def _normalize_sample_type(sample: Dict[str, Any]) -> str:
    raw = str(sample.get("sample_type", sample.get("role", "")) or "").strip().lower()
    return "negative" if raw in {"0", "negative", "neg", "false"} or sample.get("is_negative") is True else "positive"


def _normalize_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .translation import _normalize_prompt_label

    normalized = []
    for index, sample in enumerate(samples):
        category = _normalize_prompt_label(str(sample.get("category", "") or ""))
        image = sample.get("reference_image")
        box = sample.get("reference_bnd_points")
        if not category:
            raise ValueError(f"samples[{index}].category is required")
        if not isinstance(image, Image.Image):
            raise ValueError(f"samples[{index}].reference_image is required")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"samples[{index}].reference_bnd_points must be [x, y, w, h]")
        normalized.append(
            {
                **sample,
                "category": category,
                "reference_image": image.convert("RGB"),
                "reference_bnd_points": clip_bnd_points_to_image(box, image.width, image.height),
                "sample_type": _normalize_sample_type(sample),
                "sample_id": str(sample.get("sample_id") or f"sample_{index + 1}"),
                "source_image_id": str(sample.get("source_image_id") or sample.get("sample_id") or f"image_{index + 1}"),
            }
        )
    return normalized


def _prompt_specs(prompt_text: Optional[str], prompt_category_map: Optional[Dict[str, str]]) -> List[Dict[str, Any]]:
    from .translation import _normalize_prompt_label, split_prompt_classes, translate_to_english

    specs = []
    if prompt_category_map:
        prompt_items = prompt_category_map.items()
    else:
        prompt_items = ((prompt, prompt) for prompt in split_prompt_classes(prompt_text or ""))
    for category, raw_prompt in prompt_items:
        normalized_category = _normalize_prompt_label(str(category or ""))
        for prompt in split_prompt_classes(str(raw_prompt or "")):
            translated = (translate_to_english(prompt) or prompt).strip() or prompt
            if normalized_category and translated:
                specs.append(
                    {
                        "category": normalized_category,
                        "prompt": prompt,
                        "text": translated,
                        "translated_prompt": translated if translated != prompt else None,
                    }
                )
    return specs


def _categories_match(sample_category: str, prompt_category: str) -> bool:
    """同时比较原始类别和英文翻译，兼容“人/person”这类映射。"""
    from .translation import _normalize_prompt_label, translate_to_english

    sample_normalized = _normalize_prompt_label(sample_category).lower()
    prompt_normalized = _normalize_prompt_label(prompt_category).lower()
    sample_variants = {
        sample_normalized,
        _normalize_prompt_label(translate_to_english(sample_normalized) or "").lower(),
    }
    prompt_variants = {
        prompt_normalized,
        _normalize_prompt_label(translate_to_english(prompt_normalized) or "").lower(),
    }
    return bool(sample_variants.intersection(prompt_variants).difference({""}))


def _pack_geometry_outputs(outputs: Sequence[Tuple[cp.ndarray, cp.ndarray]]) -> List[Tuple[cp.ndarray, cp.ndarray]]:
    """把不同参考图的几何 token 打包到 decoder 的 60-token profile 内。"""
    packed: List[Tuple[cp.ndarray, cp.ndarray]] = []
    current_features: List[cp.ndarray] = []
    current_masks: List[cp.ndarray] = []
    current_length = 0
    for features, mask in outputs:
        token_length = int(features.shape[1])
        if current_features and current_length + token_length > predictor.MAX_DECODER_GEOMETRY_TOKENS:
            packed.append((cp.concatenate(current_features, axis=1), cp.concatenate(current_masks, axis=1)))
            current_features, current_masks, current_length = [], [], 0
        current_features.append(features)
        current_masks.append(mask)
        current_length += token_length
    if current_features:
        packed.append((cp.concatenate(current_features, axis=1), cp.concatenate(current_masks, axis=1)))
    return packed


def prepare_multi_visual_prompt_state(
    samples: List[Dict[str, Any]],
    top_k: int,
    prompt_text: Optional[str] = None,
    prompt_category_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """预编码样例图 visual prompt，URL/视频任务可以跨查询图复用。"""
    from .translation import _normalize_prompt_label, translate_to_english

    normalized_samples = _normalize_samples(samples)
    specs = _prompt_specs(prompt_text, prompt_category_map)
    grouped: Dict[str, Dict[str, Any]] = {}
    for sample in normalized_samples:
        group = grouped.setdefault(
            sample["category"],
            {"category": sample["category"], "samples": [], "text_specs": []},
        )
        group["samples"].append(sample)
        sample_prompt = _normalize_prompt_label(str(sample.get("prompt", "") or ""))
        if sample_prompt and not group["text_specs"]:
            translated = (translate_to_english(sample_prompt) or sample_prompt).strip()
            group["text_specs"].append(
                {"prompt": sample_prompt, "text": translated, "translated_prompt": translated if translated != sample_prompt else None}
            )

    for spec_index, spec in enumerate(specs):
        matching_category = next(
            (category for category in grouped if _categories_match(category, spec["category"])),
            None,
        )
        group_key = matching_category or f"__text_prompt_{spec_index}"
        group = grouped.setdefault(
            group_key,
            {"category": spec["category"], "samples": [], "text_specs": []},
        )
        if spec not in group["text_specs"]:
            group["text_specs"].append(spec)

    if not grouped:
        raise ValueError("At least one positive sample or prompt is required")
    if normalized_samples and not any(sample["sample_type"] == "positive" for sample in normalized_samples) and not specs:
        raise ValueError("At least one positive sample or prompt is required")

    encoded_groups = []
    image_feature_cache: Dict[str, Dict[str, cp.ndarray]] = {}
    for group in grouped.values():
        if group["samples"] and not any(sample["sample_type"] == "positive" for sample in group["samples"]) and not group["text_specs"]:
            continue
        geometry_outputs = []
        samples_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for sample in group["samples"]:
            samples_by_source.setdefault(sample["source_image_id"], []).append(sample)
        for source_id, source_samples in samples_by_source.items():
            image = source_samples[0]["reference_image"]
            cache_key = source_id
            features = image_feature_cache.get(cache_key)
            if features is None:
                features = predictor.encode_image(image)
                image_feature_cache[cache_key] = features
            for offset in range(0, len(source_samples), predictor.MAX_GEOMETRY_BOXES):
                chunk = source_samples[offset : offset + predictor.MAX_GEOMETRY_BOXES]
                geometry_outputs.append(
                    predictor.encode_geometry(
                        features,
                        [sample["reference_bnd_points"] for sample in chunk],
                        [0 if sample["sample_type"] == "negative" else 1 for sample in chunk],
                        (image.width, image.height),
                    )
                )
        geometry_chunks = _pack_geometry_outputs(geometry_outputs)
        # ponytail: decoder 最多 60 token；超量样例按 chunk 推理并在 CPU 侧 NMS 合并。
        text_specs = group["text_specs"] or [{"text": None, "prompt": None, "translated_prompt": None}]
        for text_spec in text_specs:
            encoded_groups.append(
                {
                    "category": group["category"],
                    "text": text_spec.get("text"),
                    "prompt": text_spec.get("prompt"),
                    "translated_prompt": text_spec.get("translated_prompt"),
                    "geometry_chunks": geometry_chunks,
                    "sample_count": len(group["samples"]),
                }
            )
    return {
        "groups": encoded_groups,
        "num_samples": len(normalized_samples),
        "reference_source_count": len(image_feature_cache),
        "top_k": int(top_k),
        "prompt": prompt_text,
        "sample_result_images": [],
        "reference_result_image": None,
    }


def _nms_records(records: Sequence[Dict[str, Any]], iou_threshold: float, top_k: int) -> List[Dict[str, Any]]:
    kept = []
    for record in sorted(records, key=lambda item: float(item["label"]["score"]), reverse=True):
        box = record["label"]["bnd_points"]
        suppressed = False
        for existing in kept:
            existing_box = existing["label"]["bnd_points"]
            if (
                bbox_iou_xywh(existing_box, box) > iou_threshold
                or bbox_overlap_over_smaller_area_xywh(existing_box, box) >= max(0.9, iou_threshold)
            ):
                suppressed = True
                break
        if not suppressed:
            kept.append(record)
            if len(kept) >= top_k:
                break
    return kept


def run_multi_visual_prompt_query_with_state(
    prompt_state: Dict[str, Any],
    query_image: Image.Image,
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    nms_iou: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
    return_result_image: bool = True,
) -> Dict[str, Any]:
    query_image = query_image.convert("RGB")
    start_time = time.perf_counter()
    all_records = []
    raw_candidate_count = 0
    with _model_inference_context():
        query_features = predictor.encode_image(query_image)
        for group in prompt_state["groups"]:
            geometries = group["geometry_chunks"] or [None]
            group_records = []
            for geometry in geometries:
                arrays = predictor.decode(
                    query_features,
                    (query_image.width, query_image.height),
                    text=group.get("text"),
                    geometry=geometry,
                    confidence_threshold=sam_threshold,
                    max_candidates=max(top_k * 2, VISUAL_PROMPT_MAX_CANDIDATES),
                )
                raw_candidate_count += int(arrays["scores"].size)
                group_records.extend(
                    _records_from_arrays(
                        arrays,
                        query_image,
                        group["category"],
                        group.get("text"),
                        polygon_simplify_epsilon,
                    )
                )
            all_records.extend(_nms_records(group_records, nms_iou, top_k))

    # 多个文字条件或 geometry chunk 可能命中同一目标，最终仍按业务类别去重并限制 top_k。
    records_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for record in all_records:
        records_by_category.setdefault(record["label"]["category"], []).append(record)
    all_records = []
    for category_records in records_by_category.values():
        all_records.extend(_nms_records(category_records, nms_iou, top_k))
    all_records.sort(key=lambda item: float(item["label"]["score"]), reverse=True)
    for record in all_records:
        score = record["label"]["score"]
        record["label"].update(
            {
                "sam_score": score,
                "similarity_score": score,
                "combined_score": score,
                "coarse_similarity": score,
            }
        )
    labels = [record["label"] for record in all_records]
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
        "num_samples": int(prompt_state.get("num_samples", 0)),
        "num_candidates": raw_candidate_count,
        "num_matches": len(labels),
        "pic_labels": labels,
        "sample_result_images": prompt_state.get("sample_result_images", []),
        "reference_result_image": prompt_state.get("reference_result_image"),
        "result_image": _visualize(query_image, all_records) if return_result_image else None,
        "created": int(time.time()),
        "processing_time_ms": _elapsed_ms(start_time),
        "profile": {
            "backend": "tensorrt",
            "native_multi_visual_prompt": True,
            "reference_feature_reuse_enabled": True,
            "reference_feature_extract_count": int(prompt_state.get("reference_source_count", 0)),
            "raw_visual_prompt_candidates": raw_candidate_count,
            "native_score_threshold": round(float(sam_threshold), 6),
            "similarity_threshold_applied": False,
            "score_type": "sam_grounding_score",
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
    prompt_text: Optional[str] = None,
    prompt_category_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if not query_images:
        raise ValueError("At least one query image is required")
    start_time = time.perf_counter()
    normalized_pic_id = _normalize_pic_id(pic_id)
    prompt_state = prepare_multi_visual_prompt_state(samples, top_k, prompt_text, prompt_category_map)
    query_results = []
    for index, query_image in enumerate(query_images):
        query_pic_id = normalized_pic_id if len(query_images) == 1 else f"{normalized_pic_id}_{index + 1}"
        result = run_multi_visual_prompt_query_with_state(
            prompt_state,
            query_image,
            top_k,
            similarity_threshold,
            sam_threshold,
            nms_iou,
            polygon_simplify_epsilon,
            query_pic_id,
        )
        result["similar_mode"] = "multi_visual_prompt"
        if query_names and index < len(query_names):
            result["query_name"] = query_names[index]
        query_results.append(result)
    total_ms = _elapsed_ms(start_time)
    if len(query_results) == 1:
        response = dict(query_results[0])
        response.update({"batch_size": 1, "query_results": query_results, "processing_time_ms": total_ms})
        return response
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
        "total_num_matches": sum(item["num_matches"] for item in query_results),
        "total_num_candidates": sum(item["num_candidates"] for item in query_results),
        "sample_result_images": prompt_state.get("sample_result_images", []),
        "reference_result_image": prompt_state.get("reference_result_image"),
        "created": int(time.time()),
        "processing_time_ms": total_ms,
    }


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
    from .translation import _normalize_prompt_label

    if similar_mode not in SIMILAR_MODES:
        raise ValueError(f"similar_mode must be one of: {', '.join(sorted(SIMILAR_MODES))}")
    if reference_bnd_points is None:
        raise ValueError("reference_bnd_points is required")
    reference_box = clip_bnd_points_to_image(reference_bnd_points, reference_image.width, reference_image.height)
    target_images = [reference_image] if similar_mode == "same_image_prompt" else query_images
    if not target_images:
        raise ValueError("At least one query image is required")
    category = _normalize_prompt_label(prompt or "") or "similar_object"
    sample = {
        "sample_id": "single_reference",
        "source_image_id": "single_reference",
        "sample_type": "positive",
        "category": category,
        "reference_image": reference_image,
        "reference_bnd_points": reference_box,
        "prompt": prompt,
    }
    result = run_multi_similar_object_batch_pipeline(
        [sample],
        target_images,
        top_k,
        similarity_threshold,
        sam_threshold,
        0.75,
        polygon_simplify_epsilon,
        pic_id,
        query_names,
    )
    result["similar_mode"] = similar_mode
    result["reference_bnd_points"] = reference_box
    result["reference_box_auto_generated"] = False
    for query_result in result.get("query_results", []):
        query_result["similar_mode"] = similar_mode
        query_result["reference_bnd_points"] = reference_box
        query_result["reference_box_auto_generated"] = False
    return result


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
    result = run_similar_object_batch_pipeline(
        reference_image,
        [query_image],
        reference_bnd_points,
        top_k,
        similarity_threshold,
        sam_threshold,
        polygon_simplify_epsilon,
        pic_id,
        None,
        similar_mode,
        prompt,
    )
    return result


def _self_check() -> None:
    """最小后处理自检：验证 NMS 会保留最高分框并去除重框。"""
    boxes = np.asarray([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    assert _nms_indices(boxes, scores, 0.5, 10).tolist() == [0, 2]


if __name__ == "__main__":
    _self_check()
    print("SAM3 TensorRT 后处理自检通过")
