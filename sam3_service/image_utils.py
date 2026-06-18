"""Image decoding, geometry and visualization helpers for SAM3 results."""

import base64
import binascii
import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import MAX_IMAGE_BYTES, RESULT_DIR, UPLOAD_DIR


def setup_chinese_font() -> str:
    """Configure matplotlib for Chinese labels."""
    chinese_fonts = [
        "GB_KT_GB18030",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "GB_SS_GB18030",
        "GB_HT_GB18030",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
            print(f"Using font: {font}")
            break

    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"][0]


CURRENT_FONT = setup_chinese_font()

def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        # NumPy has no native bfloat16 dtype in many runtime stacks.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.numpy()
    return np.asarray(value)


def save_upload_image(image: Image.Image, original_name: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", original_name or "input.jpg")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"upload_{timestamp}_{safe_name}"
    image.save(UPLOAD_DIR / filename)
    return filename


def decode_base64_image(image_base64: str) -> Image.Image:
    if not image_base64:
        raise ValueError("image_base64 is required")

    raw_data = image_base64.strip()
    if raw_data.startswith("data:image") and "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data") from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image payload too large. Max bytes: {MAX_IMAGE_BYTES}")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Decoded base64 is not a valid image") from exc

    return image


def build_class_color(index: int) -> np.ndarray:
    hue = (index * 137.5) % 360
    if hue < 60:
        rgb = (255, int(hue * 4.25), 0)
    elif hue < 120:
        rgb = (int(255 - (hue - 60) * 4.25), 255, 0)
    elif hue < 180:
        rgb = (0, 255, int((hue - 120) * 4.25))
    elif hue < 240:
        rgb = (0, int(255 - (hue - 180) * 4.25), 255)
    elif hue < 300:
        rgb = (int((hue - 240) * 4.25), 0, 255)
    else:
        rgb = (255, 0, int(255 - (hue - 300) * 4.25))
    return np.array(rgb)


def mask_to_polygons(mask: np.ndarray, epsilon: float = 2.0, min_area: float = 10.0) -> List[Dict[str, Any]]:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim > 2:
        mask_arr = np.squeeze(mask_arr)

    if mask_arr.ndim != 2:
        return []

    binary = (mask_arr > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return []

    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[Dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)

        if contour.shape[0] < 3:
            continue

        points = [[round(float(p[0][0]), 3), round(float(p[0][1]), 3)] for p in contour]
        polygons.append(
            {
                "area": float(area),
                "points": points,
            }
        )

    polygons.sort(key=lambda item: item["area"], reverse=True)

    return polygons


def bbox_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return [
        round(x1, 3),
        round(y1, 3),
        round(width, 3),
        round(height, 3),
    ]


def bbox_xywh_to_xyxy(bnd_points: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in bnd_points]
    return [
        round(x, 3),
        round(y, 3),
        round(x + w, 3),
        round(y + h, 3),
    ]


def clip_xyxy_to_image(box_xyxy: List[float], image_width: int, image_height: int) -> List[float]:
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must have exactly 4 values: [x1, y1, x2, y2]")
    x1 = _safe_float_from_any(box_xyxy[0], "box_xyxy[0]")
    y1 = _safe_float_from_any(box_xyxy[1], "box_xyxy[1]")
    x2 = _safe_float_from_any(box_xyxy[2], "box_xyxy[2]")
    y2 = _safe_float_from_any(box_xyxy[3], "box_xyxy[3]")
    x1 = max(0.0, min(x1, image_width * 1.0))
    y1 = max(0.0, min(y1, image_height * 1.0))
    x2 = max(0.0, min(x2, image_width * 1.0))
    y2 = max(0.0, min(y2, image_height * 1.0))
    if x2 <= x1:
        x2 = min(image_width * 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(image_height * 1.0, y1 + 1.0)
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def bbox_iou_xywh(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def bbox_xywh_to_polygon_points(bnd_points: List[float]) -> List[List[float]]:
    x, y, w, h = bnd_points
    return [
        [round(x, 3), round(y, 3)],
        [round(x + w, 3), round(y, 3)],
        [round(x + w, 3), round(y + h, 3)],
        [round(x, 3), round(y + h, 3)],
    ]


def clip_bnd_points_to_image(bnd_points: List[float], image_width: int, image_height: int) -> List[float]:
    if len(bnd_points) != 4:
        raise ValueError("bnd_points must have exactly 4 values: [x, y, w, h]")

    x = _safe_float_from_any(bnd_points[0], "bnd_points[0]")
    y = _safe_float_from_any(bnd_points[1], "bnd_points[1]")
    w = _safe_float_from_any(bnd_points[2], "bnd_points[2]")
    h = _safe_float_from_any(bnd_points[3], "bnd_points[3]")

    if w <= 0 or h <= 0:
        raise ValueError("bnd_points width and height must be > 0")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image size")

    x = max(0.0, min(x, image_width - 1.0))
    y = max(0.0, min(y, image_height - 1.0))
    w = max(1.0, min(w, image_width - x))
    h = max(1.0, min(h, image_height - y))
    return [round(x, 3), round(y, 3), round(w, 3), round(h, 3)]


def parse_optional_bnd_points_text(raw_value: Optional[str]) -> Optional[List[float]]:
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None

    parts = [one.strip() for one in re.split(r"[,\s]+", normalized) if one.strip()]
    if len(parts) != 4:
        raise ValueError("reference_bnd_points must be 4 numbers: x,y,w,h")

    try:
        return [float(one) for one in parts]
    except ValueError as exc:
        raise ValueError("reference_bnd_points must contain numeric values") from exc


def normalize_box_segmentation_inputs(raw_bnd_points: Any) -> List[List[float]]:
    """兼容单框 [x,y,w,h] 和多框 [[x,y,w,h], ...] 两种输入形状。"""
    if not isinstance(raw_bnd_points, list) or len(raw_bnd_points) == 0:
        raise ValueError(
            "bnd_points payload is required. Use bnd_points=[x,y,w,h] or bnd_points_list=[[x,y,w,h], ...]"
        )

    first_item = raw_bnd_points[0]
    if isinstance(first_item, (list, tuple, np.ndarray)):
        return [list(one_box) for one_box in raw_bnd_points]
    return [list(raw_bnd_points)]


def _safe_float_from_any(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def visualize_results(image: Image.Image, all_detections: List[Dict[str, Any]]) -> str:
    img_array = np.array(image)
    overlay = img_array.copy().astype(np.float32)

    for detection in all_detections:
        masks = to_numpy(detection["masks"])
        if masks.ndim == 2:
            masks = masks[None, ...]

        class_color = detection["color"].astype(np.float32)
        for mask in masks:
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            mask_bool = mask > 0.5
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + class_color * 0.5

    overlay = overlay.astype(np.uint8)

    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=100)
    ax.imshow(overlay)

    for detection in all_detections:
        boxes = to_numpy(detection["boxes"])
        scores = to_numpy(detection["scores"])
        class_name = detection.get("original_class_name", detection["class_name"])
        color_norm = detection["color"] / 255.0

        if boxes.ndim == 1 and boxes.size == 4:
            boxes = boxes[None, ...]
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [float(v) for v in box]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color_norm, facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0.0, y1 - 5),
                f"{class_name} {float(score):.2f}",
                bbox={"facecolor": color_norm, "alpha": 0.7, "edgecolor": "white", "linewidth": 1},
                fontsize=9,
                color="white",
                weight="bold",
                fontfamily="sans-serif",
            )

    ax.axis("off")
    plt.tight_layout(pad=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_filename = f"result_{timestamp}.jpg"
    result_path = RESULT_DIR / result_filename

    plt.savefig(result_path, dpi=150, bbox_inches="tight", format="jpg")
    plt.close(fig)

    return result_filename


def pil_to_bgr_numpy(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
