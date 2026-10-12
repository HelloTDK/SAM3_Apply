"""Ascend OM text, box and cross-image visual prompt inference."""

import logging
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from .acl_runtime import ACLModel, ACLRuntime


LOGGER = logging.getLogger(__name__)


def empty_arrays(height, width):
    return {
        "masks": np.zeros((0, height, width), dtype=bool),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros(0, dtype=np.float32),
        "classes": np.zeros(0, dtype=np.int64),
    }


def nms_indices(boxes, scores, iou_threshold):
    order = np.argsort(-scores, kind="stable")
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        remaining = order[1:]
        overlap = np.maximum(
            np.minimum(boxes[index, 2:], boxes[remaining, 2:])
            - np.maximum(boxes[index, :2], boxes[remaining, :2]), 0,
        ).prod(axis=1)
        union = areas[index] + areas[remaining] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess(outputs, orig_size, threshold, iou_threshold, max_candidates=None):
    height, width = orig_size
    scores = (1 / (1 + np.exp(-np.clip(outputs["pred_logits"][0].reshape(-1), -80, 80)))) * (
        1 / (1 + np.exp(-np.clip(outputs["presence_logits"][0, 0], -80, 80)))
    )
    boxes = outputs["pred_boxes"][0].copy()  # This export returns normalized xyxy.
    boxes *= np.asarray([width, height, width, height], dtype=np.float32)
    boxes = np.clip(boxes, 0, [width, height, width, height]).astype(np.float32)
    valid = np.flatnonzero(
        (scores > threshold) & np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    )
    keep = valid[nms_indices(boxes[valid], scores[valid], iou_threshold)]
    if max_candidates is not None:
        keep = keep[:max_candidates]
    arrays = empty_arrays(height, width)
    if keep.size:
        arrays = {
            "masks": np.asarray([
                cv2.resize(outputs["pred_masks"][0, i].astype(np.float32), (width, height),
                           interpolation=cv2.INTER_LINEAR) > 0
                for i in keep
            ]),
            "boxes": boxes[keep],
            "scores": scores[keep].astype(np.float32),
            "classes": np.zeros(len(keep), dtype=np.int64),
        }
    return arrays, len(scores)


class OMBackend:
    def __init__(self, model_dir, device_id, iou_threshold):
        self.model_dir = model_dir
        self.device_id = device_id
        self.iou_threshold = iou_threshold
        self.models = {}
        self.model = None
        self.features = None
        self.orig_size = None
        names = ["vision-encoder", "text-encoder", "geometry-encoder",
                 "decoder", "decoder-pos", "decoder-mix"]
        for name in names + ["tokenizer"]:
            path = model_dir / (name + (".json" if name == "tokenizer" else ".om"))
            if not path.is_file():
                raise FileNotFoundError(f"OM resource not found: {path}")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)
        self.acl_runtime = ACLRuntime(device_id)
        self.setup_model()

    def bind_context(self):
        self.acl_runtime.bind_context()

    def setup_model(self):
        self.bind_context()
        if self.model is not None:
            return
        try:
            for name in ["vision-encoder", "text-encoder", "geometry-encoder",
                         "decoder", "decoder-pos", "decoder-mix"]:
                self.models[name] = ACLModel(self.model_dir / f"{name}.om")
            self.box_capacity = self.models["geometry-encoder"].in_shapes[0][1]
            self.visual_capacity = self.models["decoder-pos"].in_shapes[4][1]
            self.imgsz = self.models["vision-encoder"].in_shapes[0][-1]
            self.model = self.models
        except Exception:
            self.close(finalize=True)
            raise

    def reset_image(self):
        self.features = None
        self.orig_size = None

    def close(self, *, finalize=False):
        if self.models:
            self.bind_context()
        for model in reversed(list(self.models.values())):
            try:
                model.close()
            except Exception:
                LOGGER.exception("释放 OM 模型失败: %s", model.path)
        self.models.clear()
        self.model = None
        self.reset_image()
        self.encode_text.cache_clear()
        if finalize:
            self.acl_runtime.close()

    def set_image(self, image):
        image = image.convert("RGB")
        self.orig_size = (image.height, image.width)
        size = self.models["vision-encoder"].in_shapes[0][-2:]
        pixels = np.asarray(image.resize((size[1], size[0]), Image.Resampling.BILINEAR))
        pixels = (pixels.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        outputs = self.models["vision-encoder"].run([pixels])
        if len(outputs) != 4:
            raise ValueError("Vision OM must output three FPN features and fpn_pos_2")
        self.features = dict(zip(["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"], outputs))
        return self.features

    @lru_cache(maxsize=128)
    def encode_text(self, text):
        encoded = self.tokenizer.encode(text)
        return self.models["text-encoder"].run([
            np.asarray([encoded.ids]), np.asarray([encoded.attention_mask]),
        ])

    def encode_visual(self, features, orig_size, boxes_xywh, labels=None):
        height, width = orig_size
        if not boxes_xywh:
            raise ValueError("At least one visual prompt box is required")
        labels = [1] * len(boxes_xywh) if labels is None else labels
        if len(labels) != len(boxes_xywh) or any(label not in {0, 1} for label in labels):
            raise ValueError("box_labels must match boxes and contain only 0 or 1")
        boxes = np.asarray(boxes_xywh, dtype=np.float32).copy()
        boxes[:, :2] += boxes[:, 2:] / 2
        boxes /= [width, height, width, height]
        embeddings, masks = [], []
        for start in range(0, len(boxes), self.box_capacity):
            chunk = boxes[start:start + self.box_capacity]
            chunk_labels = np.asarray(labels[start:start + self.box_capacity], dtype=np.int64)
            indices = np.arange(self.box_capacity) % len(chunk)
            # shortcut: OM has no box mask; repeat real boxes, re-export with a mask for exact variable-box attention.
            embed, mask = self.models["geometry-encoder"].run([
                chunk[indices][None], chunk_labels[indices][None],
                features["fpn_feat_2"], features["fpn_pos_2"],
            ])
            selected = list(range(len(chunk))) + [self.box_capacity]  # Keep the CLS token.
            embeddings.append(embed[:, selected])
            masks.append(mask[:, selected])
        return np.concatenate(embeddings, axis=1), np.concatenate(masks, axis=1)

    def grounding(self, features, orig_size, visual_embed=None, visual_mask=None,
                  text=None, confidence_threshold=0.0, max_candidates=None):
        start = time.perf_counter()
        if (visual_embed is None) != (visual_mask is None):
            raise ValueError("Visual embeddings and mask must be supplied together")
        if visual_embed is not None:
            valid = np.flatnonzero(visual_mask[0])
            if len(valid) > self.visual_capacity:
                raise ValueError(
                    f"OM decoder supports at most {self.visual_capacity} visual tokens per category; "
                    f"got {len(valid)} (boxes plus one CLS token per source group). "
                    "Export a decoder with a larger prompt length to use more samples."
                )
            visual_features = np.zeros((1, self.visual_capacity, visual_embed.shape[-1]), np.float32)
            visual_features[:, :len(valid)] = visual_embed[:, valid]
            visual_attention = np.zeros((1, self.visual_capacity), dtype=bool)
            visual_attention[:, :len(valid)] = True
        parts = []
        raw_count = 0
        for class_index, prompt in enumerate(text or [None]):
            if prompt is not None:
                prompt_features, prompt_mask = self.encode_text(prompt)
                decoder_name = "decoder"
                if visual_embed is not None:
                    prompt_features = np.concatenate([prompt_features, visual_features], axis=1)
                    prompt_mask = np.concatenate([prompt_mask, visual_attention], axis=1)
                    decoder_name = "decoder-mix"
            else:
                if visual_embed is None:
                    raise ValueError("At least one text or visual prompt is required")
                prompt_features, prompt_mask = visual_features, visual_attention
                decoder_name = "decoder-pos"
            raw = self.models[decoder_name].run([
                features["fpn_feat_0"], features["fpn_feat_1"], features["fpn_feat_2"],
                features["fpn_pos_2"], prompt_features, prompt_mask,
            ])
            outputs = dict(zip(["pred_masks", "pred_boxes", "pred_logits", "presence_logits"], raw))
            arrays, count = postprocess(outputs, orig_size, confidence_threshold,
                                        self.iou_threshold, max_candidates)
            arrays["classes"][:] = class_index
            parts.append(arrays)
            raw_count += count
        arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        return arrays, {
            "grounding_forward_ms": int((time.perf_counter() - start) * 1000),
            "raw_candidate_count": raw_count,
            "kept_candidate_count": len(arrays["scores"]),
        }

    def predict(self, image, *, text=None, bboxes=None, confidence_threshold=0.3,
                reset_cached_image=True):
        features = self.set_image(image) if reset_cached_image or self.features is None else self.features
        if bboxes is None:
            return self.grounding(features, (image.height, image.width), text=text,
                                  confidence_threshold=confidence_threshold)[0]
        boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in bboxes]
        parts = []
        for start in range(0, len(boxes_xywh), self.box_capacity):
            embed, mask = self.encode_visual(features, (image.height, image.width),
                                             boxes_xywh[start:start + self.box_capacity])
            parts.append(self.grounding(features, (image.height, image.width), embed, mask,
                                        text, confidence_threshold)[0])
        return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
