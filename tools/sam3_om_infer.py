#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于昇腾 (Ascend) ACL 的 SAM3 风格分割推理脚本
(vision encoder / text encoder / [geometry encoder] / decoder 四个 .om 模型)

用法示例:
    # 文本 prompt
    python sam3_om_infer.py --image test.jpg --text "car window"

    # 文本 + box prompt (需要 geometry-encoder.om)
    python sam3_om_infer.py --image test.jpg --text "car" \
        --geometry-om ./geometry-encoder.om \
        --box "pos:100,100,200,200;neg:300,300,50,50"

"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import acl
from tokenizers import Tokenizer


TARGET_SIZE = 1008
PAD_TOKEN_ID = 49407
MAX_TEXT_LEN = 32

ACL_SUCCESS = 0
ACL_MEM_MALLOC_NORMAL_ONLY = getattr(acl, "ACL_MEM_MALLOC_NORMAL_ONLY", 0)
ACL_MEMCPY_HOST_TO_DEVICE = getattr(acl, "ACL_MEMCPY_HOST_TO_DEVICE", 1)
ACL_MEMCPY_DEVICE_TO_HOST = getattr(acl, "ACL_MEMCPY_DEVICE_TO_HOST", 2)

_ACL_DTYPE_NAMES = [
    "ACL_FLOAT", "ACL_FLOAT16", "ACL_DOUBLE",
    "ACL_INT8", "ACL_INT16", "ACL_INT32", "ACL_INT64",
    "ACL_UINT8", "ACL_UINT16", "ACL_UINT32", "ACL_UINT64",
    "ACL_BOOL",
]
_ACL_TO_NP = {
    "ACL_FLOAT": np.float32, "ACL_FLOAT16": np.float16, "ACL_DOUBLE": np.float64,
    "ACL_INT8": np.int8, "ACL_INT16": np.int16, "ACL_INT32": np.int32,
    "ACL_INT64": np.int64, "ACL_UINT8": np.uint8, "ACL_UINT16": np.uint16,
    "ACL_UINT32": np.uint32, "ACL_UINT64": np.uint64, "ACL_BOOL": np.uint8,
}


def check_ret(message, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError("{} failed, ret = {}".format(message, ret))


_ACL_STATE = {"ctx": None, "device": None}


def init_acl(device_id):
    """幂等初始化 ACL，重复调用会复用已有 context。"""
    if _ACL_STATE["ctx"] is not None:
        print("[INFO] ACL 已初始化，复用现有 context")
        return _ACL_STATE["ctx"], _ACL_STATE["device"]
    acl.init() 
    ret = acl.rt.set_device(device_id)
    check_ret("acl.rt.set_device", ret)
    ctx, ret = acl.rt.create_context(device_id)
    check_ret("acl.rt.create_context", ret)
    _ACL_STATE.update(ctx=ctx, device=device_id)
    return ctx, device_id


def finalize_acl():
    if _ACL_STATE["ctx"] is None:
        print("[INFO] 无需释放（未初始化）")
        return
    acl.rt.destroy_context(_ACL_STATE["ctx"])
    acl.rt.reset_device(_ACL_STATE["device"])
    acl.finalize()
    _ACL_STATE["ctx"] = None
    print("[OK] ACL 已释放")


def load_model(model_path):
    model_id, ret = acl.mdl.load_from_file(model_path)
    check_ret("acl.mdl.load_from_file: " + model_path, ret)

    desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(desc, model_id)
    check_ret("acl.mdl.get_desc", ret)

    num_in = acl.mdl.get_num_inputs(desc)
    num_out = acl.mdl.get_num_outputs(desc)
    in_sizes = [acl.mdl.get_input_size_by_index(desc, i) for i in range(num_in)]

    out_shapes, out_dtypes, out_sizes, out_bufs = [], [], [], []
    out_dataset = acl.mdl.create_dataset()
    for i in range(num_out):
        size = acl.mdl.get_output_size_by_index(desc, i)
        dims = acl.mdl.get_output_dims(desc, i)
        shape = tuple(dims[0]["dims"])
        acl_dt = acl.mdl.get_output_data_type(desc, i)
        np_dtype = np.float32
        for name in _ACL_DTYPE_NAMES:
            if getattr(acl, name, None) == acl_dt:
                np_dtype = _ACL_TO_NP[name]
                break
        else:
            print("[WARN] {} 输出[{}] 未知类型 {}，按 float32".format(
                Path(model_path).name, i, acl_dt))
        n_elem = int(np.prod(shape)) if shape else 1
        if n_elem * np_dtype().itemsize != size:
            for dt in (np.int64, np.int32, np.int16, np.int8, np.uint8,
                       np.float32, np.float64):
                if n_elem * dt().itemsize == size:
                    np_dtype = dt
                    break
            print("[INFO] {} 输出[{}] 描述 dtype 与 om size 不符，按 {} 处理".format(
                Path(model_path).name, i, np_dtype.__name__))
        if any(d <= 0 for d in shape):
            shape = (size // np_dtype().itemsize,)
            print("[WARN] {} 输出[{}] 含动态维度，按 {} 处理".format(
                Path(model_path).name, i, shape))
        buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc output %d" % i, ret)
        data_buffer = acl.create_data_buffer(buf, size)
        _, ret = acl.mdl.add_dataset_buffer(out_dataset, data_buffer)
        check_ret("acl.mdl.add_dataset_buffer output %d" % i, ret)
        out_shapes.append(shape)
        out_dtypes.append(np_dtype)
        out_sizes.append(size)
        out_bufs.append(buf)

    print("  Loaded {}: {} in, {} out".format(Path(model_path).name, num_in, num_out))
    print("    out shapes:", out_shapes)
    return {
        "path": model_path, "model_id": model_id, "desc": desc,
        "num_in": num_in, "num_out": num_out, "in_sizes": in_sizes,
        "out_shapes": out_shapes, "out_dtypes": out_dtypes,
        "out_sizes": out_sizes, "out_bufs": out_bufs, "out_dataset": out_dataset,
    }


def _fit_input(model, index, arr):
    """若输入字节数与 om 要求不符，尝试转换 dtype 使其匹配。"""
    arr = np.ascontiguousarray(arr)
    if arr.nbytes == model["in_sizes"][index]:
        return arr
    for target in (np.int32, np.int16, np.int8, np.uint8,
                   np.float32, np.float64, np.bool_):
        if target == arr.dtype:
            continue
        cand = arr.astype(target)
        if cand.nbytes == model["in_sizes"][index]:
            print("[INFO] {} 输入[{}] {} -> {}".format(
                Path(model["path"]).name, index, arr.dtype, target))
            return cand
    raise RuntimeError(
        "{} 输入[{}] 字节数 {} 与 om 要求 {} 不匹配，请检查 dtype/shape".format(
            Path(model["path"]).name, index, arr.nbytes, model["in_sizes"][index]))


def run_model(model, input_arrays):
    if len(input_arrays) != model["num_in"]:
        raise ValueError("{} 期望 {} 个输入，实际 {} 个".format(
            Path(model["path"]).name, model["num_in"], len(input_arrays)))

    in_dataset = acl.mdl.create_dataset()
    in_bufs, in_dbuffers = [], []
    try:
        for i, arr in enumerate(input_arrays):
            data = _fit_input(model, i, arr)
            size = model["in_sizes"][i]
            dev_buf, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc input %d" % i, ret)
            in_bufs.append(dev_buf)
            if hasattr(acl.util, "bytes_to_ptr"):
                data_bytes = data.tobytes()
                host_ptr = acl.util.bytes_to_ptr(data_bytes)
            else:
                host_ptr = acl.util.numpy_to_ptr(data)
            ret = acl.rt.memcpy(dev_buf, size, host_ptr, data.nbytes,
                                ACL_MEMCPY_HOST_TO_DEVICE)
            check_ret("acl.rt.memcpy H2D input %d" % i, ret)
            dbuf = acl.create_data_buffer(dev_buf, size)
            in_dbuffers.append(dbuf)
            _, ret = acl.mdl.add_dataset_buffer(in_dataset, dbuf)
            check_ret("acl.mdl.add_dataset_buffer input %d" % i, ret)

        ret = acl.mdl.execute(model["model_id"], in_dataset, model["out_dataset"])
        check_ret("acl.mdl.execute: " + model["path"], ret)


        outputs = []
        for i in range(model["num_out"]):
            out_size = model["out_sizes"][i]
            np_dtype = model["out_dtypes"][i]
            n_elem = int(np.prod(model["out_shapes"][i])) \
                if model["out_shapes"][i] else 1
            host_buf, ret = acl.rt.malloc_host(out_size)
            check_ret("acl.rt.malloc_host output %d" % i, ret)
            try:
                ret = acl.rt.memcpy(host_buf, out_size,
                                    model["out_bufs"][i], out_size,
                                    ACL_MEMCPY_DEVICE_TO_HOST)
                check_ret("acl.rt.memcpy D2H output %d" % i, ret)
                want_bytes = n_elem * np_dtype().itemsize
                if want_bytes != out_size:
                    print("[WARN] {} 输出[{}] shape 字节数 {} != om size {}，"
                          "按实际 {}B 展开".format(
                              Path(model["path"]).name, i,
                              want_bytes, out_size, out_size))
                n_fill = n_elem if want_bytes == out_size \
                    else out_size // np_dtype().itemsize
                shape = model["out_shapes"][i] if n_fill == n_elem else (n_fill,)
                if hasattr(acl.util, "ptr_to_bytes"):
                    raw = acl.util.ptr_to_bytes(host_buf, out_size)
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
                else:
                    arr = acl.util.ptr_to_numpy(
                        host_buf, shape, np.dtype(np_dtype).num).copy()
            finally:
                acl.rt.free_host(host_buf)
            outputs.append(arr)
        return outputs
    finally:
        for dbuf in in_dbuffers:
            acl.destroy_data_buffer(dbuf)
        for buf in in_bufs:
            acl.rt.free(buf)
        acl.mdl.destroy_dataset(in_dataset)


def unload_model(model):
    num = acl.mdl.get_dataset_num_buffers(model["out_dataset"])
    for i in range(num):
        buf = acl.mdl.get_dataset_buffer(model["out_dataset"], i)
        if buf:
            acl.destroy_data_buffer(buf)
    acl.mdl.destroy_dataset(model["out_dataset"])
    for buf in model["out_bufs"]:
        acl.rt.free(buf)
    acl.mdl.destroy_desc(model["desc"])
    acl.mdl.unload(model["model_id"])


def load_tokenizer(tokenizer_path):
    """返回 (tokenizer, is_transformers)。"""
    try:
        tok = Tokenizer.from_file(tokenizer_path)
        tok.enable_padding(length=MAX_TEXT_LEN, pad_id=PAD_TOKEN_ID)
        tok.enable_truncation(max_length=MAX_TEXT_LEN)
        return tok, False
    except Exception as e:
        print("Tokenizer.from_file 失败: %s；回退 transformers" % e)
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_path), True


def preprocess_image(image):
    """image: RGB uint8 (H,W,3)。Resize 到 1008x1008 并归一化到 [-1,1]，
    返回 (NCHW tensor, 原图尺寸(h,w))。"""
    from PIL import Image as PILImage
    orig_size = image.shape[:2]
    pil_image = PILImage.fromarray(image)
    resized = np.array(pil_image.resize((TARGET_SIZE, TARGET_SIZE), PILImage.BILINEAR))
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    tensor = normalized.transpose(2, 0, 1)[np.newaxis]
    return tensor, orig_size


def parse_box_prompts(box_str):
    """解析 box prompt: "pos:x,y,w,h;neg:x,y,w,h;..." (xywh, 像素)。"""
    boxes, labels = [], []
    for part in box_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("pos:"):
            label, coords = 1, part[4:]
        elif part.startswith("neg:"):
            label, coords = 0, part[4:]
        else:
            label, coords = 1, part
        x, y, w, h = [float(v) for v in coords.split(",")]
        boxes.append([x, y, w, h])
        labels.append(label)
    return boxes, labels


def xywh_to_cxcywh_normalized(boxes, img_w, img_h):
    """xywh(像素) -> cxcywh(归一化)。"""
    result = []
    for x, y, w, h in boxes:
        result.append([(x + w / 2) / img_w, (y + h / 2) / img_h,
                       w / img_w, h / img_h])
    return np.array(result, dtype=np.float32)



def encode_image(vision_model, pixel_values, fpn_pos2=None):
    """Vision Encoder：输出 4 个（含 fpn_pos_2）或 3 个 + npy 常量。"""
    outputs = run_model(vision_model, [pixel_values])
    if len(outputs) == 4:
        return {
            "fpn_feat_0": outputs[0],
            "fpn_feat_1": outputs[1],
            "fpn_feat_2": outputs[2],
            "fpn_pos_2": outputs[3],
        }
    if len(outputs) == 3 and fpn_pos2 is not None:
        if fpn_pos2.ndim == 3:
            fpn_pos2 = fpn_pos2[np.newaxis]
        return {
            "fpn_feat_0": outputs[0],
            "fpn_feat_1": outputs[1],
            "fpn_feat_2": outputs[2],
            "fpn_pos_2": fpn_pos2,
        }
    raise RuntimeError("vision om 输出个数 {} 无法识别（期望 4 个，或 3 个 + npy）".format(
        len(outputs)))


def encode_text(text_model, tokenizer, is_transformers, text):
    """Text Encoder：返回 (text_features, text_mask)。"""
    if is_transformers:
        encoded = tokenizer(text, padding="max_length", truncation=True,
                            max_length=MAX_TEXT_LEN)
        input_ids = np.array([encoded["input_ids"]], dtype=np.int64)
        attention_mask = np.array([encoded["attention_mask"]], dtype=np.int64)
    else:
        tokenizer.enable_padding(pad_id=PAD_TOKEN_ID, length=MAX_TEXT_LEN)
        tokenizer.enable_truncation(max_length=MAX_TEXT_LEN)
        encoded = tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    outputs = run_model(text_model, [input_ids, attention_mask])
    return outputs[0], outputs[1]


def encode_boxes(geometry_model, boxes, labels, fpn_feat, fpn_pos):
    """Geometry Encoder（box prompt，需要 geometry-encoder.om）。"""
    outputs = run_model(geometry_model, [
        boxes.astype(np.float32),
        labels.astype(np.int64),
        fpn_feat,
        fpn_pos,
    ])
    return outputs[0], outputs[1]


def decode(decoder_model, vision_features, prompt_features, prompt_mask):
    """Decoder：融合 image/prompt feature，输出 mask/box/logits。"""
    outputs = run_model(decoder_model, [
        vision_features["fpn_feat_0"],
        vision_features["fpn_feat_1"],
        vision_features["fpn_feat_2"],
        vision_features["fpn_pos_2"],
        prompt_features,
        prompt_mask,
    ])
    return {
        "pred_masks": outputs[0],
        "pred_boxes": outputs[1],
        "pred_logits": outputs[2],
        "presence_logits": outputs[3],
    }



def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _postprocess(outputs, orig_size, conf_threshold, input_boxes=None):
    pred_masks = outputs["pred_masks"][0]
    pred_boxes = outputs["pred_boxes"][0]
    pred_logits = outputs["pred_logits"][0].reshape(-1)
    presence_logits = outputs["presence_logits"][0, 0]
    scores = _sigmoid(pred_logits) * _sigmoid(presence_logits)
    keep = scores > conf_threshold
    h, w = orig_size

    masks = []
    for m in pred_masks[keep]:
        mask_resized = cv2.resize(m.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_LINEAR)
        masks.append(mask_resized > 0)

    if not np.any(keep):
        return {"masks": masks, "boxes": None, "scores": None,
                "orig_size": orig_size, "input_boxes": input_boxes}

    boxes = pred_boxes[keep].copy()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    boxes = np.clip(boxes, 0, [[w, h, w, h]])
    return {"masks": masks, "boxes": boxes, "scores": scores[keep],
            "orig_size": orig_size, "input_boxes": input_boxes}


def visualize_results(image_bgr, results, output_path, alpha=0.35):
    vis = image_bgr.copy()
    colors = [
        (30, 144, 255), (255, 144, 30), (144, 255, 30),
        (255, 30, 144), (30, 255, 144),
    ]
    if results.get("boxes") is None or results.get("scores") is None:
        cv2.imwrite(output_path, vis)
        print("  ✓ 未检出目标，原图已保存: {}".format(output_path))
        return vis
    for i, (mask, box, score) in enumerate(
            zip(results["masks"], results["boxes"], results["scores"])):
        color = colors[i % len(colors)]
        mask_bool = mask > 0
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, "{:.2f}".format(score), (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(output_path, vis)
    print("  ✓ Saved: {}".format(output_path))
    return vis



def predict(image_rgb, text, boxes, box_labels, conf_threshold,
            vision_model, text_model, decoder_model, geometry_model,
            tokenizer, is_transformers, fpn_pos2=None):
    pixel_values, orig_size = preprocess_image(image_rgb)
    vision_features = encode_image(vision_model, pixel_values, fpn_pos2)
    h, w = orig_size

    if text:
        text_features, text_mask = encode_text(
            text_model, tokenizer, is_transformers, text)
    else:
        pad_ids = np.full((1, MAX_TEXT_LEN), PAD_TOKEN_ID, dtype=np.int64)
        pad_mask = np.zeros((1, MAX_TEXT_LEN), dtype=np.int64)
        pad_mask[0, 0] = 1
        text_features, text_mask = run_model(text_model, [pad_ids, pad_mask])[:2]

    if boxes:
        if geometry_model is None:
            raise RuntimeError("使用 box prompt 需要指定 --geometry-om")
        boxes_array = xywh_to_cxcywh_normalized(boxes, w, h).reshape(1, -1, 4)
        if box_labels:
            labels_array = np.array(box_labels, dtype=np.int64).reshape(1, -1)
        else:
            labels_array = np.ones((1, len(boxes)), dtype=np.int64)
        geom_features, geom_mask = encode_boxes(
            geometry_model, boxes_array, labels_array,
            vision_features["fpn_feat_2"], vision_features["fpn_pos_2"])
        prompt_features = np.concatenate([text_features, geom_features], axis=1)
        prompt_mask = np.concatenate(
            [text_mask.astype(np.int64), geom_mask.astype(np.int64)], axis=1)
    else:
        prompt_features = text_features
        prompt_mask = text_mask.astype(np.int64)

    outputs = decode(decoder_model, vision_features, prompt_features, prompt_mask)
    return _postprocess(outputs, orig_size, conf_threshold, boxes)


def parse_args():
    p = argparse.ArgumentParser(description="Ascend OM 分割推理 (vision/text/geometry/decoder)")
    p.add_argument("--image", required=True, help="输入图片路径")
    p.add_argument("--vision-om", default="./om_weights/vision-encoder.om")
    p.add_argument("--text-om", default="./om_weights/text-encoder.om")
    p.add_argument("--decoder-om", default="./om_weights/decoder.om")
    p.add_argument("--geometry-om", default="./om_weights/geometry-encoder.om", help="使用 box prompt 时需要")
    p.add_argument("--tokenizer", default="./om_weights/tokenizer.json", help="HuggingFace tokenizer.json")
    p.add_argument("--fpn-pos2", default=None,
                   help="vision om 只输出 3 个 fpn 特征时提供 fpn_pos_2 的 npy 路径")
    p.add_argument("--text", default=None, help="文本 prompt")
    p.add_argument("--box", default=None,
                   help='box prompt, 如 "pos:100,100,200,200;neg:300,300,50,50" (xywh 像素)')
    p.add_argument("--output", default="result.jpg", help="可视化结果保存路径")
    p.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    p.add_argument("--device-id", type=int, default=0, help="昇腾设备号")
    return p.parse_args()


def main():
    args = parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print("无法读取图片: {}".format(args.image), file=sys.stderr)
        return 1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    boxes, box_labels = None, None
    if args.box:
        boxes, box_labels = parse_box_prompts(args.box)
        print("  Box prompts: {} 个, labels={}".format(len(boxes), box_labels))
        if not args.geometry_om:
            print("使用 --box 需要同时指定 --geometry-om", file=sys.stderr)
            return 1
    if args.text:
        print("  Text prompt: '{}'".format(args.text))

    init_acl(args.device_id)
    models = []
    try:
        vision_model = load_model(args.vision_om); models.append(vision_model)
        text_model = load_model(args.text_om); models.append(text_model)
        decoder_model = load_model(args.decoder_om); models.append(decoder_model)
        geometry_model = None
        if args.geometry_om:
            geometry_model = load_model(args.geometry_om); models.append(geometry_model)

        tokenizer, is_transformers = load_tokenizer(args.tokenizer)
        fpn_pos2 = np.load(args.fpn_pos2).astype(np.float32) if args.fpn_pos2 else None
        print("[OK] 模型与 tokenizer 加载完成")

        t_start = time.time()
        results = predict(
            image_rgb, args.text, boxes, box_labels, args.conf,
            vision_model, text_model, decoder_model, geometry_model,
            tokenizer, is_transformers, fpn_pos2)
        elapsed = time.time() - t_start

        print("检测耗时: {:.3f} 秒".format(elapsed))
        print("检出 {} 个目标".format(len(results["masks"])))
        if results.get("boxes") is not None:
            for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
                print("  [{:2d}] score={:.3f} box=[{:.0f}, {:.0f}, {:.0f}, {:.0f}]".format(
                    i, score, box[0], box[1], box[2], box[3]))

        visualize_results(image_bgr, results, args.output)
    finally:
        for m in models:
            try:
                unload_model(m)
            except Exception as e:  # 释放阶段不因单个失败而中断
                print("[WARN] unload 失败: {}".format(e))
        finalize_acl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
