import cv2
import torch
import time
from ultralytics.models.sam import SAM3SemanticPredictor


def patch_openai_clip_tokenizer() -> None:
    """为 openai-clip 的旧版 SimpleTokenizer 补充 SAM3 所需的调用接口。"""
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 需要 CLIP tokenizer，请在当前 yolo 环境安装 openai-clip。"
        ) from exc

    tokenizer_module = getattr(clip, "simple_tokenizer", None)
    tokenizer_cls = getattr(tokenizer_module, "SimpleTokenizer", None)
    tokenize = getattr(clip, "tokenize", None)
    if tokenizer_cls is None or tokenize is None:
        raise RuntimeError("当前 clip 包不包含 SAM3 所需的 SimpleTokenizer 或 tokenize 接口。")
    if callable(tokenizer_cls()):
        return

    def tokenizer_call(self, texts: list[str], context_length: int = 77) -> torch.Tensor:
        """将一个或多个类别文本编码为 SAM3 文本编码器需要的 token 张量。"""
        return tokenize(texts, context_length=context_length, truncate=True)

    # Ultralytics SAM3 把 SimpleTokenizer 实例当作函数调用，openai-clip 原版未实现该接口。
    tokenizer_cls.__call__ = tokenizer_call

DEVICE = "cuda:0"

MODEL_PATH = "/expdata/givap/givap-train-server-v2.2.1/src/SAM3_Apply/weights/sam3.pt"
patch_openai_clip_tokenizer()
t1 = time.time()
predictor = SAM3SemanticPredictor(
    overrides={
        "model": str(MODEL_PATH),
        "device": DEVICE,
        "conf": 0.01,
        "iou": 0.7,
        "imgsz": 1036,
        # 新版 Ultralytics 通过 quantize=16 启用 FP16；half 不再接受 torch.dtype。
        "quantize": 16,
        "save": False,
    }
)
predictor.setup_model()
t2 = time.time()
print(f"cost init model:{t2-t1}")


def run_detection_pipeline(
    image_path: str,
    prompt: list[str],
    conf: float,
    prompt_batch_size: int = 1,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """对单张图片执行多个文本类别的 SAM3 检测与分割。

    ``prompt_batch_size`` 控制一次送入 grounding encoder/decoder 的类别数。
    设为 1 可以显著降低显存峰值；代价是每个类别都要单独执行一次 decoder。
    """
    if not prompt:
        raise ValueError("prompt 至少需要包含一个文本类别")
    if prompt_batch_size < 1:
        raise ValueError("prompt_batch_size 必须是正整数")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 推理阈值由 predictor 读取，调用前同步本次请求指定的阈值。
    predictor.args.conf = conf
    predictor.set_image(image)
    features = predictor.features
    image_shape = image.shape[:2]
    print(f"image_shape:{image_shape}")

    mask_chunks = []
    box_chunks = []
    for start in range(0, len(prompt), prompt_batch_size):
        class_chunk = prompt[start : start + prompt_batch_size]
        masks, boxes = predictor.inference_features(
            dict(features),
            (image_shape[0], image_shape[1]),
            text=class_chunk,
        )
        if masks is not None and masks.numel() > 0:
            mask_chunks.append(masks)
        if boxes is not None and boxes.numel() > 0:
            # 每个 chunk 内的类别编号从 0 开始，合并前加上全局类别偏移量。
            boxes = boxes.clone()
            boxes[:, 5] += start
            box_chunks.append(boxes)

    merged_masks = torch.cat(mask_chunks, dim=0) if mask_chunks else None
    merged_boxes = (
        torch.cat(box_chunks, dim=0)
        if box_chunks
        else torch.zeros((0, 6), device=predictor.device)
    )
    return merged_masks, merged_boxes




if __name__ == "__main__":
    image_path = "/expdata/givap/givap-train-server-v2.2.1/src/0000000000000005.jpg"
    # 一个 list 元素对应一个类别，可一次传入任意多个文本类别。
    prompt = ["car"]
    for i in range(1000):
        t3 = time.time()
        # 显存有限时建议设为 1；显存充足可增大该值以换取更高吞吐。
        run_detection_pipeline(image_path, prompt, 0.1, prompt_batch_size=1)
        t4 = time.time()
        print(f"cost infer time:{t4 -t3}")
