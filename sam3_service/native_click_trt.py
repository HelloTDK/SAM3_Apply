"""TensorRT runtime for SAM3's native point-prompt and mask-refinement branch."""

import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image


@contextmanager
def timed_step(action):
    """Print wall time even when a step fails; GPU callers synchronize before exit."""
    timing = {}
    started = perf_counter()
    try:
        yield timing
    finally:
        timing["ms"] = (perf_counter() - started) * 1000
        print(f"[SAM3][TensorRT] {action}耗时：{timing['ms']:.2f} ms", flush=True)


def timed_interaction(action):
    """Keep Gradio's callback signature and append the total to its status output."""

    def decorate(callback):
        @wraps(callback)
        def measured(*args, **kwargs):
            with timed_step(f"{action}总计") as timing:
                result = callback(*args, **kwargs)
            return (*result[:-1], f"{result[-1]}，耗时：{timing['ms']:.2f} ms")

        return measured

    return decorate


class TensorRTEngine:
    """Execute one TensorRT engine using CuPy-backed tensor addresses."""

    def __init__(self, engine_path: Path):
        import cupy as cp
        import tensorrt as trt

        self.cp = cp
        self.trt = trt
        self.name = engine_path.stem
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        with engine_path.open("rb") as engine_file, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT engine load failed: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"TensorRT context creation failed: {engine_path}")
        self.stream = cp.cuda.Stream()
        self.lock = threading.Lock()
        self.tensors = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.tensors[name] = (
                self.engine.get_tensor_mode(name),
                self.engine.get_tensor_dtype(name),
                tuple(self.engine.get_tensor_shape(name)),
            )

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.tensors[name][2]

    def _dtype(self, dtype):
        mapping = {
            self.trt.float32: self.cp.float32,
            self.trt.float16: self.cp.float16,
            self.trt.int32: self.cp.int32,
            self.trt.int64: self.cp.int64,
        }
        if dtype not in mapping:
            raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
        return mapping[dtype]

    def __call__(self, **inputs):
        # Copies and inference share one stream so step boundaries include completed work.
        with self.lock, self.stream:
            expected = {
                name for name, (mode, _, _) in self.tensors.items()
                if mode == self.trt.TensorIOMode.INPUT
            }
            missing = expected.difference(inputs)
            if missing:
                raise ValueError(f"Missing TensorRT inputs: {', '.join(sorted(missing))}")

            with timed_step(f"{self.name} 输入传输与缓冲准备"):
                gpu_inputs = {}
                gpu_outputs = {}
                for name in expected:
                    dtype = self._dtype(self.tensors[name][1])
                    value = self.cp.asarray(inputs[name], dtype=dtype)
                    if not value.flags.c_contiguous:
                        value = self.cp.ascontiguousarray(value)
                    gpu_inputs[name] = value
                    if not self.context.set_input_shape(name, tuple(value.shape)):
                        raise ValueError(f"Input shape outside TensorRT profile: {name}={value.shape}")
                    self.context.set_tensor_address(name, int(value.data.ptr))

                for name, (mode, dtype, _) in self.tensors.items():
                    if mode != self.trt.TensorIOMode.OUTPUT:
                        continue
                    shape = tuple(self.context.get_tensor_shape(name))
                    if any(dimension < 0 for dimension in shape):
                        raise RuntimeError(f"TensorRT could not resolve output shape: {name}={shape}")
                    output = self.cp.empty(shape, dtype=self._dtype(dtype))
                    gpu_outputs[name] = output
                    self.context.set_tensor_address(name, int(output.data.ptr))
                self.stream.synchronize()

            with timed_step(f"{self.name} 推理"):
                if not self.context.execute_async_v3(stream_handle=self.stream.ptr):
                    raise RuntimeError("TensorRT execution failed")
                self.stream.synchronize()
            return gpu_outputs


def prepare_points(clicks, image_size, target_size):
    """Scale original-image click coordinates to the square SAM3 input frame."""
    width, height = image_size
    target_width, target_height = target_size
    coords = np.asarray(
        [[x * target_width / width, y * target_height / height] for x, y, _ in clicks],
        dtype=np.float32,
    )[None]
    labels = np.asarray([[label for _, _, label in clicks]], dtype=np.int32)
    return coords, labels


def remove_small_mask_regions(mask, clicks):
    """Drop islands below 1% of the largest region, keeping positive-click regions."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    keep = stats[:, cv2.CC_STAT_AREA] >= stats[1:, cv2.CC_STAT_AREA].max() * 0.01
    for x, y, label in clicks:
        if label == 1:
            # Explicitly selected small objects must survive the noise filter.
            keep[labels[int(y), int(x)]] = True
    keep[0] = False
    return keep[labels]


class NativeInteractiveBackend:
    """Cache encoder features and run the SAM3 first-click/refinement engines."""

    MAX_POINTS = 32

    def __init__(self, engine_dir: Path, encoder=None):
        engine_dir = Path(engine_dir)
        # The HTTP service injects its shared encoder and execution context here.
        self.encoder = encoder
        if self.encoder is None:
            with timed_step("加载图像编码器 engine"):
                self.encoder = TensorRTEngine(engine_dir / "interactive-image-encoder.engine")
        with timed_step("加载首点解码器 engine"):
            self.first_decoder = TensorRTEngine(engine_dir / "interactive-decoder-first.engine")
        with timed_step("加载细化解码器 engine"):
            self.refine_decoder = TensorRTEngine(engine_dir / "interactive-decoder-refine.engine")
        _, channels, height, width = self.encoder.get_tensor_shape("images")
        if (channels, height, width) != (3, 1008, 1008):
            raise ValueError(f"Unexpected SAM3 image input shape: {self.encoder.get_tensor_shape('images')}")
        self.target_size = (width, height)
        self.max_clicks = self.MAX_POINTS

    @timed_step("图片编码合计")
    def encode_image(self, image: Image.Image):
        with timed_step("图片预处理"):
            width, height = self.target_size
            rgb = np.asarray(
                image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            pixels = np.ascontiguousarray((rgb / 127.5 - 1.0).transpose(2, 0, 1)[None])
        output = self.encoder(images=pixels)
        # Keep only native features in TCP sessions, releasing unused detection outputs.
        return {name: output[name] for name in ("image_embed", "high_res_0", "high_res_1")}

    @timed_step("点击推理合计")
    def segment(self, features, image_size, clicks, previous_logits=None):
        if not clicks or len(clicks) > self.max_clicks:
            raise ValueError(f"SAM3 TensorRT supports 1 to {self.max_clicks} points")
        with timed_step("点击与 mask 提示准备"):
            coords, labels = prepare_points(clicks, image_size, self.target_size)
            decoder = self.refine_decoder if previous_logits is not None else self.first_decoder
            inputs = {
                **features,
                "point_coords": coords,
                "point_labels": labels,
            }
            if previous_logits is not None:
                inputs["mask_input"] = np.asarray(previous_logits, dtype=np.float32).reshape(
                    1, 1, 288, 288
                )
        output = decoder(**inputs)
        with timed_step("输出回传 CPU"):
            masks = output["masks"].get()[0].astype(np.float32)
            scores = output["scores"].get()[0].astype(np.float32)
        with timed_step("mask 后处理"):
            best = int(np.argmax(scores))
            logits = masks[best][None, None]
            width, height = image_size
            mask = cv2.resize(logits[0, 0], (width, height), interpolation=cv2.INTER_LINEAR) > 0.0
            mask = remove_small_mask_regions(mask, clicks)
        return mask, float(scores[best]), logits
