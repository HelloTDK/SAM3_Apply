"""SAM3 运行时兼容入口，所有推理统一转发到 TensorRT 后端。"""

from . import trt_runtime as _trt_runtime
from .trt_runtime import (
    INFERENCE_SEMAPHORE,
    INFERENCE_STATE_LOCK,
    MODEL_LOCK,
    device,
    get_cuda_memory_stats,
    predictor,
    prepare_multi_visual_prompt_state,
    run_box_segmentation_pipeline,
    run_detection_pipeline,
    run_inference_in_thread,
    run_multi_similar_object_batch_pipeline,
    run_multi_visual_prompt_query_with_state,
    run_similar_object_batch_pipeline,
    run_similar_object_pipeline,
)


def __getattr__(name: str):
    """动态读取会变化的运行状态，避免整数在兼容模块中被静态复制。"""
    if name in {"ACTIVE_INFERENCE_COUNT", "IDLE_MODEL_UNLOADED", "LAST_INFERENCE_FINISHED_AT"}:
        return getattr(_trt_runtime, name)
    raise AttributeError(name)

__all__ = [
    "INFERENCE_SEMAPHORE",
    "INFERENCE_STATE_LOCK",
    "MODEL_LOCK",
    "device",
    "get_cuda_memory_stats",
    "predictor",
    "prepare_multi_visual_prompt_state",
    "run_box_segmentation_pipeline",
    "run_detection_pipeline",
    "run_inference_in_thread",
    "run_multi_similar_object_batch_pipeline",
    "run_multi_visual_prompt_query_with_state",
    "run_similar_object_batch_pipeline",
    "run_similar_object_pipeline",
]
