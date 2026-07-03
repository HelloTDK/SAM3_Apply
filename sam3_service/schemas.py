from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .image_utils import normalize_box_segmentation_inputs


class SegmentationRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    image_base64: str = Field(..., description="Base64 image string or data URL")
    prompt: str = Field(..., description="Classes separated by ';' or ','")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)


class BoxSegmentationRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    image_base64: str = Field(..., description="Base64 image string or data URL")
    bnd_points: Optional[Union[List[float], List[List[float]]]] = Field(
        default=None,
        description="[x, y, w, h] or [[x, y, w, h], ...]",
    )
    bnd_points_list: Optional[List[List[float]]] = Field(
        default=None,
        min_length=1,
        description="Multiple boxes: [[x, y, w, h], ...]",
    )
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)

    @model_validator(mode="after")
    def normalize_bnd_points(self) -> "BoxSegmentationRequest":
        payload = self.bnd_points_list if self.bnd_points_list is not None else self.bnd_points
        if payload is None:
            raise ValueError("Either bnd_points or bnd_points_list is required")

        normalized = normalize_box_segmentation_inputs(payload)
        self.bnd_points = normalized
        self.bnd_points_list = None
        return self


class SimilarObjectRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    reference_image_base64: str = Field(..., description="Base64 sample image string or data URL")
    query_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 query image string or data URL. Kept for single-image compatibility.",
    )
    query_image_base64_list: Optional[List[str]] = Field(
        default=None,
        description="Multiple base64 query image strings or data URLs.",
    )
    reference_bnd_points: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Sample object box [x, y, w, h] in reference image",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text prompt describing the similar target.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Candidate boxes to verify in query image")
    similarity_threshold: float = Field(
        default=0.6,
        ge=-1.0,
        le=1.0,
        description="兼容旧客户端的字段；当前 visual prompt 推理不使用余弦相似度过滤。",
    )
    sam_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="SAM3 grounding score 过滤阈值。",
    )
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)
    similar_mode: Literal["feature_match", "same_image_prompt"] = Field(default="feature_match")

    @model_validator(mode="after")
    def normalize_query_images(self) -> "SimilarObjectRequest":
        if self.query_image_base64_list is None:
            if self.query_image_base64 is None:
                if self.similar_mode == "same_image_prompt":
                    self.query_image_base64_list = []
                    return self
                raise ValueError("Either query_image_base64 or query_image_base64_list is required")
            self.query_image_base64_list = [self.query_image_base64]
        elif len(self.query_image_base64_list) == 0 and self.similar_mode != "same_image_prompt":
            raise ValueError("query_image_base64_list must contain at least one image")
        return self


class MultiSimilarInstanceRequest(BaseModel):
    instance_id: Optional[str] = Field(default=None, max_length=128)
    sample_type: Literal["positive", "negative"] = Field(
        default="positive",
        description="Whether this reference instance is a positive or negative sample",
    )
    is_negative: Optional[bool] = Field(default=None, description="Compatibility flag for negative samples")
    category: str = Field(..., min_length=1, max_length=128, description="Target category mapped to this sample")
    reference_bnd_points: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Sample object box [x, y, w, h] in reference image",
    )
    paste_bnd_points: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional larger paste region [x, y, w, h] in reference image",
    )
    prompt: Optional[str] = Field(default=None, max_length=256, description="Optional text prompt for box+prompt mode")


class MultiSimilarSampleRequest(BaseModel):
    sample_id: Optional[str] = Field(default=None, max_length=128)
    reference_image_base64: str = Field(..., description="Base64 sample image string or data URL")
    sample_type: Literal["positive", "negative"] = Field(
        default="positive",
        description="Whether this legacy single-instance reference is a positive or negative sample",
    )
    is_negative: Optional[bool] = Field(default=None, description="Compatibility flag for negative samples")
    category: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Target category for backward-compatible single-instance sample",
    )
    reference_bnd_points: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Single sample object box [x, y, w, h], kept for compatibility",
    )
    paste_bnd_points: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional larger paste region [x, y, w, h], kept for compatibility",
    )
    prompt: Optional[str] = Field(default=None, max_length=256, description="Optional single-instance text prompt")
    instances: Optional[List[MultiSimilarInstanceRequest]] = Field(
        default=None,
        min_length=1,
        description="Multiple target instances selected from this same sample image",
    )

    @model_validator(mode="after")
    def validate_instances_or_legacy_box(self) -> "MultiSimilarSampleRequest":
        if self.instances:
            return self
        if self.reference_bnd_points is None or not self.category:
            raise ValueError("Each sample requires either instances[] or legacy category + reference_bnd_points")
        return self


class MultiSimilarObjectRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    samples: List[MultiSimilarSampleRequest] = Field(..., min_length=1, max_length=20)
    query_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 query image string or data URL. Kept for single-image compatibility.",
    )
    query_image_base64_list: Optional[List[str]] = Field(
        default=None,
        description="Multiple base64 query image strings or data URLs.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Max results kept per category after NMS")
    similarity_threshold: float = Field(
        default=0.6,
        ge=-1.0,
        le=1.0,
        description="兼容旧客户端的字段；当前 multi visual prompt 推理不使用余弦相似度过滤。",
    )
    sam_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="SAM3 grounding score 过滤阈值。",
    )
    nms_iou: float = Field(default=0.45, ge=0.0, le=1.0)
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)

    @model_validator(mode="after")
    def normalize_query_images(self) -> "MultiSimilarObjectRequest":
        if self.query_image_base64_list is None:
            if self.query_image_base64 is None:
                raise ValueError("Either query_image_base64 or query_image_base64_list is required")
            self.query_image_base64_list = [self.query_image_base64]
        elif len(self.query_image_base64_list) == 0:
            raise ValueError("query_image_base64_list must contain at least one image")
        return self


class SimilarObjectByUrlRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    download_url: str = Field(..., min_length=1, description="Base URL used to download relative sample/query paths")
    sample_url: str = Field(..., min_length=1, description="Remote sample manifest path or URL")
    query_image_url: str = Field(..., min_length=1, description="Remote query image path or URL")
    top_k: int = Field(default=5, ge=1, le=50, description="Max results kept per category after NMS")
    similarity_threshold: float = Field(default=0.6, ge=-1.0, le=1.0)
    sam_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.45, ge=0.0, le=1.0)
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)
    return_result_image: bool = Field(default=False)


class SimilarObjectTaskCreateRequest(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=128)
    download_url: str = Field(..., min_length=1, description="Base URL used to download relative paths")
    data_type: int = Field(default=0, description="0=image list; non-zero values are treated as video list")
    data_url: str = Field(..., min_length=1, description="Remote image/video list manifest path or URL")
    sample_url: str = Field(..., min_length=1, description="Remote sample manifest path or URL")
    infer_batch_size: int = Field(default=16, ge=1, le=64)
    frame_time: int = Field(default=1, ge=0, description="Video sampling interval in frames; 0 means every frame")
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.6, ge=-1.0, le=1.0)
    sam_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.45, ge=0.0, le=1.0)
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)
    return_result_image: bool = Field(default=False)
    result_ttl_seconds: int = Field(default=86400, ge=60, le=604800)


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    role: Literal["client", "admin"] = Field(default="client")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)
