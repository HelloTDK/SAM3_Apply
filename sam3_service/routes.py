from typing import Any, Dict, List, Optional

import html
import io
import json
import time
import traceback
from functools import partial

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from PIL import Image

from .auth import api_key_manager, require_admin_key, require_api_key
from .config import (
    CUDA_CLEANUP_AFTER_REQUEST,
    IDLE_MODEL_UNLOAD_SECONDS,
    MAX_CONCURRENT_INFERENCES,
    MAX_IMAGE_BYTES,
    MODEL_LABEL,
    SAVE_UPLOADS,
    SERIALIZE_MODEL_ACCESS,
    STATIC_DIR,
    ULTRALYTICS_IMGSZ,
)
from .image_utils import decode_base64_image, parse_optional_bnd_points_text, save_upload_image
from .runtime import (
    ACTIVE_INFERENCE_COUNT,
    IDLE_MODEL_UNLOADED,
    INFERENCE_SEMAPHORE,
    INFERENCE_STATE_LOCK,
    LAST_INFERENCE_FINISHED_AT,
    device,
    get_cuda_memory_stats,
    predictor,
    run_box_segmentation_pipeline,
    run_detection_pipeline,
    run_inference_in_thread,
    run_multi_similar_object_batch_pipeline,
    run_similar_object_batch_pipeline,
)
from .schemas import (
    BoxSegmentationRequest,
    CreateApiKeyRequest,
    MultiSimilarObjectRequest,
    SegmentationRequest,
    SimilarObjectByUrlRequest,
    SimilarObjectRequest,
    SimilarObjectTaskCreateRequest,
)
from .url_tasks import TASK_REGISTRY, run_by_url_request


def register_routes(app: FastAPI) -> None:
    def infer_auth_scope(route: APIRoute) -> str:
        dependant = getattr(route, "dependant", None)
        dependencies = getattr(dependant, "dependencies", []) if dependant is not None else []
        dependency_calls = {
            getattr(dep, "call", None).__name__
            for dep in dependencies
            if getattr(dep, "call", None) is not None
        }

        if "require_admin_key" in dependency_calls:
            return "admin_api_key"
        if "require_api_key" in dependency_calls:
            return "api_key"
        return "public"


    def get_available_api_routes() -> List[Dict[str, Any]]:
        routes: List[Dict[str, Any]] = []
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue

            methods = sorted(method for method in route.methods if method not in {"HEAD", "OPTIONS"})
            if not methods:
                continue

            routes.append(
                {
                    "name": route.name,
                    "path": route.path,
                    "methods": methods,
                    "auth": infer_auth_scope(route),
                    "summary": route.summary or "",
                }
            )

        routes.sort(key=lambda item: (item["path"], ",".join(item["methods"])))
        return routes


    def build_single_positive_sample(
        reference_image: Image.Image,
        reference_bnd_points: List[float],
        prompt: Optional[str],
    ) -> List[Dict[str, Any]]:
        prompt_text = (prompt or "").strip() or None
        return [
            {
                "sample_id": "single_reference",
                "source_image_id": "single_reference",
                "sample_type": "positive",
                "is_negative": False,
                "category": prompt_text or "similar_object",
                "reference_image": reference_image,
                "reference_bnd_points": reference_bnd_points,
                "paste_bnd_points": None,
                "prompt": prompt_text,
            }
        ]


    @app.get("/api-list.json")
    async def available_api_list_json() -> Dict[str, Any]:
        routes = get_available_api_routes()
        return {
            "success": True,
            "count": len(routes),
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
            "data": routes,
        }


    @app.get("/api-list", response_class=HTMLResponse)
    @app.get("/apis", response_class=HTMLResponse)
    async def available_api_list_page() -> HTMLResponse:
        routes = get_available_api_routes()
        rows = []
        for route in routes:
            methods = html.escape(", ".join(route["methods"]))
            path = html.escape(route["path"])
            auth = html.escape(route["auth"])
            summary = html.escape(route["summary"])
            rows.append(
                "<tr>"
                f"<td>{methods}</td>"
                f"<td><code>{path}</code></td>"
                f"<td>{auth}</td>"
                f"<td>{summary}</td>"
                "</tr>"
            )

        content = f"""
        <!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>SAM3 可用 API 列表</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
            th, td {{ border: 1px solid #ddd; text-align: left; padding: 8px; vertical-align: top; }}
            th {{ background: #f7f7f7; }}
            code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }}
          </style>
        </head>
        <body>
          <h2>SAM3 可用 API 列表</h2>
          <p>
            当前共 <strong>{len(routes)}</strong> 个接口。
            <a href="/docs">Swagger 文档</a> |
            <a href="/openapi.json">OpenAPI JSON</a> |
            <a href="/api-list.json">API 列表 JSON</a>
          </p>
          <table>
            <thead>
              <tr>
                <th>Methods</th>
                <th>Path</th>
                <th>Auth</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </body>
        </html>
        """
        return HTMLResponse(content)


    # 推理类接口统一遵循同一个并发模型：
    # 先用 INFERENCE_SEMAPHORE 控制进入后台线程池的请求数，再交给 runtime
    # 中的 pipeline。真正访问共享 predictor 时，runtime 内部还会用 MODEL_LOCK
    # 串行化模型状态，避免多个请求互相覆盖当前图片、特征或 prompt。
    @app.post("/v1/segmentations")
    async def create_segmentation(
        payload: SegmentationRequest,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        try:
            image = decode_base64_image(payload.image_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if SAVE_UPLOADS:
            save_upload_image(image, "base64_input.jpg")

        try:
            async with INFERENCE_SEMAPHORE:
                result = await run_inference_in_thread(
                    run_detection_pipeline,
                    image,
                    payload.prompt,
                    payload.confidence_threshold,
                    payload.polygon_simplify_epsilon,
                    payload.pic_id,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        return result


    @app.post("/v1/box-segmentations")
    async def create_box_segmentation(
        payload: BoxSegmentationRequest,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        try:
            image = decode_base64_image(payload.image_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if SAVE_UPLOADS:
            save_upload_image(image, "box_seg_input.jpg")

        try:
            async with INFERENCE_SEMAPHORE:
                result = await run_inference_in_thread(
                    run_box_segmentation_pipeline,
                    image,
                    payload.bnd_points,
                    payload.polygon_simplify_epsilon,
                    payload.pic_id,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Box segmentation failed: {exc}") from exc

        return result


    @app.post("/v1/similar-object-segmentations")
    async def create_similar_object_segmentation(
        payload: SimilarObjectRequest,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        try:
            reference_image = decode_base64_image(payload.reference_image_base64)
            query_images = [decode_base64_image(item) for item in payload.query_image_base64_list or []]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if SAVE_UPLOADS:
            save_upload_image(reference_image, "similar_reference_input.jpg")
            for index, query_image in enumerate(query_images, start=1):
                save_upload_image(query_image, f"similar_query_input_{index}.jpg")

        try:
            async with INFERENCE_SEMAPHORE:
                if payload.similar_mode == "same_image_prompt":
                    result = await run_inference_in_thread(
                        run_similar_object_batch_pipeline,
                        reference_image,
                        query_images,
                        payload.reference_bnd_points,
                        payload.top_k,
                        payload.similarity_threshold,
                        payload.sam_threshold,
                        payload.polygon_simplify_epsilon,
                        payload.pic_id,
                        None,
                        payload.similar_mode,
                        payload.prompt,
                    )
                else:
                    result = await run_inference_in_thread(
                        run_multi_similar_object_batch_pipeline,
                        build_single_positive_sample(reference_image, payload.reference_bnd_points, payload.prompt),
                        query_images,
                        payload.top_k,
                        payload.similarity_threshold,
                        payload.sam_threshold,
                        0.45,
                        payload.polygon_simplify_epsilon,
                        payload.pic_id,
                        None,
                    )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Similar object segmentation failed: {exc}") from exc

        return result


    @app.post("/v1/multi-similar-object-segmentations")
    async def create_multi_similar_object_segmentation(
        payload: MultiSimilarObjectRequest,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        try:
            samples: List[Dict[str, Any]] = []
            for sample_index, sample in enumerate(payload.samples, start=1):
                reference_image = decode_base64_image(sample.reference_image_base64)
                image_sample_id = sample.sample_id or f"image_{sample_index}"
                if sample.instances:
                    for instance_index, instance in enumerate(sample.instances, start=1):
                        samples.append(
                            {
                                "sample_id": instance.instance_id or f"{image_sample_id}_inst_{instance_index}",
                                "source_image_id": image_sample_id,
                                "sample_type": "negative" if instance.is_negative is True else instance.sample_type,
                                "is_negative": instance.is_negative is True or instance.sample_type == "negative",
                                "category": instance.category,
                                "reference_image": reference_image.copy(),
                                "reference_bnd_points": instance.reference_bnd_points,
                                "paste_bnd_points": instance.paste_bnd_points,
                                "prompt": instance.prompt,
                            }
                        )
                else:
                    samples.append(
                        {
                            "sample_id": image_sample_id,
                            "source_image_id": image_sample_id,
                            "sample_type": "negative" if sample.is_negative is True else sample.sample_type,
                            "is_negative": sample.is_negative is True or sample.sample_type == "negative",
                            "category": sample.category,
                            "reference_image": reference_image,
                            "reference_bnd_points": sample.reference_bnd_points,
                            "paste_bnd_points": sample.paste_bnd_points,
                            "prompt": sample.prompt,
                        }
                    )
            query_images = [decode_base64_image(item) for item in payload.query_image_base64_list or []]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if SAVE_UPLOADS:
            for index, sample in enumerate(samples, start=1):
                save_upload_image(sample["reference_image"], f"multi_similar_reference_input_{index}.jpg")
            for index, query_image in enumerate(query_images, start=1):
                save_upload_image(query_image, f"multi_similar_query_input_{index}.jpg")

        try:
            async with INFERENCE_SEMAPHORE:
                result = await run_inference_in_thread(
                    run_multi_similar_object_batch_pipeline,
                    samples,
                    query_images,
                    payload.top_k,
                    payload.similarity_threshold,
                    payload.sam_threshold,
                    payload.nms_iou,
                    payload.polygon_simplify_epsilon,
                    payload.pic_id,
                    None,
                    payload.prompt,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Multi similar object segmentation failed: {exc}") from exc

        return result


    @app.post("/v1/similar-object-segmentations/by-url")
    async def create_similar_object_segmentation_by_url(
        payload: SimilarObjectByUrlRequest,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        try:
            async with INFERENCE_SEMAPHORE:
                result = await run_inference_in_thread(
                    partial(
                        run_by_url_request,
                        download_url=payload.download_url,
                        sample_url=payload.sample_url,
                        prompt=payload.prompt,
                        query_image_url=payload.query_image_url,
                        pic_id=payload.pic_id,
                        top_k=payload.top_k,
                        similarity_threshold=payload.similarity_threshold,
                        sam_threshold=payload.sam_threshold,
                        nms_iou=payload.nms_iou,
                        polygon_simplify_epsilon=payload.polygon_simplify_epsilon,
                        return_result_image=payload.return_result_image,
                    ),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Similar object URL segmentation failed: {exc}") from exc

        return result


    @app.post("/v1/similar-object-segmentations/tasks")
    async def create_similar_object_segmentation_task(
        payload: SimilarObjectTaskCreateRequest,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        try:
            task = TASK_REGISTRY.create(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return task.snapshot()


    @app.get("/v1/similar-object-segmentations/tasks/{task_id}")
    async def get_similar_object_segmentation_task(
        task_id: str,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        task = TASK_REGISTRY.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return task.snapshot()


    @app.get("/v1/similar-object-segmentations/tasks/{task_id}/results")
    async def get_similar_object_segmentation_task_results(
        task_id: str,
        offset: int = 0,
        limit: int = 50,
        wait_timeout: float = 0.0,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        task = TASK_REGISTRY.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return task.wait_result_page(offset, limit, wait_timeout)


    @app.delete("/v1/similar-object-segmentations/tasks/{task_id}")
    async def cancel_similar_object_segmentation_task(
        task_id: str,
        _: Dict[str, Any] = Depends(require_api_key),
    ) -> Dict[str, Any]:
        cancelled = TASK_REGISTRY.cancel(task_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return {"success": True, "task_id": task_id, "status": "cancelled", "message": "cancel requested"}


    @app.post("/detect")
    async def detect_objects(
        file: UploadFile = File(...),
        prompt: Optional[str] = Form(default=None),
        confidence: float = Form(0.3),
        polygon_simplify_epsilon: float = Form(2.0),
        pic_id: Optional[str] = Form(default=None),
    ) -> Dict[str, Any]:
        try:
            contents = await file.read()
            if len(contents) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail=f"Image payload too large. Max bytes: {MAX_IMAGE_BYTES}")

            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid upload image: {exc}") from exc

        if SAVE_UPLOADS:
            save_upload_image(image, file.filename or "upload.jpg")

        try:
            async with INFERENCE_SEMAPHORE:
                result = await run_inference_in_thread(
                    run_detection_pipeline,
                    image,
                    prompt,
                    confidence,
                    polygon_simplify_epsilon,
                    pic_id,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        return result


    @app.post("/similar-detect")
    async def detect_similar_objects(
        request: Request,
        reference_file: UploadFile = File(...),
        reference_bnd_points: str = Form(...),
        prompt: Optional[str] = Form(default=None),
        top_k: int = Form(5),
        similarity_threshold: float = Form(0.6),
        sam_threshold: float = Form(0.6),
        polygon_simplify_epsilon: float = Form(2.0),
        similar_mode: str = Form("feature_match"),
        pic_id: Optional[str] = Form(default=None),
    ) -> Dict[str, Any]:
        try:
            reference_contents = await reference_file.read()
            if len(reference_contents) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail=f"Reference image too large. Max bytes: {MAX_IMAGE_BYTES}")
            is_same_image_mode = similar_mode == "same_image_prompt"
            form = await request.form()
            query_files = [
                item
                for item in form.getlist("query_file")
                if getattr(item, "filename", None) and hasattr(item, "read")
            ]
            if not query_files and not is_same_image_mode:
                raise HTTPException(status_code=400, detail="At least one query image is required")

            reference_image = Image.open(io.BytesIO(reference_contents)).convert("RGB")
            query_images: List[Image.Image] = []
            query_names: List[str] = []
            for index, one_file in enumerate(query_files, start=1):
                query_contents = await one_file.read()
                if len(query_contents) > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Query image #{index} too large. Max bytes: {MAX_IMAGE_BYTES}",
                    )
                query_images.append(Image.open(io.BytesIO(query_contents)).convert("RGB"))
                query_names.append(one_file.filename or f"query_upload_{index}.jpg")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid upload image: {exc}") from exc

        try:
            parsed_reference_bnd_points = parse_optional_bnd_points_text(reference_bnd_points)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if parsed_reference_bnd_points is None:
            raise HTTPException(status_code=400, detail="reference_bnd_points is required, format: x,y,w,h")

        prompt_text = (prompt or "").strip() or None

        if SAVE_UPLOADS:
            save_upload_image(reference_image, reference_file.filename or "reference_upload.jpg")
            for index, query_image in enumerate(query_images):
                query_name = query_names[index] if index < len(query_names) else f"query_upload_{index + 1}.jpg"
                save_upload_image(query_image, query_name)

        try:
            async with INFERENCE_SEMAPHORE:
                if is_same_image_mode:
                    result = await run_inference_in_thread(
                        run_similar_object_batch_pipeline,
                        reference_image,
                        query_images,
                        parsed_reference_bnd_points,
                        top_k,
                        similarity_threshold,
                        sam_threshold,
                        polygon_simplify_epsilon,
                        pic_id,
                        query_names,
                        similar_mode,
                        prompt_text,
                    )
                else:
                    result = await run_inference_in_thread(
                        run_multi_similar_object_batch_pipeline,
                        build_single_positive_sample(reference_image, parsed_reference_bnd_points, prompt_text),
                        query_images,
                        top_k,
                        similarity_threshold,
                        sam_threshold,
                        0.45,
                        polygon_simplify_epsilon,
                        pic_id,
                        query_names,
                    )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Similar object detection failed: {exc}") from exc

        return result


    @app.post("/multi-similar-detect")
    async def detect_multi_similar_objects(
        request: Request,
        sample_meta: str = Form("[]"),
        prompt: Optional[str] = Form(default=None),
        top_k: int = Form(5),
        similarity_threshold: float = Form(0.6),
        sam_threshold: float = Form(0.6),
        nms_iou: float = Form(0.45),
        polygon_simplify_epsilon: float = Form(2.0),
        pic_id: Optional[str] = Form(default=None),
    ) -> Dict[str, Any]:
        try:
            form = await request.form()
            parsed_meta = json.loads(sample_meta)
            if not isinstance(parsed_meta, list):
                raise HTTPException(status_code=400, detail="sample_meta must be a JSON array")

            sample_files = [
                item
                for item in form.getlist("sample_file")
                if getattr(item, "filename", None) and hasattr(item, "read")
            ]
            query_files = [
                item
                for item in form.getlist("query_file")
                if getattr(item, "filename", None) and hasattr(item, "read")
            ]
            if parsed_meta and not sample_files:
                raise HTTPException(status_code=400, detail="At least one sample image is required")
            if not query_files:
                raise HTTPException(status_code=400, detail="At least one query image is required")

            sample_images: List[Image.Image] = []
            for index, one_file in enumerate(sample_files, start=1):
                sample_contents = await one_file.read()
                if len(sample_contents) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=400, detail=f"Sample image #{index} too large. Max bytes: {MAX_IMAGE_BYTES}")
                sample_images.append(Image.open(io.BytesIO(sample_contents)).convert("RGB"))

            samples: List[Dict[str, Any]] = []
            for index, one_meta in enumerate(parsed_meta, start=1):
                if not isinstance(one_meta, dict):
                    raise HTTPException(status_code=400, detail=f"sample_meta[{index - 1}] must be an object")
                raw_file_index = one_meta.get("file_index", index - 1)
                try:
                    file_index = int(raw_file_index)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"sample_meta[{index - 1}].file_index must be an integer") from exc
                if file_index < 0 or file_index >= len(sample_images):
                    raise HTTPException(status_code=400, detail=f"sample_meta[{index - 1}].file_index is out of range")

                raw_bnd_points = one_meta.get("reference_bnd_points")
                if isinstance(raw_bnd_points, str):
                    parsed_bnd_points = parse_optional_bnd_points_text(raw_bnd_points)
                elif isinstance(raw_bnd_points, list):
                    parsed_bnd_points = [float(v) for v in raw_bnd_points]
                else:
                    parsed_bnd_points = None
                if parsed_bnd_points is None or len(parsed_bnd_points) != 4:
                    raise HTTPException(
                        status_code=400,
                        detail=f"sample_meta[{index - 1}].reference_bnd_points is required, format: x,y,w,h",
                    )
                raw_paste_bnd_points = one_meta.get("paste_bnd_points")
                if isinstance(raw_paste_bnd_points, str):
                    parsed_paste_bnd_points = parse_optional_bnd_points_text(raw_paste_bnd_points)
                elif isinstance(raw_paste_bnd_points, list):
                    parsed_paste_bnd_points = [float(v) for v in raw_paste_bnd_points]
                else:
                    parsed_paste_bnd_points = None

                samples.append(
                    {
                        "sample_id": one_meta.get("sample_id") or f"sample_{index}",
                        "source_image_id": one_meta.get("source_image_id") or f"image_{file_index + 1}",
                        "source_file_index": file_index,
                        "sample_type": one_meta.get("sample_type") or one_meta.get("role") or "positive",
                        "is_negative": one_meta.get("is_negative") is True,
                        "category": one_meta.get("category"),
                        "reference_image": sample_images[file_index].copy(),
                        "reference_bnd_points": parsed_bnd_points,
                        "paste_bnd_points": parsed_paste_bnd_points,
                        "prompt": one_meta.get("prompt"),
                    }
                )

            query_images: List[Image.Image] = []
            query_names: List[str] = []
            for index, one_file in enumerate(query_files, start=1):
                query_contents = await one_file.read()
                if len(query_contents) > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Query image #{index} too large. Max bytes: {MAX_IMAGE_BYTES}",
                    )
                query_images.append(Image.open(io.BytesIO(query_contents)).convert("RGB"))
                query_names.append(one_file.filename or f"query_upload_{index}.jpg")
        except HTTPException:
            raise
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid sample_meta JSON: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid multi-similar upload payload: {exc}") from exc

        if SAVE_UPLOADS:
            saved_source_ids = set()
            for sample in samples:
                source_image_id = sample.get("source_image_id") or sample.get("sample_id") or "sample"
                source_key = sample.get("source_file_index")
                dedupe_key = ("file", source_key) if source_key is not None else ("source", source_image_id)
                if dedupe_key in saved_source_ids:
                    continue
                saved_source_ids.add(dedupe_key)
                save_upload_image(sample["reference_image"], f"multi_reference_upload_{source_image_id}.jpg")
            for index, query_image in enumerate(query_images):
                query_name = query_names[index] if index < len(query_names) else f"query_upload_{index + 1}.jpg"
                save_upload_image(query_image, query_name)

        try:
            async with INFERENCE_SEMAPHORE:
                result = await run_inference_in_thread(
                    run_multi_similar_object_batch_pipeline,
                    samples,
                    query_images,
                    top_k,
                    similarity_threshold,
                    sam_threshold,
                    nms_iou,
                    polygon_simplify_epsilon,
                    pic_id,
                    query_names,
                    (prompt or "").strip() or None,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Multi similar object detection failed: {exc}") from exc

        return result


    @app.post("/ui/api-keys")
    async def ui_create_api_key(payload: CreateApiKeyRequest) -> Dict[str, Any]:
        created = api_key_manager.create_key(
            name=payload.name,
            role=payload.role,
            expires_in_days=payload.expires_in_days,
        )
        return {
            "success": True,
            "message": "API key created. Save api_key now; it will not be shown again.",
            "data": created,
        }


    @app.get("/ui/api-keys")
    async def ui_list_api_keys() -> Dict[str, Any]:
        return {
            "success": True,
            "data": api_key_manager.list_keys(),
        }


    @app.delete("/ui/api-keys/{key_id}")
    async def ui_delete_api_key(key_id: str) -> Dict[str, Any]:
        try:
            deleted = api_key_manager.delete_key(key_id, protect_last_admin=False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found")

        return {
            "success": True,
            "message": f"API key {key_id} deleted",
        }


    @app.post("/v1/api-keys")
    async def create_api_key(
        payload: CreateApiKeyRequest,
        _: Dict[str, Any] = Depends(require_admin_key),
    ) -> Dict[str, Any]:
        created = api_key_manager.create_key(
            name=payload.name,
            role=payload.role,
            expires_in_days=payload.expires_in_days,
        )
        return {
            "success": True,
            "message": "API key created. Save api_key now; it will not be shown again.",
            "data": created,
        }


    @app.get("/v1/api-keys")
    async def list_api_keys(_: Dict[str, Any] = Depends(require_admin_key)) -> Dict[str, Any]:
        return {
            "success": True,
            "data": api_key_manager.list_keys(),
        }


    @app.delete("/v1/api-keys/{key_id}")
    async def delete_api_key(
        key_id: str,
        _: Dict[str, Any] = Depends(require_admin_key),
    ) -> Dict[str, Any]:
        try:
            deleted = api_key_manager.delete_key(key_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found")

        return {
            "success": True,
            "message": f"API key {key_id} deleted",
        }


    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))

        return HTMLResponse("<h3>SAM3 API is running</h3><p>Use /docs for OpenAPI documentation.</p>")


    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        with INFERENCE_STATE_LOCK:
            active_inferences = ACTIVE_INFERENCE_COUNT
            last_finished_at = LAST_INFERENCE_FINISHED_AT
            idle_model_unloaded = IDLE_MODEL_UNLOADED

        idle_seconds = None
        if last_finished_at > 0 and active_inferences == 0:
            idle_seconds = int(time.monotonic() - last_finished_at)

        return {
            "status": "healthy",
            "model": MODEL_LABEL,
            "device": device,
            "model_loaded": getattr(predictor, "model", None) is not None,
            "active_inferences": active_inferences,
            "idle_seconds": idle_seconds,
            "idle_model_unloaded": idle_model_unloaded,
            "max_concurrent_inferences": MAX_CONCURRENT_INFERENCES,
            "model_access_mode": "serialized" if SERIALIZE_MODEL_ACCESS else "parallel",
            "backend": "ultralytics",
            "ultralytics_imgsz": ULTRALYTICS_IMGSZ,
            "cuda_cleanup_after_request": CUDA_CLEANUP_AFTER_REQUEST,
            "idle_model_unload_seconds": IDLE_MODEL_UNLOAD_SECONDS,
            "cuda_memory": get_cuda_memory_stats(),
        }
