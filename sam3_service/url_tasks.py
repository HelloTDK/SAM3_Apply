"""URL/manifest based similar-object task helpers for SAM3 service."""

import hashlib
import io
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from PIL import Image

from .config import MAX_IMAGE_BYTES
from .runtime import (
    prepare_multi_visual_prompt_state,
    run_multi_visual_prompt_query_with_state,
)


SAMPLE_IMAGE_LIMIT = 300
SAMPLE_INSTANCE_LIMIT = 2000
QUERY_IMAGE_LIMIT = 5000
SAMPLE_STATE_CACHE_TTL_SECONDS = 3600
SAMPLE_STATE_CACHE_MAX_ITEMS = 8


def resolve_remote_url(download_url: str, remote_path: str) -> str:
    remote = str(remote_path or "").strip().replace("\\", "")
    if not remote:
        raise ValueError("remote path is empty")
    if remote.startswith("http://") or remote.startswith("https://"):
        return remote
    base = str(download_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("download_url is required for relative remote paths")
    return f"{base}/{remote.lstrip('/')}"


def download_bytes(download_url: str, remote_path: str, *, timeout: int = 60) -> bytes:
    url = resolve_remote_url(download_url, remote_path)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_text(download_url: str, remote_path: str, *, timeout: int = 60) -> str:
    content = download_bytes(download_url, remote_path, timeout=timeout)
    return content.decode("utf-8", errors="ignore")


def download_image(download_url: str, remote_path: str, *, timeout: int = 60) -> Image.Image:
    content = download_bytes(download_url, remote_path, timeout=timeout)
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image payload too large. Max bytes: {MAX_IMAGE_BYTES}")
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image: {remote_path}") from exc


def _stable_id_from_path(remote_path: str) -> str:
    parsed = urlparse(str(remote_path or ""))
    name = (parsed.path.rsplit("/", 1)[-1] or "image").rsplit(".", 1)[0]
    if name:
        return name
    return hashlib.md5(str(remote_path).encode("utf-8")).hexdigest()[:16]


def _load_json_from_text(text: str, source_name: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_name} is not valid JSON") from exc


def _parse_mark_info(mark_info: Any, field_name: str) -> List[float]:
    if isinstance(mark_info, str):
        mark_info = _load_json_from_text(mark_info, field_name)
    if isinstance(mark_info, (list, tuple)):
        if len(mark_info) != 4:
            raise ValueError(f"{field_name} array must contain [x, y, width, height]")
        try:
            x = float(mark_info[0])
            y = float(mark_info[1])
            w = float(mark_info[2])
            h = float(mark_info[3])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} array must contain numeric x/y/width/height") from exc
        if w <= 0 or h <= 0:
            raise ValueError(f"{field_name} width/height must be > 0")
        return [x, y, w, h]
    if not isinstance(mark_info, dict):
        raise ValueError(f"{field_name} must be a JSON object/object string or [x, y, width, height] array")
    try:
        x = float(mark_info["x"])
        y = float(mark_info["y"])
        w = float(mark_info["width"])
        h = float(mark_info["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain x/y/width/height") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"{field_name} width/height must be > 0")
    return [x, y, w, h]


def parse_sample_manifest_text(text: str, download_url: str) -> List[Dict[str, Any]]:
    raw_manifest = _load_json_from_text(text, "sample_url")
    if not isinstance(raw_manifest, list):
        raise ValueError("sample_url content must be a JSON array")

    samples: List[Dict[str, Any]] = []
    image_url_to_index: Dict[str, int] = {}
    image_count = 0
    instance_count = 0

    for label_index, label_group in enumerate(raw_manifest):
        if not isinstance(label_group, dict):
            continue
        category = str(label_group.get("label_id", "") or "").strip()
        if not category:
            raise ValueError(f"sample_url[{label_index}].label_id is required")
        label_sample_data = label_group.get("label_sample_data") or []
        if not isinstance(label_sample_data, list):
            raise ValueError(f"sample_url[{label_index}].label_sample_data must be array")

        for image_index, image_item in enumerate(label_sample_data):
            if not isinstance(image_item, dict):
                continue
            image_url = str(image_item.get("image_url", "") or "").strip()
            if not image_url:
                raise ValueError(f"sample_url[{label_index}].label_sample_data[{image_index}].image_url is required")

            if image_url not in image_url_to_index:
                image_url_to_index[image_url] = image_count
                image_count += 1
            if image_count > SAMPLE_IMAGE_LIMIT:
                raise ValueError(f"sample images exceed limit: {SAMPLE_IMAGE_LIMIT}")

            image_id = str(image_item.get("image_id", "") or "").strip() or _stable_id_from_path(image_url)
            image_mark = image_item.get("image_mark") or []
            if not isinstance(image_mark, list):
                raise ValueError(f"sample image_mark must be array: image_url={image_url}")

            for mark_index, mark in enumerate(image_mark):
                if not isinstance(mark, dict):
                    continue
                box = _parse_mark_info(
                    mark.get("mark_info"),
                    f"sample_url[{label_index}].label_sample_data[{image_index}].image_mark[{mark_index}].mark_info",
                )
                raw_sample_type_value = mark.get("sample_type", "1")
                if raw_sample_type_value is None:
                    raw_sample_type_value = "1"
                raw_sample_type = str(raw_sample_type_value).strip().lower()
                sample_type = "negative" if raw_sample_type in {"0", "negative", "neg", "false"} else "positive"
                instance_count += 1
                if instance_count > SAMPLE_INSTANCE_LIMIT:
                    raise ValueError(f"sample instances exceed limit: {SAMPLE_INSTANCE_LIMIT}")
                samples.append(
                    {
                        "sample_id": f"{category}_{instance_count:04d}",
                        "source_image_id": image_id,
                        "source_file_index": image_url_to_index[image_url],
                        "category": category,
                        "sample_type": sample_type,
                        "reference_image_url": image_url,
                        "reference_bnd_points": box,
                        "prompt": None,
                    }
                )

    if not samples:
        raise ValueError("sample_url contains no valid sample marks")
    if not any(item["sample_type"] != "negative" for item in samples):
        raise ValueError("At least one positive sample is required")
    return samples


def load_sample_manifest(download_url: str, sample_url: str) -> tuple[List[Dict[str, Any]], str]:
    text = download_text(download_url, sample_url, timeout=60)
    return parse_sample_manifest_text(text, download_url), hashlib.sha256(text.encode("utf-8")).hexdigest()


def attach_sample_images(download_url: str, parsed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    image_cache: Dict[str, Image.Image] = {}
    samples_with_images: List[Dict[str, Any]] = []
    for sample in parsed_samples:
        sample_with_image = dict(sample)
        image_url = sample["reference_image_url"]
        image = image_cache.get(image_url)
        if image is None:
            image = download_image(download_url, image_url, timeout=60)
            image_cache[image_url] = image
        sample_with_image["reference_image"] = image.copy()
        samples_with_images.append(sample_with_image)
    return samples_with_images


def load_samples_from_manifest(download_url: str, sample_url: str) -> tuple[List[Dict[str, Any]], str]:
    parsed_samples, manifest_hash = load_sample_manifest(download_url, sample_url)
    return attach_sample_images(download_url, parsed_samples), manifest_hash


def parse_query_manifest_text(text: str) -> List[Dict[str, str]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("data_url content is empty")

    items: List[Dict[str, str]] = []
    if stripped.startswith("["):
        raw_items = _load_json_from_text(stripped, "data_url")
        if not isinstance(raw_items, list):
            raise ValueError("data_url JSON content must be an array")
        for index, item in enumerate(raw_items):
            if isinstance(item, str):
                image_url = item.strip()
                image_id = _stable_id_from_path(image_url)
            elif isinstance(item, dict):
                image_url = str(item.get("image_url") or item.get("url") or item.get("path") or "").strip()
                image_id = str(item.get("image_id") or item.get("pic_id") or "").strip() or _stable_id_from_path(image_url)
            else:
                continue
            if not image_url:
                raise ValueError(f"data_url[{index}].image_url is required")
            items.append({"image_id": image_id, "image_url": image_url})
    else:
        for line_index, line in enumerate(stripped.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if "=" in line:
                image_id, image_url = line.split("=", 1)
                image_id = image_id.strip() or f"image_{line_index}"
                image_url = image_url.strip()
            else:
                image_url = line
                image_id = _stable_id_from_path(image_url) or f"image_{line_index}"
            if not image_url:
                continue
            items.append({"image_id": image_id, "image_url": image_url})

    if not items:
        raise ValueError("data_url contains no query images")
    if len(items) > QUERY_IMAGE_LIMIT:
        raise ValueError(f"query images exceed limit: {QUERY_IMAGE_LIMIT}")
    return items


def load_query_items(download_url: str, data_url: str) -> List[Dict[str, str]]:
    return parse_query_manifest_text(download_text(download_url, data_url, timeout=60))


class SampleStateCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            if now - float(item.get("created", 0.0)) > SAMPLE_STATE_CACHE_TTL_SECONDS:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return item["value"]

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = {"created": time.time(), "value": value}
            self._items.move_to_end(key)
            while len(self._items) > SAMPLE_STATE_CACHE_MAX_ITEMS:
                self._items.popitem(last=False)


SAMPLE_STATE_CACHE = SampleStateCache()


def build_sample_state(
    *,
    download_url: str,
    sample_url: str,
    top_k: int,
    manifest_hash: Optional[str] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if manifest_hash is None:
        parsed_samples, manifest_hash = load_sample_manifest(download_url, sample_url)
    else:
        parsed_samples = samples or []
    cache_key = hashlib.sha256(
        f"{download_url}|{sample_url}|{manifest_hash}|{top_k}".encode("utf-8")
    ).hexdigest()
    cached = SAMPLE_STATE_CACHE.get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached
    if samples is None or not all("reference_image" in sample for sample in samples):
        samples = attach_sample_images(download_url, parsed_samples)
    prompt_state = prepare_multi_visual_prompt_state(samples, top_k)
    prompt_state["cache_hit"] = False
    prompt_state["cache_key"] = cache_key
    SAMPLE_STATE_CACHE.set(cache_key, prompt_state)
    return prompt_state


def run_by_url_request(
    *,
    download_url: str,
    sample_url: str,
    query_image_url: str,
    pic_id: str,
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    nms_iou: float,
    polygon_simplify_epsilon: float,
    return_result_image: bool,
) -> Dict[str, Any]:
    parsed_samples, manifest_hash = load_sample_manifest(download_url, sample_url)
    sample_state = build_sample_state(
        download_url=download_url,
        sample_url=sample_url,
        top_k=top_k,
        manifest_hash=manifest_hash,
        samples=parsed_samples,
    )
    query_image = download_image(download_url, query_image_url, timeout=60)
    return run_multi_visual_prompt_query_with_state(
        sample_state,
        query_image,
        top_k,
        similarity_threshold,
        sam_threshold,
        nms_iou,
        polygon_simplify_epsilon,
        pic_id,
        return_result_image=return_result_image,
    )


class SimilarObjectTask:
    def __init__(self, request: Any):
        self.task_id = request.task_id
        self.request = request
        self.status = "pending"
        self.message = "pending"
        self.created = int(time.time())
        self.updated = self.created
        self.total = 0
        self.processed = 0
        self.success_count = 0
        self.fail_count = 0
        self.results: List[Dict[str, Any]] = []
        self.result_ttl_seconds = int(getattr(request, "result_ttl_seconds", 86400) or 86400)
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "success": True,
                "task_id": self.task_id,
                "status": self.status,
                "total": self.total,
                "processed": self.processed,
                "success_count": self.success_count,
                "fail_count": self.fail_count,
                "message": self.message,
                "created": self.created,
                "updated": self.updated,
            }

    def result_page(self, offset: int, limit: int) -> Dict[str, Any]:
        with self.lock:
            safe_offset = max(0, int(offset))
            safe_limit = max(1, min(int(limit), 500))
            return {
                "success": True,
                "task_id": self.task_id,
                "status": self.status,
                "processed": self.processed,
                "success_count": self.success_count,
                "fail_count": self.fail_count,
                "message": self.message,
                "offset": safe_offset,
                "limit": safe_limit,
                "total": self.total,
                "result_total": len(self.results),
                "items": self.results[safe_offset : safe_offset + safe_limit],
            }

    def wait_result_page(self, offset: int, limit: int, wait_timeout: float = 0.0) -> Dict[str, Any]:
        safe_offset = max(0, int(offset))
        safe_limit = max(1, min(int(limit), 500))
        timeout_seconds = max(0.0, min(float(wait_timeout or 0.0), 60.0))
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            while (
                len(self.results) <= safe_offset
                and self.status in {"pending", "running"}
                and timeout_seconds > 0
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
            return {
                "success": True,
                "task_id": self.task_id,
                "status": self.status,
                "processed": self.processed,
                "success_count": self.success_count,
                "fail_count": self.fail_count,
                "message": self.message,
                "offset": safe_offset,
                "limit": safe_limit,
                "total": self.total,
                "result_total": len(self.results),
                "items": self.results[safe_offset : safe_offset + safe_limit],
            }

    def update(self, **kwargs: Any) -> None:
        with self.condition:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.updated = int(time.time())
            self.condition.notify_all()

    def append_result(self, item: Dict[str, Any], success: bool) -> None:
        with self.condition:
            self.results.append(item)
            self.processed += 1
            if success:
                self.success_count += 1
            else:
                self.fail_count += 1
            self.updated = int(time.time())
            self.condition.notify_all()


class SimilarObjectTaskRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, SimilarObjectTask] = {}

    def create(self, request: Any) -> SimilarObjectTask:
        with self._lock:
            self._cleanup_expired_locked()
            existing = self._tasks.get(request.task_id)
            if existing and existing.snapshot()["status"] in {"pending", "running"}:
                raise ValueError(f"Task already exists: {request.task_id}")
            task = SimilarObjectTask(request)
            self._tasks[request.task_id] = task
        thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
        thread.start()
        return task

    def get(self, task_id: str) -> Optional[SimilarObjectTask]:
        with self._lock:
            self._cleanup_expired_locked()
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        task = self.get(task_id)
        if not task:
            return False
        task.cancel_event.set()
        task.update(status="cancelled", message="cancel requested")
        return True

    def _cleanup_expired_locked(self) -> None:
        now = int(time.time())
        for task_id, task in list(self._tasks.items()):
            snapshot = task.snapshot()
            if snapshot["status"] in {"pending", "running"}:
                continue
            ttl = max(60, int(getattr(task, "result_ttl_seconds", 86400)))
            if now - int(snapshot.get("updated", snapshot.get("created", now))) > ttl:
                self._tasks.pop(task_id, None)

    def _run_task(self, task: SimilarObjectTask) -> None:
        request = task.request
        try:
            if int(request.data_type) != 0:
                raise ValueError("URL sample task currently supports data_type=0 only")
            task.update(status="running", message="loading manifests")
            parsed_samples, manifest_hash = load_sample_manifest(request.download_url, request.sample_url)
            query_items = load_query_items(request.download_url, request.data_url)
            task.update(total=len(query_items), message="preparing sample prompts")
            sample_state = build_sample_state(
                download_url=request.download_url,
                sample_url=request.sample_url,
                top_k=request.top_k,
                manifest_hash=manifest_hash,
                samples=parsed_samples,
            )
            task.update(message="running")
            for query_item in query_items:
                if task.cancel_event.is_set():
                    task.update(status="cancelled", message="cancelled")
                    return
                pic_id = query_item["image_id"]
                try:
                    query_image = download_image(request.download_url, query_item["image_url"], timeout=60)
                    result = run_multi_visual_prompt_query_with_state(
                        sample_state,
                        query_image,
                        request.top_k,
                        request.similarity_threshold,
                        request.sam_threshold,
                        request.nms_iou,
                        request.polygon_simplify_epsilon,
                        pic_id,
                        return_result_image=request.return_result_image,
                    )
                    task.append_result(
                        {
                            "pic_id": pic_id,
                            "status": 1,
                            "message": "标注成功",
                            "pic_labels": result.get("pic_labels", []),
                        },
                        success=True,
                    )
                except Exception as exc:
                    task.append_result(
                        {
                            "pic_id": pic_id,
                            "status": 0,
                            "message": str(exc),
                            "pic_labels": [],
                        },
                        success=False,
                    )
            task.update(status="completed", message="completed")
        except Exception as exc:
            task.update(status="failed", message=str(exc))


TASK_REGISTRY = SimilarObjectTaskRegistry()
