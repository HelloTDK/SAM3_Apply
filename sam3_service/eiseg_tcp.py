"""EISeg-compatible TCP transport backed by SAM3's native interactive engines."""

import base64
import io
import json
import logging
import math
import os
import socketserver
import warnings
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image

from .config import MAX_IMAGE_BYTES, TCP_POLYGON_EPSILON


LOGGER = logging.getLogger(__name__)
DELIMITER = b"##@##"
# Base64 expands a binary image by 4/3; leave room for the JSON envelope.
MAX_FRAME_BYTES = (MAX_IMAGE_BYTES + 2) // 3 * 4 + 65536


def load_image(data):
    """Read legacy URL/base64 input without an image-id-based disk cache."""
    image_url = data.get("image_url")
    if image_url:
        if not isinstance(image_url, str):
            raise ValueError("image_url must be a string")
        if not urlsplit(image_url).scheme:
            image_url = urljoin(
                os.getenv("SAM3_TCP_IMAGE_BASE_URL", "http://192.168.100.25:8092/"), image_url
            )
        if urlsplit(image_url).scheme not in {"http", "https"}:
            raise ValueError("image_url must use HTTP or HTTPS")
        with urlopen(image_url, timeout=30) as response:
            content = response.read(MAX_IMAGE_BYTES + 1)
    else:
        encoded = data.get("image_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("image_url or image_base64 is required")
        content = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image payload too large. Max bytes: {MAX_IMAGE_BYTES}")
    # Treat Pillow's decompression warning as an error before allocating RGB pixels.
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(content)) as image:
            return image.convert("RGB")


def mask_to_polygons(mask, epsilon=TCP_POLYGON_EPSILON):
    """Simplify contours, then bridge holes into the legacy flat polygon format."""
    contours, hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if hierarchy is None:
        return []

    # Simplify each contour before adding the artificial hole-to-outer bridge.
    def simplify(contour):
        points = contour.reshape(-1, 2)
        if epsilon <= 0:
            return points
        reduced = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        return reduced if len(reduced) >= 3 else points

    polygons = []
    for index, relation in enumerate(hierarchy[0]):
        if relation[3] != -1:
            continue
        outer = simplify(contours[index])
        if len(outer) < 3:
            continue
        child = relation[2]
        while child != -1:
            hole = simplify(contours[child])
            # Search one hole vertex at a time to avoid a large pairwise-distance array.
            best = (float("inf"), 0, 0)
            for hole_index, point in enumerate(hole):
                distances = np.sum((outer.astype(np.float64) - point) ** 2, axis=1)
                outer_index = int(np.argmin(distances))
                if distances[outer_index] < best[0]:
                    best = (distances[outer_index], outer_index, hole_index)
            _, outer_index, hole_index = best
            hole = np.concatenate((hole[hole_index:], hole[: hole_index + 1]))
            outer = np.concatenate((outer[: outer_index + 1], hole, outer[outer_index:]))
            child = hierarchy[0][child][0]
        polygons.append(outer.tolist())
    return polygons


def box_info(polygon, image_size):
    """Keep the original EISeg rotated-box convention, including integer corners."""
    contour = np.asarray(polygon, dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
    for axis, limit in enumerate(image_size):
        lower, upper = box[:, axis].min(), box[:, axis].max()
        offset = -lower if lower < 0 else limit - upper if upper > limit else 0
        box[:, axis] += offset
    (cx, cy), (width, height), angle = cv2.minAreaRect(box.astype(np.float32))
    return (
        [int(cx - width / 2), int(cy - height / 2), int(width), int(height), round(angle, 3)],
        list(cv2.boundingRect(contour)),
    )


class SegmentationSession:
    """One connection owns its image features and successful click history."""

    def __init__(self, backend):
        self.backend = backend
        self.clear_image()

    def clear_image(self):
        self.image_id = None
        self.image_size = None
        self.features = None
        self.clear_clicks()

    def clear_clicks(self):
        self.clicks = []
        self.history = []

    def dispatch(self, request):
        cmd, data = None, {}
        try:
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            cmd = request.get("cmd")
            if type(cmd) is not int:
                raise ValueError("cmd must be an integer")
            data = request.get("data", {})
            if not isinstance(data, dict):
                data = {}
                raise ValueError("data must be an object")
            if cmd == 8000:
                model = data.get("model", 0)
                if type(model) is not int or model not in (0, 1):
                    raise ValueError("model must be 0 or 1")
                # Both legacy model slots share SAM3; use_gpu=false remains accepted for old clients.
                self.clear_image()
                result = {"status": 1, "message": "Model initialized successfully!"}
            elif cmd == 8001:
                # A failed image change must not leave the previous image available for clicks.
                self.clear_image()
                image_id = data.get("image_id")
                if not isinstance(image_id, str) or not image_id:
                    raise ValueError("image_id must be a non-empty string")
                image = load_image(data)
                features = self.backend.encode_image(image)
                self.image_id, self.image_size, self.features = image_id, image.size, features
                result = {"image_id": image_id, "status": 1, "message": "image init success!"}
            elif cmd == 8002:
                result = self.click(data)
            elif cmd == 8004:
                result = [
                    {"model": index, "model_name": name, "use_gpu": 1}
                    for index, name in enumerate(("Normal model", "Person model"))
                ]
            else:
                raise ValueError(f"Unsupported cmd: {cmd}")
        except Exception as exc:
            LOGGER.warning("TCP command %s failed: %s", cmd, exc)
            result = {"status": 0, "message": str(exc)}
            if cmd in (8001, 8002):
                result["image_id"] = data.get("image_id")
            if cmd == 8002:
                result.update(click_type=data.get("click_type"), ploygon=[], bnd_boxs=[], min_boxs=[])
        return {"cmd": cmd + 1000 if type(cmd) is int else 0, "data": result}

    def click(self, data):
        if self.features is None:
            raise ValueError("Image is not initialized")
        if data.get("image_id", self.image_id) != self.image_id:
            raise ValueError("image_id does not match the initialized image")
        click_type = data.get("click_type")
        if type(click_type) is not int or click_type not in (1, 2, 3, 4, 5):
            raise ValueError("click_type must be 1, 2, 3, 4 or 5")
        if click_type in (1, 2):
            if click_type == 2 and not self.clicks:
                raise ValueError("Forbid first click is false")
            coordinate = data.get("coordinate")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ValueError("coordinate must be [x, y]")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in coordinate):
                raise ValueError("coordinate must contain two finite numbers")
            x, y = coordinate
            width, height = self.image_size
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("点击越界")
            if len(self.clicks) >= self.backend.max_clicks:
                raise ValueError(f"Current TensorRT engines support at most {self.backend.max_clicks} clicks per object")
            clicks = self.clicks + [(x, y, 1 if click_type == 1 else 0)]
            previous_logits = self.history[-1]["logits"] if self.history else None
            mask, _, logits = self.backend.segment(
                self.features, self.image_size, clicks, previous_logits
            )
            polygons = mask_to_polygons(mask)
            # Commit only after inference succeeds, so retries and undo remain consistent.
            self.clicks = clicks
            self.history.append({"polygons": polygons, "logits": logits})
        elif click_type == 3 and self.clicks:
            self.clicks.pop()
            self.history.pop()
        polygons = self.history[-1]["polygons"] if self.history else []
        if click_type == 5:
            polygons = []
        min_boxes, bnd_boxes = [], []
        for polygon in polygons:
            min_box, bnd_box = box_info(polygon, self.image_size)
            min_boxes.append(min_box)
            bnd_boxes.append(bnd_box)
        result = {
            "status": 1, "image_id": data.get("image_id", self.image_id), "click_type": click_type,
            "ploygon": polygons, "bnd_boxs": bnd_boxes, "min_boxs": min_boxes,
            "message": "Click processed successfully",
        }
        # Finish returns the last preview before starting the next object; reset returns empty geometry.
        if click_type in (4, 5):
            self.clear_clicks()
        return result


class SegmentationRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        session = SegmentationSession(self.server.backend)
        self.request.settimeout(600)
        buffer = b""
        try:
            while True:
                chunk = self.request.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                while DELIMITER in buffer:
                    message, buffer = buffer.split(DELIMITER, 1)
                    if len(message) > MAX_FRAME_BYTES:
                        self.send_error("TCP frame too large")
                        return
                    try:
                        request = json.loads(message.decode("utf-8"))
                    except (ValueError, UnicodeError) as exc:
                        self.send_error(f"Invalid JSON: {exc}")
                        continue
                    response = session.dispatch(request)
                    self.request.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8") + DELIMITER)
                if len(buffer) > MAX_FRAME_BYTES:
                    self.send_error("TCP frame too large")
                    return
        except (OSError, TimeoutError) as exc:
            LOGGER.info("TCP client %s disconnected: %s", self.client_address, exc)
        finally:
            # Drop per-client GPU feature references when the connection closes or times out.
            session.clear_image()

    def send_error(self, message):
        response = {"cmd": 0, "data": {"status": 0, "message": message}}
        self.request.sendall(json.dumps(response).encode("utf-8") + DELIMITER)


class InteractiveSegmentationServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 10

    def __init__(self, server_address=("0.0.0.0", 6063), backend=None):
        # Bind before loading engines so port conflicts fail without allocating GPU memory.
        super().__init__(server_address, SegmentationRequestHandler)
        try:
            if backend is None:
                # Standalone TCP uses the same native engines as the combined Gradio app.
                from .native_click_trt import NativeInteractiveBackend

                default_dir = Path(__file__).resolve().parents[1] / "engines" / "native_interactive"
                engine_dir = Path(os.getenv("SAM3_NATIVE_ENGINE_DIR", str(default_dir)))
                backend = NativeInteractiveBackend(engine_dir)
            self.backend = backend
        except Exception:
            self.server_close()
            raise

    def start_server(self):
        LOGGER.info("Server start, listening on %s", self.server_address)
        self.serve_forever()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    address = (os.getenv("SAM3_TCP_HOST", "0.0.0.0"), int(os.getenv("SAM3_TCP_PORT", "6063")))
    with InteractiveSegmentationServer(address) as server:
        try:
            server.start_server()
        except KeyboardInterrupt:
            LOGGER.info("Server socket is closed")
