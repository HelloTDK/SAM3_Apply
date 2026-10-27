"""Protocol and interaction checks that do not require GPU engines or Paddle."""

import base64
import io
import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from sam3_service.eiseg_tcp import (
    DELIMITER,
    InteractiveSegmentationServer,
    SegmentationSession,
    mask_to_polygons,
)


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(output, format="PNG")
    return output.getvalue()


def image_request(image_id="test"):
    return {"cmd": 8001, "data": {
        "image_id": image_id, "image_url": "",
        "image_base64": "data:image/png;base64," + base64.b64encode(image_bytes()).decode(),
    }}


def click_request(click_type, coordinate=None, image_id="test"):
    return {"cmd": 8002, "data": {
        "image_id": image_id, "click_type": click_type, "coordinate": coordinate,
    }}


class FakeBackend:
    max_clicks = 3

    def __init__(self):
        self.previous_logits = []

    def encode_image(self, image):
        return np.asarray(image)

    def segment(self, features, image_size, clicks, previous_logits=None):
        # Distinct previews let undo/finish tests detect stale or shared session state.
        self.previous_logits.append(previous_logits)
        edge = 10 + len(clicks)
        mask = np.zeros((image_size[1], image_size[0]), dtype=bool)
        mask[2:edge + 1, 2:edge + 1] = True
        return mask, 0.9, np.full((1, 1, 288, 288), len(clicks), dtype=np.float32)


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.session = SegmentationSession(FakeBackend())
        self.assertEqual(self.session.dispatch(image_request())["data"]["status"], 1)

    def click(self, click_type, coordinate=None):
        return self.session.dispatch(click_request(click_type, coordinate))["data"]

    def test_positive_negative_undo_finish_reset_and_next_object(self):
        self.assertEqual(self.click(2, [20, 20])["status"], 0)
        first = self.click(1, [5, 5])
        second = self.click(2, [20, 20])
        self.assertEqual(first["status"], 1)
        self.assertEqual(second["status"], 1)
        self.assertNotEqual(first["ploygon"], second["ploygon"])
        self.assertEqual(first["bnd_boxs"], [[2, 2, 10, 10]])
        self.assertEqual(len(first["min_boxs"][0]), 5)
        self.assertEqual(self.click(3)["ploygon"], first["ploygon"])
        self.assertEqual(self.click(4)["ploygon"], first["ploygon"])
        self.assertEqual(self.click(4)["ploygon"], [])
        self.assertEqual(self.click(2, [20, 20])["status"], 0)
        self.assertEqual(self.click(1, [5, 5])["status"], 1)
        self.assertEqual(self.click(5)["ploygon"], [])
        self.assertEqual(self.click(3)["ploygon"], [])
        self.click(1, [5, 5])
        self.assertEqual(self.click(3)["ploygon"], [])
        self.assertEqual(self.click(2, [20, 20])["status"], 0)

    def test_invalid_inputs_and_failed_inference_do_not_commit_clicks(self):
        for coordinate in (None, [], [1], [-1, 1], [40, 1], [1, 30], [float("nan"), 1], [True, 2]):
            self.assertEqual(self.click(1, coordinate)["status"], 0)
        self.assertEqual(self.session.clicks, [])
        preview = self.click(1, [5, 5])["ploygon"]
        with patch.object(self.session.backend, "segment", side_effect=RuntimeError("inference failed")):
            self.assertEqual(self.click(1, [6, 6])["status"], 0)
        self.assertEqual(len(self.session.clicks), 1)
        self.assertEqual(self.session.history[-1]["polygons"], preview)
        self.click(1, [6, 6])
        self.click(1, [7, 7])
        self.assertEqual(self.click(1, [8, 8])["status"], 0)
        self.assertEqual(len(self.session.clicks), 3)
        wrong_image = self.session.dispatch(click_request(1, [5, 5], "other"))["data"]
        self.assertEqual(wrong_image["status"], 0)

    def test_refinement_undo_and_object_reset_use_matching_logits(self):
        self.click(1, [5, 5])
        first_logits = self.session.history[-1]["logits"]
        self.assertIsNone(self.session.backend.previous_logits[-1])
        self.click(2, [20, 20])
        self.assertIs(self.session.backend.previous_logits[-1], first_logits)
        self.click(3)
        self.click(1, [6, 6])
        self.assertIs(self.session.backend.previous_logits[-1], first_logits)
        for action in (4, 5):
            self.click(action)
            self.click(1, [5, 5])
            self.assertIsNone(self.session.backend.previous_logits[-1])
        self.click(3)
        self.click(1, [5, 5])
        self.assertIsNone(self.session.backend.previous_logits[-1])

    def test_tcp_image_loading_never_saves_images_or_uses_an_image_id_cache(self):
        request = image_request()
        # Block all image/file writers while reloading the same ID and running a click.
        with patch("builtins.open", side_effect=AssertionError("disk access")), \
                patch("cv2.imwrite", side_effect=AssertionError("image write")), \
                patch.object(Image.Image, "save", side_effect=AssertionError("image save")):
            self.assertEqual(self.session.dispatch(request)["data"]["status"], 1)
            self.assertEqual(self.click(1, [5, 5])["status"], 1)
            request["data"]["image_base64"] = "broken"
            self.assertEqual(self.session.dispatch(request)["data"]["status"], 0)
        self.assertIsNone(self.session.features)

    def test_image_change_failure_invalidates_previous_features(self):
        self.click(1, [5, 5])
        invalid = image_request("broken")
        invalid["data"]["image_base64"] = "not valid base64"
        self.assertEqual(self.session.dispatch(invalid)["data"]["status"], 0)
        self.assertEqual(self.click(1, [5, 5])["status"], 0)
        self.assertEqual(self.session.dispatch(image_request())["data"]["status"], 1)
        self.assertEqual(self.click(3)["ploygon"], [])

    def test_model_aliases_query_and_invalid_envelopes(self):
        for model in (0, 1):
            reply = self.session.dispatch({"cmd": 8000, "data": {"model": model, "use_gpu": False}})
            self.assertEqual((reply["cmd"], reply["data"]["status"]), (9000, 1))
        models = self.session.dispatch({"cmd": 8004})
        self.assertEqual(models["cmd"], 9004)
        self.assertEqual([entry["model"] for entry in models["data"]], [0, 1])
        self.assertTrue(all(entry["use_gpu"] == 1 for entry in models["data"]))
        for request in ([], {"cmd": "8001"}, {"cmd": 8001, "data": []}, {"cmd": 8000, "data": {"model": 2}}):
            self.assertEqual(self.session.dispatch(request)["data"]["status"], 0)


class BackendTest(unittest.TestCase):
    def test_holes_and_disconnected_objects_remain_in_polygons(self):
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[1:25, 1:25] = 1
        mask[8:18, 8:18] = 0
        mask[28:38, 35:50] = 1
        polygons = mask_to_polygons(mask)
        self.assertEqual(len(polygons), 2)
        rendered = np.zeros_like(mask)
        for polygon in polygons:
            cv2.fillPoly(rendered, [np.asarray(polygon, dtype=np.int32)], 1)
        self.assertEqual(rendered[12, 12], 0)
        self.assertEqual(rendered[3, 3], 1)
        self.assertEqual(rendered[32, 40], 1)

    def test_tcp_polygon_epsilon_reduces_vertices_with_limited_shape_change(self):
        y, x = np.indices((240, 240))
        angle = np.arctan2(y - 120, x - 120)
        radius = np.hypot(x - 120, y - 120)
        mask = (radius < 75 + 4 * np.sin(angle * 24)).astype(np.uint8)
        detailed = mask_to_polygons(mask, epsilon=0)
        simplified = mask_to_polygons(mask, epsilon=2)
        self.assertLess(len(simplified[0]), len(detailed[0]))

        rendered = []
        for polygons in (detailed, simplified):
            image = np.zeros_like(mask)
            cv2.fillPoly(image, [np.asarray(polygons[0], dtype=np.int32)], 1)
            rendered.append(image.astype(bool))
        intersection = np.logical_and(*rendered).sum()
        union = np.logical_or(*rendered).sum()
        self.assertGreater(intersection / union, 0.98)


class WireTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = InteractiveSegmentationServer(("127.0.0.1", 0), FakeBackend())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def connect(self):
        connection = socket.create_connection(self.server.server_address, timeout=5)
        self.addCleanup(connection.close)
        stream = connection.makefile("rb")
        self.addCleanup(stream.close)
        return connection, stream

    def receive(self, stream):
        # Delimiters can span recv boundaries; read exactly one response without losing the next.
        data = bytearray()
        while not data.endswith(DELIMITER):
            char = stream.read(1)
            self.assertTrue(char, "server closed before sending a complete response")
            data.extend(char)
        return json.loads(data[:-len(DELIMITER)])

    def send(self, connection, request):
        connection.sendall(json.dumps(request).encode() + DELIMITER)

    def test_fragmented_frames_batched_frames_and_connection_isolation(self):
        connection, stream = self.connect()
        frame = json.dumps(image_request()).encode() + DELIMITER
        connection.sendall(frame[:11])
        connection.sendall(frame[11:-2])
        connection.sendall(frame[-2:] + json.dumps(click_request(1, [5, 5])).encode() + DELIMITER)
        self.assertEqual(self.receive(stream)["cmd"], 9001)
        preview = self.receive(stream)
        self.assertEqual((preview["cmd"], preview["data"]["status"]), (9002, 1))
        other, other_stream = self.connect()
        self.send(other, click_request(1, [5, 5]))
        self.assertEqual(self.receive(other_stream)["data"]["status"], 0)
        self.send(other, image_request("other"))
        self.assertEqual(self.receive(other_stream)["data"]["status"], 1)
        self.send(other, click_request(3, image_id="other"))
        self.assertEqual(self.receive(other_stream)["data"]["ploygon"], [])
        self.send(connection, click_request(4))
        self.assertEqual(self.receive(stream)["data"]["ploygon"], preview["data"]["ploygon"])

    def test_malformed_and_oversized_frames(self):
        connection, stream = self.connect()
        connection.sendall(b"{invalid" + DELIMITER + b"\xff" + DELIMITER)
        self.assertEqual(self.receive(stream)["data"]["status"], 0)
        self.assertEqual(self.receive(stream)["data"]["status"], 0)
        self.send(connection, {"cmd": 8004})
        self.assertEqual(self.receive(stream)["cmd"], 9004)
        with patch("sam3_service.eiseg_tcp.MAX_FRAME_BYTES", 64):
            connection.sendall(b"x" * 65)
            self.assertEqual(self.receive(stream)["data"]["message"], "TCP frame too large")
            self.assertEqual(stream.read(1), b"")

    def test_relative_image_url_uses_legacy_base(self):
        class ImageHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(image_bytes())

            def log_message(self, *args):
                pass

        http = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        connection, stream = self.connect()
        with patch.dict("os.environ", {"SAM3_TCP_IMAGE_BASE_URL": f"http://127.0.0.1:{http.server_port}/"}):
            self.send(connection, {"cmd": 8001, "data": {"image_id": "url", "image_url": "demo.png"}})
            self.assertEqual(self.receive(stream)["data"]["status"], 1)
        self.send(connection, click_request(1, [5, 5], "url"))
        self.assertEqual(self.receive(stream)["data"]["status"], 1)


if __name__ == "__main__":
    unittest.main()
