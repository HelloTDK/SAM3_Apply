"""Unit checks; set SAM3_RUN_OM_INTEGRATION=1 to exercise real OM HTTP inference."""

import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

try:
    import acl
except ImportError:
    if os.getenv("SAM3_RUN_OM_INTEGRATION") == "1":
        raise
    sys.modules["acl"] = Mock()

from sam3_service.acl_runtime import ACLModel
from sam3_service.om_backend import OMBackend, postprocess


class OMUnitTests(unittest.TestCase):
    def test_postprocess_nms_coordinates_and_logit_threshold(self):
        outputs = {
            "pred_masks": np.asarray([[[[1, -1], [1, -1]]] * 3], dtype=np.float32),
            "pred_boxes": np.asarray([[[.1, .2, .5, .8], [.1, .2, .5, .8], [.6, .2, .9, .8]]]),
            "pred_logits": np.asarray([[4, 3, -4]], dtype=np.float32),
            "presence_logits": np.asarray([[4]], dtype=np.float32),
        }
        arrays, count = postprocess(outputs, (10, 20), .3, .7)
        self.assertEqual(count, 3)
        np.testing.assert_allclose(arrays["boxes"], [[2, 2, 10, 8]])
        self.assertEqual(arrays["masks"].shape, (1, 10, 20))
        self.assertTrue(arrays["masks"][0, :, 0].all())
        self.assertFalse(arrays["masks"][0, :, -1].any())
        empty, _ = postprocess(outputs, (10, 20), 1, .7)
        self.assertEqual(empty["masks"].shape, (0, 10, 20))

    def test_input_shape_rejected_before_acl_allocation(self):
        model = ACLModel.__new__(ACLModel)
        model.path = "fixture.om"
        model.in_shapes, model.in_dtypes = [(1, 50, 4)], [np.dtype("float32")]
        with self.assertRaisesRegex(ValueError, "expects"):
            model.run([np.zeros((1, 1, 4))])

    def test_visual_prompts_keep_negative_labels_and_cls(self):
        backend = OMBackend.__new__(OMBackend)
        backend.box_capacity = 50
        geometry = Mock()
        geometry.run.return_value = (np.arange(51 * 256).reshape(1, 51, 256), np.ones((1, 51)))
        backend.models = {"geometry-encoder": geometry}
        embed, mask = backend.encode_visual(
            {"fpn_feat_2": np.zeros(1), "fpn_pos_2": np.zeros(1)},
            (100, 200), [[20, 10, 40, 20], [100, 20, 20, 40]], [1, 0],
        )
        boxes, labels = geometry.run.call_args.args[0][:2]
        np.testing.assert_allclose(boxes[0, :2], [[.2, .2, .2, .2], [.55, .4, .1, .4]])
        self.assertEqual(labels[0, :4].tolist(), [1, 0, 1, 0])
        self.assertEqual(embed.shape, (1, 3, 256))
        self.assertEqual(embed[0, -1, 0], 50 * 256)
        self.assertEqual(mask.shape, (1, 3))
        with self.assertRaises(ValueError):
            backend.encode_visual({}, (100, 200), [[0, 0, 1, 1]], [2])

    def test_prompt_overflow_never_silently_truncates(self):
        backend = OMBackend.__new__(OMBackend)
        backend.visual_capacity = 51
        with self.assertRaisesRegex(ValueError, "got 52"):
            backend.grounding({}, (10, 20), np.zeros((1, 52, 256)), np.ones((1, 52)))

    def test_acl_execute_failure_releases_input_buffers(self):
        model = ACLModel.__new__(ACLModel)
        model.path, model.model_id, model.output_dataset = "fixture.om", 7, 8
        model.in_shapes, model.in_dtypes, model.in_sizes = [(1,)], [np.dtype("float32")], [4]
        fake_acl = Mock()
        fake_acl.mdl.create_dataset.return_value = 11
        fake_acl.rt.malloc.return_value = (12, 0)
        fake_acl.rt.memcpy.return_value = 0
        fake_acl.create_data_buffer.return_value = 13
        fake_acl.mdl.add_dataset_buffer.return_value = (None, 0)
        fake_acl.mdl.execute.return_value = 500
        with patch("sam3_service.acl_runtime.acl", fake_acl):
            with self.assertRaisesRegex(RuntimeError, "ret=500"):
                model.run([np.ones(1)])
        fake_acl.destroy_data_buffer.assert_called_once_with(13)
        fake_acl.mdl.destroy_dataset.assert_called_once_with(11)
        fake_acl.rt.free.assert_called_once_with(12)


@unittest.skipUnless(os.getenv("SAM3_RUN_OM_INTEGRATION") == "1", "requires Ascend OM models")
class OMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sam3-om-test-")
        os.environ.update(SAM3_BACKEND="om", SAM3_SAVE_UPLOADS="0", SAM3_ARGOS_FORCE_OFFLINE="1",
                          SAM3_ARGOS_AUTO_INSTALL="0", SAM3_SERIALIZE_MODEL_ACCESS="0",
                          SAM3_API_KEYS_FILE=cls.tmp.name + "/keys.json")
        from fastapi.testclient import TestClient
        from sam3_service.app import app
        cls.client = TestClient(app).__enter__()
        cls.image_bytes = Path(os.environ["SAM3_OM_TEST_IMAGE"]).read_bytes()
        cls.image_base64 = base64.b64encode(cls.image_bytes).decode()
        key = cls.client.post("/ui/api-keys", json={"name": "integration", "role": "admin"}).json()["data"]
        cls.headers = {"X-API-Key": key["api_key"]}
        cls.key_id = key["id"]
        cls.box = [356, 281, 138, 378]
        cls.other_box = [941, 308, 212, 279]

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.tmp.cleanup()

    def post(self, path, payload):
        response = self.client.post(path, json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text[:1000])
        return response.json()

    def test_01_text_box_auth_and_public_pages(self):
        for path in ["/", "/health", "/docs", "/api-list", "/apis", "/api-list.json", "/openapi.json"]:
            self.assertEqual(self.client.get(path).status_code, 200)
        self.assertEqual(self.client.get("/health").json()["backend"], "om")
        self.assertEqual(self.client.get("/health").json()["model_access_mode"], "serialized")
        payload = {"pic_id": "text", "image_base64": self.image_base64, "prompt": "#person#basketball hoop"}
        self.assertEqual(self.client.post("/v1/segmentations", json=payload).status_code, 401)
        result = self.post("/v1/segmentations", payload)
        self.assertGreater(result["num_detections"], 0)
        self.assertEqual(result["classes_detected"], 2)
        for label in result["pic_labels"]:
            self.assertGreater(label["mask_area"], 0)
            self.assertGreaterEqual(len(label["polygon_points"]), 3)
        boxes = self.post("/v1/box-segmentations", {
            "pic_id": "box", "image_base64": self.image_base64,
            "bnd_points_list": [self.box, self.other_box],
        })
        self.assertEqual(boxes["num_segmentations"], 2)
        self.assertFalse(any(item["used_fallback"] for item in boxes["segmentations"]))
        response = self.client.post("/detect", files={"file": ("image.jpg", self.image_bytes)},
                                    data={"prompt": "#person"})
        self.assertEqual(response.status_code, 200, response.text[:1000])
        self.assertGreater(response.json()["num_detections"], 0)
        self.assertEqual(self.client.get("/v1/api-keys", headers=self.headers).status_code, 200)

    def test_02_single_multi_and_multipart_visual_prompts(self):
        payload = {"pic_id": "similar", "reference_image_base64": self.image_base64,
                   "query_image_base64_list": [self.image_base64, self.image_base64],
                   "reference_bnd_points": self.box, "sam_threshold": .1, "prompt": "person"}
        result = self.post("/v1/similar-object-segmentations", payload)
        self.assertEqual(result["batch_size"], 2)
        self.assertGreater(result["total_num_matches"], 0)
        same_image = self.post("/v1/similar-object-segmentations", {
            **payload, "similar_mode": "same_image_prompt", "query_image_base64_list": [],
        })
        self.assertEqual(same_image["batch_size"], 1)
        self.assertGreater(same_image["num_matches"], 0)
        multi = self.post("/v1/multi-similar-object-segmentations", {
            "pic_id": "multi", "query_image_base64": self.image_base64, "sam_threshold": .1,
            "prompt": "#basketball hoop", "samples": [{
                "sample_id": "players", "reference_image_base64": self.image_base64,
                "instances": [
                    {"category": "player", "reference_bnd_points": self.box},
                    {"category": "player", "reference_bnd_points": self.other_box, "sample_type": "negative"},
                ],
            }],
        })
        self.assertEqual(multi["num_negative_samples"], 1)
        self.assertGreater(multi["num_matches"], 0)
        self.assertGreaterEqual(multi["num_groups"], 2)
        negative_only = self.post("/v1/multi-similar-object-segmentations", {
            "pic_id": "negative-text", "query_image_base64": self.image_base64,
            "prompt": "#person", "sam_threshold": .1, "samples": [{
                "reference_image_base64": self.image_base64, "category": "person",
                "reference_bnd_points": self.other_box, "sample_type": "negative",
            }],
        })
        self.assertEqual(negative_only["num_negative_samples"], 1)
        self.assertEqual(negative_only["profile"]["groups"][0]["num_negative_samples"], 1)
        response = self.client.post("/similar-detect", files={"reference_file": ("ref.jpg", self.image_bytes)},
                                    data={"reference_bnd_points": ",".join(map(str, self.box)),
                                          "similar_mode": "same_image_prompt", "sam_threshold": .1})
        self.assertEqual(response.status_code, 200, response.text[:1000])
        response = self.client.post("/multi-similar-detect", files=[
            ("sample_file", ("ref.jpg", self.image_bytes)),
            ("query_file", ("query.jpg", self.image_bytes)),
        ], data={"sample_meta": "[]", "prompt": "#person", "sam_threshold": .1})
        self.assertEqual(response.status_code, 200, response.text[:1000])

    def test_03_url_cached_samples_and_image_video_tasks(self):
        root = Path(self.tmp.name)
        (root / "image.jpg").write_bytes(self.image_bytes)
        (root / "samples.json").write_text(json.dumps([{"label_id": "player", "label_sample_data": [{
            "image_id": "sample", "image_url": "image.jpg",
            "image_mark": [{"mark_info": self.box, "sample_type": "1"}],
        }]}]))
        (root / "images.json").write_text(json.dumps([{"image_id": "one", "image_url": "image.jpg"}]))
        frame = cv2.imdecode(np.frombuffer(self.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        writer = cv2.VideoWriter(str(root / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 1, (frame.shape[1], frame.shape[0]))
        self.assertTrue(writer.isOpened())
        writer.write(frame)
        writer.release()
        (root / "videos.json").write_text(json.dumps([{"video_id": "clip", "video_url": "clip.avi"}]))
        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            shared = {"download_url": f"http://127.0.0.1:{server.server_port}",
                      "sample_url": "samples.json", "sam_threshold": .1,
                      "prompt_category_map": {"player": "person"}}
            first = self.post("/v1/similar-object-segmentations/by-url", {
                **shared, "pic_id": "url", "query_image_url": "image.jpg", "return_result_image": True,
            })
            self.assertGreater(first["num_matches"], 0)
            cached = self.post("/v1/similar-object-segmentations/by-url", {
                **shared, "pic_id": "url-cached", "query_image_url": "image.jpg",
            })
            self.assertTrue(cached["profile"]["sample_cache_hit"])
            prompt_only = self.post("/v1/similar-object-segmentations/by-url", {
                "download_url": shared["download_url"], "pic_id": "text-url",
                "query_image_url": "image.jpg", "prompt": "#person", "sam_threshold": .1,
            })
            self.assertGreater(prompt_only["num_matches"], 0)
            for data_type, manifest in [(0, "images.json"), (1, "videos.json")]:
                task_id = f"integration-{data_type}"
                self.post("/v1/similar-object-segmentations/tasks", {
                    **shared, "task_id": task_id, "data_type": data_type, "data_url": manifest,
                })
                task_path = f"/v1/similar-object-segmentations/tasks/{task_id}"
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    status = self.client.get(task_path, headers=self.headers).json()
                    if status["status"] not in {"pending", "running"}:
                        break
                    time.sleep(.1)
                self.assertEqual(status["status"], "completed", status)
                page = self.client.get(task_path + "/results", headers=self.headers).json()
                self.assertEqual(page["items"][0]["status"], 1, page)
                self.assertEqual(self.client.delete(task_path, headers=self.headers).status_code, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_04_concurrency_and_idle_reload(self):
        from sam3_service import runtime
        payload = {"pic_id": "parallel", "image_base64": self.image_base64, "prompt": "#person"}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: self.post("/v1/segmentations", item),
                                    [payload, {**payload, "prompt": "#basketball hoop"}]))
        self.assertEqual(results[0]["pic_labels"][0]["category"], "person")
        self.assertEqual(results[1]["pic_labels"][0]["category"], "basketball hoop")
        runtime.cleanup_cuda_runtime_state(unload_model=True)
        health = self.client.get("/health").json()
        self.assertFalse(health["model_loaded"])
        self.assertTrue(health["idle_model_unloaded"])
        self.assertGreater(self.post("/v1/segmentations", payload)["num_detections"], 0)
        self.assertTrue(self.client.get("/health").json()["model_loaded"])


if __name__ == "__main__":
    unittest.main()
