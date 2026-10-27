"""Check the actual Gradio callbacks and timing without TensorRT or a GPU."""

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from sam3_service.native_click_trt import NativeInteractiveBackend
from test_native_click_trt import FakeDecoder


class DemoTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "app_tensorrt.py"
        spec = importlib.util.spec_from_file_location("tensorrt_demo_timing_test", path)
        cls.demo = importlib.util.module_from_spec(spec)
        # Keep Gradio's background analytics out of this local callback check.
        with patch("gradio.analytics.analytics_enabled", return_value=False):
            spec.loader.exec_module(cls.demo)

    def test_combined_startup_shares_backend_and_closes_tcp_on_gradio_exit(self):
        backend = Mock()
        for error in (None, RuntimeError("Gradio launch failed")):
            with self.subTest(error=error), \
                    patch.object(self.demo, "get_backend", return_value=backend), \
                    patch("sam3_service.eiseg_tcp.InteractiveSegmentationServer") as server_type, \
                    patch.object(self.demo.threading, "Thread") as thread_type, \
                    patch.object(self.demo.demo, "queue") as queue, \
                    patch.dict("os.environ", {"SAM3_TCP_HOST": "127.0.0.1", "SAM3_TCP_PORT": "16063"}):
                queue.return_value.launch.side_effect = error
                if error is None:
                    self.demo.main()
                else:
                    with self.assertRaisesRegex(RuntimeError, "Gradio launch failed"):
                        self.demo.main()
                server_type.assert_called_once_with(("127.0.0.1", 16063), backend=backend)
                server = server_type.return_value.__enter__.return_value
                thread_type.assert_called_once_with(target=server.start_server, daemon=True)
                thread_type.return_value.start.assert_called_once()
                server.shutdown.assert_called_once()
                thread_type.return_value.join.assert_called_once()

    def test_upload_click_refinement_undo_clear_and_failure_have_timing(self):
        backend = NativeInteractiveBackend.__new__(NativeInteractiveBackend)
        backend.target_size = (1008, 1008)
        backend.max_clicks = 32
        backend.encoder = Mock(return_value={
            name: np.zeros((1, 1)) for name in ("image_embed", "high_res_0", "high_res_1")
        })
        backend.first_decoder = FakeDecoder(
            np.ones((1, 3, 288, 288), dtype=np.float32), np.asarray([[0.1, 0.9, 0.2]])
        )
        backend.refine_decoder = FakeDecoder(
            np.ones((1, 1, 288, 288), dtype=np.float32), np.asarray([[0.8]])
        )
        image = Image.new("RGB", (100, 80))
        output = io.StringIO()
        with patch.object(self.demo, "BACKEND", backend), redirect_stdout(output):
            session, status = self.demo.load_image(image)
            self.assertIn("耗时：", status)
            for label in (1, 0):
                session, preview, mask, status = self.demo.predict_click(
                    image, session, SimpleNamespace(x=50, y=40, label=label)
                )
                self.assertEqual(preview.size, image.size)
                self.assertEqual(mask.size, image.size)
                self.assertIn("耗时：", status)
            self.assertEqual(len(session["clicks"]), 2)
            self.assertEqual(backend.encoder.call_count, 1)
            self.assertEqual(len(backend.first_decoder.calls), 1)
            self.assertEqual(len(backend.refine_decoder.calls), 1)
            session, _, _, status = self.demo.undo_click(session)
            self.assertEqual(len(session["clicks"]), 1)
            self.assertIn("耗时：", status)
            session, _, mask, status = self.demo.clear_clicks(session)
            self.assertEqual(session["history"], [])
            self.assertIsNone(mask)
            self.assertIn("耗时：", status)
            self.assertIn("耗时：", self.demo.clear_image()[-1])

            with patch.object(backend, "segment", side_effect=RuntimeError("expected")):
                _, _, _, status = self.demo.predict_click(
                    image, session, SimpleNamespace(x=50, y=40, label=1)
                )
            self.assertIn("分割失败：expected", status)
            self.assertIn("耗时：", status)
            self.assertEqual(session["history"], [])

        for step in ("图片预处理", "图片编码合计", "点击与 mask 提示准备", "输出回传 CPU",
                     "mask 后处理", "预览绘制", "加载图片总计", "点击分割总计",
                     "撤销点击总计", "清空点击总计", "清除图片总计"):
            self.assertIn(f"{step}耗时：", output.getvalue())


if __name__ == "__main__":
    unittest.main()
