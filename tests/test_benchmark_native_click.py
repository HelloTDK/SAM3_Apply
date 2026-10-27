"""Exercise the load tester's session protocol without Gradio, TensorRT or a GPU."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "native_click_benchmark", Path(__file__).resolve().parents[1] / "tools/benchmark_native_click.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class SessionProtocolTest(unittest.TestCase):
    def setUp(self):
        names = ("load_image", "predict_click", "clear_image", "clear_clicks")
        self.config = {"api_prefix": "/gradio_api", "dependencies": [
            {"id": index, "api_name": name} for index, name in enumerate(names)
        ]}
        self.pending = {}
        self.counts = {}
        self.clicks = []
        self.error_status = None

    def server(self, url, timeout, body=None, content_type=None):
        if url.endswith("/queue/join"):
            payload = json.loads(body)
            self.pending[payload["session_hash"]] = payload
            return io.BytesIO(b'{"event_id": "event"}')
        session = benchmark.urllib.parse.parse_qs(benchmark.urllib.parse.urlparse(url).query)["session_hash"][0]
        payload = self.pending.pop(session)
        if payload["fn_index"] == 0:
            self.counts[session] = 0
            data = [None, "图片已加载：100 × 80"]
        elif payload["fn_index"] == 2:
            self.counts.pop(session)
            data = [None, None, None, "请先上传图片"]
        else:
            # State must stay server-side; clients must send only a null placeholder.
            self.assertEqual(payload["data"][1], None)
            self.clicks.append((session, payload["event_data"]))
            self.counts[session] += 1
            data = [None, {"path": "preview.png"}, {"path": "mask.png"},
                    self.error_status or f"正例，点击数：{self.counts[session]}，置信度：0.9"]
        messages = [
            {"msg": "estimation", "event_id": "event", "rank": 1},
            {"msg": "process_completed", "event_id": "event", "success": True,
             "output": {"data": data}},
            {"msg": "close_stream"},
        ]
        return io.BytesIO(b"".join(b"data: " + json.dumps(msg).encode() + b"\n\n" for msg in messages))

    def client(self):
        client = benchmark.DemoSession("http://localhost:7861", self.config, 10)
        client.image = {"path": "uploaded.jpg", "meta": {"_type": "gradio.FileData"}}
        return client

    def test_users_have_independent_history_and_send_both_click_labels(self):
        first, second = self.client(), self.client()
        positive, negative = {"x": 50, "y": 40, "label": 1}, {"x": 90, "y": 72, "label": 0}
        with patch.object(benchmark, "request", side_effect=self.server):
            first.call("load_image")
            second.call("load_image")
            first.call("predict_click", positive)
            first.call("predict_click", negative)
            second.call("predict_click", positive)
            self.assertEqual((first.click_count, second.click_count), (2, 1))
            first.call("clear_image")
        self.assertNotEqual(first.session_hash, second.session_hash)
        self.assertEqual(self.clicks[1], (first.session_hash, negative))
        self.assertNotIn(first.session_hash, self.counts)
        self.assertEqual(self.counts[second.session_hash], 1)

    def test_http_success_with_inference_failure_is_reported_as_failure(self):
        self.error_status = "分割失败：CUDA out of memory"
        client = self.client()
        with patch.object(benchmark, "request", side_effect=self.server):
            client.call("load_image")
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                client.call("predict_click", {"x": 1, "y": 1, "label": 1})
        self.assertEqual(client.click_count, 0)

    def test_lost_server_state_is_not_counted_as_a_successful_click(self):
        client = self.client()
        with patch.object(benchmark, "request", side_effect=self.server):
            client.call("load_image")
            self.counts[client.session_hash] = 5
            with self.assertRaisesRegex(RuntimeError, "session state was lost"):
                client.call("predict_click", {"x": 1, "y": 1, "label": 1})

    def test_latency_percentiles_exclude_failures_but_report_failure_count(self):
        rows = [{"success": True, "seconds": value} for value in range(1, 21)]
        rows.append({"success": False, "seconds": 1000})
        self.assertEqual(benchmark.latency_summary(rows), {
            "requests": 21, "failures": 1, "p50_s": 10.5, "p95_s": 19,
        })

    def test_gpu_monitor_separates_server_memory_from_other_gpu_processes(self):
        monitor = benchmark.GpuMonitor("0", 123, 0.5)
        responses = [SimpleNamespace(stdout="GPU-test, 9000, 16384, 80\n"),
                     SimpleNamespace(stdout="123, 6000\n456, 2500\n")]
        with patch.object(benchmark.subprocess, "run", side_effect=responses):
            row = monitor.sample()
        self.assertEqual(row["gpu_used_mib"], 9000)
        self.assertEqual(row["server_used_mib"], 6000)


if __name__ == "__main__":
    unittest.main()
