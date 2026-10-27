#!/usr/bin/env python3
"""Measure the live native-click Gradio demo with independent concurrent sessions.

No CUDA/Python client packages are needed: requests use Gradio's browser queue
protocol, including event_data for positive/negative clicks. Run nvidia-smi on
the server host (or inside its GPU-enabled container) to measure the right GPU.
"""

import argparse
import concurrent.futures
import csv
import json
import math
import mimetypes
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid


def request(url, timeout, body=None, content_type="application/json"):
    headers = {"Content-Type": content_type} if body is not None else {}
    return urllib.request.urlopen(
        urllib.request.Request(url, data=body, headers=headers), timeout=timeout
    )


class DemoSession:
    """One browser-equivalent session; gr.State is addressed by session_hash."""

    def __init__(self, base_url, config, timeout):
        self.api = base_url.rstrip("/") + config.get("api_prefix", "/gradio_api")
        self.functions = {
            dep["api_name"].lstrip("/"): dep["id"]
            for dep in config["dependencies"] if dep.get("api_name")
        }
        for name in ("load_image", "predict_click", "clear_image", "clear_clicks"):
            if name not in self.functions:
                raise ValueError(f"This is not the native-click demo: missing {name}")
        self.timeout = timeout
        self.session_hash = "sam3-bench-" + uuid.uuid4().hex
        self.image = None
        self.click_count = 0

    def upload(self, image_path):
        boundary = uuid.uuid4().hex
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        # A safe fixed filename keeps arbitrary local filenames out of MIME headers.
        filename = "benchmark" + image_path.suffix.lower()
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
            f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode() + image_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        with request(self.api + "/upload", self.timeout, body,
                     f"multipart/form-data; boundary={boundary}") as response:
            paths = json.load(response)
        self.image = {"path": paths[0], "meta": {"_type": "gradio.FileData"}}

    def call(self, name, event_data=None):
        inputs = {
            "load_image": [self.image], "predict_click": [self.image, None],
            "clear_image": [], "clear_clicks": [None],
        }
        body = json.dumps({
            "fn_index": self.functions[name], "session_hash": self.session_hash,
            "data": inputs[name], "event_data": event_data,
        }).encode()
        deadline = time.monotonic() + self.timeout
        with request(self.api + "/queue/join", self.timeout, body) as response:
            event_id = json.load(response)["event_id"]
        query = urllib.parse.urlencode({"session_hash": self.session_hash})
        output = None
        # Drain close_stream before the next call, so a closing SSE connection cannot
        # cancel the next event in this same session.
        with request(self.api + "/queue/data?" + query, self.timeout) as response:
            for line in response:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} exceeded {self.timeout}s including queue time")
                if not line.startswith(b"data:"):
                    continue
                message = json.loads(line[5:])
                if message.get("msg") in ("unexpected_error", "server_stopped"):
                    raise RuntimeError(str(message))
                if message.get("event_id") == event_id and message.get("msg") == "process_completed":
                    output = message.get("output", {})
                    if not message.get("success", False) or output.get("error"):
                        raise RuntimeError(str(output))
                if message.get("msg") == "close_stream":
                    break
        if output is None:
            raise RuntimeError(f"No completed result for {name}")
        data = output.get("data", [])
        status = data[-1] if data else ""
        # The demo catches inference errors and returns HTTP 200: check its status too.
        if name == "load_image":
            size = re.search(r"图片已加载[：:]\s*(\d+)\s*×\s*(\d+)", str(status))
            if not size:
                raise RuntimeError(f"Image load failed: {status}")
            self.width, self.height = map(int, size.groups())
        elif name == "predict_click":
            count = re.search(r"点击数[：:]\s*(\d+)", str(status))
            if not count or int(count[1]) != self.click_count + 1 or len(data) < 3 or data[-2] is None:
                raise RuntimeError(f"Click failed or session state was lost: {status}")
            self.click_count += 1
        elif name == "clear_clicks":
            self.click_count = 0
        elif status != "请先上传图片":
            raise RuntimeError(f"Session cleanup failed: {status}")
        return status


class GpuMonitor:
    def __init__(self, gpu, pid, interval):
        self.gpu, self.pid, self.interval = gpu, pid, interval
        self.rows = []
        self.errors = []
        self.phase = "idle"
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        # Capture the phase before nvidia-smi, so stage transitions do not relabel old data.
        phase = self.phase
        result = subprocess.run([
            "nvidia-smi", f"--id={self.gpu}",
            "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=5)
        gpu_uuid, used, total, utilization = next(csv.reader(result.stdout.splitlines()))
        row = {
            "elapsed_s": round(time.monotonic() - self.started, 3), "phase": phase,
            "gpu_uuid": gpu_uuid.strip(), "gpu_used_mib": float(used),
            "gpu_total_mib": float(total), "gpu_utilization_pct": float(utilization),
            "server_used_mib": None,
        }
        if self.pid is not None:
            result = subprocess.run([
                "nvidia-smi", f"--id={self.gpu}",
                "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits",
            ], check=True, capture_output=True, text=True, timeout=5)
            memory = [float(memory) for pid, memory in csv.reader(result.stdout.splitlines())
                      if pid.strip() == str(self.pid)]
            row["server_used_mib"] = sum(memory) if memory else None
        with self.lock:
            self.rows.append(row)
        return row

    def run(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.errors.append(str(exc))


def latency_summary(records):
    values = sorted(row["seconds"] for row in records if row["success"])
    return {
        "requests": len(records), "failures": sum(not row["success"] for row in records),
        "p50_s": round(statistics.median(values), 4) if values else None,
        "p95_s": round(values[max(0, math.ceil(len(values) * 0.95) - 1)], 4) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7861")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--users", nargs="+", type=int, default=[1, 4, 8, 16, 32],
                        help="Increasing TOTAL retained sessions per stage")
    parser.add_argument("--clicks", type=int, default=4, help="Clicks per user in each stage (2..32)")
    parser.add_argument("--hold-seconds", type=float, default=5, help="Retain all sessions after clicks")
    parser.add_argument("--interval", type=float, default=0.5, help="GPU sampling interval in seconds")
    parser.add_argument("--timeout", type=float, default=180, help="Per-request timeout including queue time")
    parser.add_argument("--gpu", default="0", help="Local nvidia-smi GPU index or UUID")
    parser.add_argument("--pid", type=int, help="Optional server PID as reported by nvidia-smi")
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / time.strftime("native_click_benchmark_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("--image must be a readable local image")
    if not args.users or args.users[0] < 1 or args.users != sorted(set(args.users)):
        parser.error("--users must contain increasing positive counts, e.g. 1 4 8 16 32")
    if not 2 <= args.clicks <= 32 or args.hold_seconds < 0 or args.interval <= 0 or args.timeout <= 0:
        parser.error("Require 2 <= clicks <= 32, hold-seconds >= 0, interval > 0, timeout > 0")
    args.output.mkdir(parents=True, exist_ok=False)
    with request(args.url.rstrip("/") + "/config", args.timeout) as response:
        config = json.load(response)
    monitor = GpuMonitor(args.gpu, args.pid, args.interval)
    idle = monitor.sample()  # Fail clearly before sending load if GPU monitoring is unavailable.
    monitor.thread.start()
    sessions, records, stages = [], [], []
    report = {"url": args.url, "image": str(args.image.resolve()), "gpu": args.gpu,
              "pid": args.pid, "gradio_version": config.get("version"), "idle": idle,
              "stages": stages, "errors": [], "monitor_errors": monitor.errors}
    records_lock = threading.Lock()

    def timed(session, operation, function):
        started = time.monotonic()
        row = {"stage": monitor.phase, "session": session.session_hash,
               "operation": operation, "success": False, "error": ""}
        try:
            value = function()
            row["success"] = True
            return value
        except Exception as exc:
            row["error"] = str(exc)
            raise
        finally:
            row["seconds"] = round(time.monotonic() - started, 4)
            with records_lock:
                records.append(row)

    def load(session):
        timed(session, "upload", lambda: session.upload(args.image))
        timed(session, "load_image", lambda: session.call("load_image"))

    def click(session):
        if session.click_count + args.clicks > 32:
            timed(session, "clear_clicks", lambda: session.call("clear_clicks"))
        for index in range(args.clicks):
            # Alternate an interior positive point and an exterior negative point.
            fraction = 0.5 if index % 2 == 0 else 0.9
            event = {"x": min(session.width - 1, int(session.width * fraction)),
                     "y": min(session.height - 1, int(session.height * fraction)),
                     "label": 1 if index % 2 == 0 else 0}
            operation = "first_click" if session.click_count == 0 else "refine_click"
            timed(session, operation, lambda: session.call("predict_click", event))

    def parallel(function, clients):
        if not clients:
            return
        # One worker per user: calls within a session stay ordered, across users overlap.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as pool:
            futures = [pool.submit(function, session) for session in clients]
            failures = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(str(exc))
        if failures:
            raise RuntimeError(f"{len(failures)} user(s) failed: {failures[0]}")

    def settle():
        # Keep sampling during the hold rather than only observing request boundaries.
        monitor.stop.wait(args.hold_seconds)
        return monitor.sample()

    try:
        print(f"Testing {args.url}; GPU {args.gpu}; output: {args.output}", flush=True)
        monitor.phase = "warmup"
        sessions.append(DemoSession(args.url, config, args.timeout))
        load(sessions[0])
        click(sessions[0])
        timed(sessions[0], "clear_clicks", lambda: sessions[0].call("clear_clicks"))
        baseline = settle()
        report["warm_baseline"] = baseline
        if args.pid is not None and baseline["server_used_mib"] is None:
            raise RuntimeError("--pid was not reported on this GPU; use the PID shown by nvidia-smi")
        print(f"Warm baseline (one cached image): {baseline['gpu_used_mib']:.0f} MiB", flush=True)
        if args.pid is not None:
            print(f"Server PID {args.pid} warm baseline: {baseline['server_used_mib']:.0f} MiB", flush=True)
        print("users | held MiB | peak MiB | delta/warm MiB | click p50/p95 s | failures", flush=True)
        for count in args.users:
            phase = f"users_{count}"
            monitor.phase = phase
            first_record = len(records)
            started = time.monotonic()
            added = [DemoSession(args.url, config, args.timeout) for _ in range(count - len(sessions))]
            sessions.extend(added)
            parallel(load, added)
            monitor.sample()
            click_started = time.monotonic()
            parallel(click, sessions)
            click_seconds = time.monotonic() - click_started
            held = settle()
            samples = [row for row in monitor.rows if row["phase"] == phase]
            clicks = [row for row in records[first_record:] if row["operation"].endswith("click")]
            stats = latency_summary(clicks)
            stage = {
                "users": count, "seconds": round(time.monotonic() - started, 3),
                "held_gpu_mib": held["gpu_used_mib"],
                "peak_gpu_mib": max(row["gpu_used_mib"] for row in samples),
                "delta_from_warm_mib": held["gpu_used_mib"] - baseline["gpu_used_mib"],
                "held_server_mib": held["server_used_mib"],
                "peak_server_mib": max((row["server_used_mib"] for row in samples
                                        if row["server_used_mib"] is not None), default=None),
                "server_delta_from_warm_mib": (
                    held["server_used_mib"] - baseline["server_used_mib"]
                    if held["server_used_mib"] is not None and baseline["server_used_mib"] is not None
                    else None
                ),
                "clicks": stats,
                "clicks_per_second": round(len(clicks) / click_seconds, 3),
                "operations": {name: latency_summary([row for row in records[first_record:]
                                                      if row["operation"] == name])
                               for name in sorted({row["operation"] for row in records[first_record:]})},
            }
            stages.append(stage)
            print(f"{count:5d} | {stage['held_gpu_mib']:8.0f} | {stage['peak_gpu_mib']:8.0f} | "
                  f"{stage['delta_from_warm_mib']:14.0f} | {stats['p50_s']:.3f}/{stats['p95_s']:.3f} | "
                  f"{stats['failures']}", flush=True)
            if args.pid is not None:
                print(f"      server: held={stage['held_server_mib']} MiB, "
                      f"peak={stage['peak_server_mib']} MiB, "
                      f"delta/warm={stage['server_delta_from_warm_mib']} MiB", flush=True)
    except Exception as exc:
        report["errors"].append(str(exc))
        print(f"STOP: {exc}", flush=True)
    finally:
        monitor.phase = "cleanup"
        # Reset only our own states; never restart the service or clear its global pool.
        for session in sessions:
            try:
                timed(session, "clear_image", lambda: session.call("clear_image"))
            except Exception as exc:
                report["errors"].append(f"Cleanup {session.session_hash}: {exc}")
        try:
            report["after_cleanup"] = settle()
        except Exception as exc:
            report["monitor_errors"].append(str(exc))
        monitor.stop.set()
        monitor.thread.join(timeout=12)
        report["requests"] = latency_summary(records)
        for name, rows in (("gpu_memory.csv", monitor.rows), ("requests.csv", records)):
            if rows:
                with (args.output / name).open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        (args.output / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Reports saved to {args.output.resolve()}", flush=True)
        print("GPU totals include other processes; CuPy may retain freed memory in its pool.", flush=True)
    return 1 if report["errors"] or report["monitor_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
