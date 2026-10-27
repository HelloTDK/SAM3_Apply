"""Exercise the browser bridge against real sockets without loading GPU models."""

import json
import socket
import socketserver
import threading
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sam3_service.eiseg_tcp import DELIMITER, InteractiveSegmentationServer
from sam3_service.tcp_bridge import router
from test_eiseg_tcp import FakeBackend, click_request, image_request


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    with server:
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            thread.join()


class TCPBridgeTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def connect(self, port, **kwargs):
        return self.client.websocket_connect(f"/ui/tcp-test?port={port}", **kwargs)

    def exchange(self, websocket, request):
        websocket.send_json(request)
        message = websocket.receive_json()
        self.assertEqual(message["type"], "response")
        self.assertEqual(message["response"]["cmd"], request["cmd"] + 1000)
        return message["response"]["data"]

    def test_persistent_tcp_session_and_all_commands(self):
        server = InteractiveSegmentationServer(("127.0.0.1", 0), backend=FakeBackend())
        with running(server) as port, self.connect(port) as ws:
            self.assertEqual(ws.receive_json()["port"], port)
            self.assertEqual(self.exchange(ws, {"cmd": 8004})[0]["model"], 0)
            self.assertEqual(self.exchange(ws, {"cmd": 8000, "data": {"model": 0}})["status"], 1)
            self.assertEqual(self.exchange(ws, image_request())["status"], 1)
            self.assertEqual(self.exchange(ws, click_request(2, [20, 20]))["status"], 0)
            first = self.exchange(ws, click_request(1, [5, 5]))["ploygon"]
            self.assertTrue(first)
            self.exchange(ws, click_request(2, [20, 20]))
            self.assertEqual(self.exchange(ws, click_request(3))["ploygon"], first)
            self.assertEqual(self.exchange(ws, click_request(4))["ploygon"], first)
            self.assertEqual(self.exchange(ws, click_request(5))["ploygon"], [])

    def test_destination_port_is_used_and_connections_have_independent_state(self):
        backends = [FakeBackend(), FakeBackend()]
        with running(InteractiveSegmentationServer(("127.0.0.1", 0), backend=backends[0])) as first, \
                running(InteractiveSegmentationServer(("127.0.0.1", 0), backend=backends[1])) as second:
            for index, port in enumerate((first, second)):
                url = f"/ui/tcp-test?port={port}"
                if index == 0:
                    # Extra query fields cannot redirect the server-side bridge.
                    url += "&host=example.invalid"
                with self.client.websocket_connect(url) as ws:
                    connected = ws.receive_json()
                    self.assertEqual((connected["host"], connected["port"]), ("127.0.0.1", port))
                    self.assertEqual(self.exchange(ws, click_request(1, [5, 5]))["status"], 0)
                    self.exchange(ws, image_request())
                    self.exchange(ws, click_request(1, [5, 5]))
            self.assertEqual([len(backend.previous_logits) for backend in backends], [1, 1])

    def test_split_and_coalesced_tcp_replies(self):
        reply = {"cmd": 9004, "data": [{"model_name": "测试模型"}]}
        payload = json.dumps(reply, ensure_ascii=False).encode() + DELIMITER

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.recv(4096)
                self.request.sendall(payload[:2])
                self.request.sendall(payload[2:] + payload)

        with running(socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)) as port, self.connect(port) as ws:
            ws.receive_json()
            ws.send_json({"cmd": 8004})
            for _ in range(2):
                self.assertEqual(ws.receive_json(), {"type": "response", "response": reply})

    def test_browser_disconnect_closes_tcp_while_reply_is_stalled(self):
        received, closed = threading.Event(), threading.Event()

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(3)
                self.request.recv(4096)
                received.set()
                if self.request.recv(4096) == b"":
                    closed.set()

        with running(socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)) as port:
            with self.connect(port) as ws:
                ws.receive_json()
                ws.send_json({"cmd": 8004})
                self.assertTrue(received.wait(2))
            self.assertTrue(closed.wait(2), "Closing the browser must release its TCP connection promptly")

    def test_malformed_or_oversized_tcp_reply_reports_error(self):
        for payload in (b"broken" + DELIMITER, b"[1]" + DELIMITER, b"partial", b"x" * 257):
            with self.subTest(payload=payload[:10]):
                class Handler(socketserver.BaseRequestHandler):
                    def handle(self):
                        self.request.sendall(payload)

                with patch("sam3_service.tcp_bridge.MAX_FRAME_BYTES", 256), \
                        running(socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)) as port, self.connect(port) as ws:
                    ws.receive_json()
                    self.assertEqual(ws.receive_json()["type"], "error")

    def test_invalid_requests_cannot_inject_arbitrary_tcp_messages(self):
        server = InteractiveSegmentationServer(("127.0.0.1", 0), backend=FakeBackend())
        with running(server) as port:
            for payload in ('[]', '{', '{"cmd":22}', '{"cmd":true}',
                            '{"cmd":8004,"data":[]}', '{"cmd":8004,"data":{"x":"##@##"}}'):
                with self.subTest(payload=payload), self.connect(port) as ws:
                    ws.receive_json()
                    ws.send_text(payload)
                    self.assertEqual(ws.receive_json()["type"], "error")

    def test_binary_and_oversized_browser_requests_are_rejected(self):
        server = InteractiveSegmentationServer(("127.0.0.1", 0), backend=FakeBackend())
        with running(server) as port, patch("sam3_service.tcp_bridge.MAX_FRAME_BYTES", 256):
            for payload in (b"binary", " " * 257):
                with self.connect(port) as ws:
                    ws.receive_json()
                    if isinstance(payload, bytes):
                        ws.send_bytes(payload)
                    else:
                        ws.send_text(payload)
                    self.assertEqual(ws.receive_json()["type"], "error")

    def test_invalid_destination_and_connection_failure_are_reported(self):
        for query in ("port=0", "port=65536"):
            with self.client.websocket_connect(f"/ui/tcp-test?{query}") as ws:
                self.assertEqual(ws.receive_json()["type"], "error")
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            # A bound, non-listening port reliably refuses a connection.
            with self.connect(occupied.getsockname()[1]) as ws:
                self.assertEqual(ws.receive_json()["type"], "error")

    def test_cross_origin_browser_is_rejected(self):
        for origin in ("https://other.example", "null", "ftp://testserver", "http://[invalid"):
            with self.subTest(origin=origin), self.assertRaises(WebSocketDisconnect) as rejected:
                with self.connect(6063, headers={"origin": origin}):
                    pass
            self.assertEqual(rejected.exception.code, 1008)
        with self.client.websocket_connect("/ui/tcp-test?port=0", headers={"origin": "http://testserver"}) as ws:
            self.assertEqual(ws.receive_json()["type"], "error")


if __name__ == "__main__":
    unittest.main()
