"""Check HTTP-owned TCP lifecycle and shared model identity without CUDA."""

import importlib.util
import json
import socket
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from sam3_service.eiseg_tcp import DELIMITER
from test_eiseg_tcp import FakeBackend


class CombinedServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "sam3_service" / "app.py"
        spec = importlib.util.spec_from_file_location("sam3_service.combined_service_test", path)
        cls.module = importlib.util.module_from_spec(spec)

        def register_routes(app):
            @app.get("/health")
            def health():
                return {"status": "healthy"}

        # Replace only heavy route imports; run the real FastAPI/TCP lifecycle below.
        with patch.dict(sys.modules, {"sam3_service.routes": SimpleNamespace(register_routes=register_routes)}):
            spec.loader.exec_module(cls.module)

    def test_http_and_tcp_share_backend_and_tcp_closes_with_http(self):
        backend = FakeBackend()
        runtime = SimpleNamespace(predictor=SimpleNamespace(interactive=backend))
        with patch.dict(sys.modules, {"sam3_service.trt_runtime": runtime}), \
                patch.dict("os.environ", {"SAM3_TCP_HOST": "127.0.0.1", "SAM3_TCP_PORT": "0"}):
            app = self.module.create_app()
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").json(), {"status": "healthy"})
                server = app.state.tcp_server
                self.assertIs(server.backend, backend)
                address = server.server_address
                with socket.create_connection(address, timeout=5) as connection:
                    connection.sendall(b'{"cmd":8004}' + DELIMITER)
                    response = bytearray()
                    while not response.endswith(DELIMITER):
                        response.extend(connection.recv(4096))
                    self.assertEqual(json.loads(response[:-len(DELIMITER)])["cmd"], 9004)
            with socket.socket() as probe:
                self.assertNotEqual(probe.connect_ex(address), 0)

    def test_occupied_tcp_port_fails_service_startup(self):
        runtime = SimpleNamespace(predictor=SimpleNamespace(interactive=FakeBackend()))
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            with patch.dict(sys.modules, {"sam3_service.trt_runtime": runtime}), \
                    patch.dict("os.environ", {
                        "SAM3_TCP_HOST": "127.0.0.1", "SAM3_TCP_PORT": str(occupied.getsockname()[1]),
                    }):
                with self.assertRaises(OSError):
                    with TestClient(self.module.create_app()):
                        pass


if __name__ == "__main__":
    unittest.main()
