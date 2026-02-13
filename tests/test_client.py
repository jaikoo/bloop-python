"""Tests for the bloop Python SDK client."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from bloop import BloopClient


class CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures requests for testing."""

    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        CaptureHandler.received.append(
            {
                "path": self.path,
                "body": json.loads(body),
                "headers": dict(self.headers),
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"accepted"}')

    def log_message(self, format, *args):
        pass  # Suppress output


class TestBloopClient(unittest.TestCase):
    server: HTTPServer
    thread: threading.Thread

    @classmethod
    def setUpClass(cls):
        CaptureHandler.received = []
        cls.server = HTTPServer(("127.0.0.1", 0), CaptureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    @property
    def base_url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def setUp(self):
        CaptureHandler.received.clear()

    def test_capture_and_flush(self):
        client = BloopClient(
            self.base_url,
            "test-key",
            flush_interval=999,
            environment="test",
            release="1.0.0",
        )
        client.capture("TypeError", "test error")
        client.flush()
        client.close()

        # Wait for background send thread
        import time
        time.sleep(0.5)

        self.assertEqual(len(CaptureHandler.received), 1)
        req = CaptureHandler.received[0]
        self.assertEqual(req["path"], "/v1/ingest")
        self.assertEqual(req["body"]["error_type"], "TypeError")
        self.assertEqual(req["body"]["message"], "test error")
        self.assertEqual(req["body"]["environment"], "test")
        self.assertIn("X-Signature", req["headers"])
        self.assertIn("X-Project-Key", req["headers"])

    def test_batch_flush(self):
        client = BloopClient(
            self.base_url,
            "test-key",
            flush_interval=999,
        )
        client.capture("Error1", "msg1")
        client.capture("Error2", "msg2")
        client.flush()
        client.close()

        import time
        time.sleep(0.5)

        self.assertEqual(len(CaptureHandler.received), 1)
        req = CaptureHandler.received[0]
        self.assertEqual(req["path"], "/v1/ingest/batch")
        self.assertEqual(len(req["body"]["events"]), 2)

    def test_context_manager(self):
        with BloopClient(self.base_url, "test-key", flush_interval=999) as client:
            client.capture("TestError", "ctx manager test")

        import time
        time.sleep(0.5)

        self.assertEqual(len(CaptureHandler.received), 1)

    def test_hmac_signature(self):
        import hashlib
        import hmac

        client = BloopClient(self.base_url, "my-secret-key", flush_interval=999)
        client.capture("SignTest", "verify hmac")
        client.flush()
        client.close()

        import time
        time.sleep(0.5)

        req = CaptureHandler.received[0]
        body_bytes = json.dumps(req["body"]).encode("utf-8")
        expected = hmac.new(
            b"my-secret-key", body_bytes, hashlib.sha256
        ).hexdigest()
        self.assertEqual(req["headers"]["X-Signature"], expected)

    def test_no_send_after_close(self):
        client = BloopClient(self.base_url, "test-key", flush_interval=999)
        client.close()
        client.capture("Ignored", "should not send")
        client.flush()

        import time
        time.sleep(0.5)

        self.assertEqual(len(CaptureHandler.received), 0)


if __name__ == "__main__":
    unittest.main()
