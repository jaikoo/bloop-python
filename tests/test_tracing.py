"""Tests for bloop LLM tracing support."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from bloop.tracing import Span, Trace


class MockClient:
    """Minimal mock of BloopClient for unit testing Trace/Span without network."""

    def __init__(self):
        self._lock = threading.Lock()
        self._trace_buffer: list[dict] = []

    def _enqueue_trace(self, trace) -> None:
        with self._lock:
            self._trace_buffer.append(trace.to_dict())


class TestSpanCreation(unittest.TestCase):
    def test_span_has_id(self):
        span = Span(span_type="generation", name="llm-call")
        self.assertIsInstance(span.id, str)
        self.assertTrue(len(span.id) > 0)

    def test_span_has_correct_type(self):
        span = Span(span_type="generation")
        self.assertEqual(span.span_type, "generation")

    def test_span_has_started_at(self):
        before = int(time.time() * 1000)
        span = Span(span_type="tool")
        after = int(time.time() * 1000)
        self.assertGreaterEqual(span.started_at, before)
        self.assertLessEqual(span.started_at, after)

    def test_span_default_status_is_none(self):
        span = Span(span_type="custom")
        self.assertIsNone(span.status)

    def test_span_stores_model_and_provider(self):
        span = Span(span_type="generation", model="gpt-4", provider="openai")
        self.assertEqual(span.model, "gpt-4")
        self.assertEqual(span.provider, "openai")

    def test_span_stores_input(self):
        span = Span(span_type="generation", input="Hello world")
        self.assertEqual(span.input, "Hello world")

    def test_span_stores_metadata(self):
        span = Span(span_type="custom", metadata={"key": "value"})
        self.assertEqual(span.metadata, {"key": "value"})

    def test_span_stores_parent_span_id(self):
        span = Span(span_type="tool", parent_span_id="parent-123")
        self.assertEqual(span.parent_span_id, "parent-123")


class TestSpanEnd(unittest.TestCase):
    def test_end_sets_latency_ms(self):
        span = Span(span_type="generation")
        time.sleep(0.01)
        span.end()
        self.assertIsNotNone(span.latency_ms)
        self.assertGreater(span.latency_ms, 0)

    def test_end_sets_status_ok_by_default(self):
        span = Span(span_type="generation")
        span.end()
        self.assertEqual(span.status, "ok")

    def test_end_sets_status_error(self):
        span = Span(span_type="generation")
        span.end(status="error", error_message="timeout")
        self.assertEqual(span.status, "error")
        self.assertEqual(span.error_message, "timeout")

    def test_end_sets_token_counts(self):
        span = Span(span_type="generation")
        span.end(input_tokens=100, output_tokens=50)
        self.assertEqual(span.input_tokens, 100)
        self.assertEqual(span.output_tokens, 50)

    def test_end_sets_cost(self):
        span = Span(span_type="generation")
        span.end(cost=0.003)
        self.assertEqual(span.cost, 0.003)

    def test_end_sets_output(self):
        span = Span(span_type="generation")
        span.end(output="The answer is 42")
        self.assertEqual(span.output, "The answer is 42")

    def test_end_sets_time_to_first_token(self):
        span = Span(span_type="generation")
        span.end(time_to_first_token_ms=150)
        self.assertEqual(span.time_to_first_token_ms, 150)


class TestSpanSetUsage(unittest.TestCase):
    def test_set_usage(self):
        span = Span(span_type="generation")
        span.set_usage(input_tokens=200, output_tokens=100, cost=0.01)
        self.assertEqual(span.input_tokens, 200)
        self.assertEqual(span.output_tokens, 100)
        self.assertEqual(span.cost, 0.01)

    def test_set_usage_partial(self):
        span = Span(span_type="generation")
        span.set_usage(input_tokens=50)
        self.assertEqual(span.input_tokens, 50)
        self.assertIsNone(span.output_tokens)

    def test_set_output(self):
        span = Span(span_type="generation")
        span.set_output("response text")
        self.assertEqual(span.output, "response text")


class TestSpanSerialization(unittest.TestCase):
    def test_to_dict_has_required_keys(self):
        span = Span(span_type="generation", name="test-span")
        span.end()
        d = span.to_dict()
        self.assertIn("id", d)
        self.assertIn("span_type", d)
        self.assertIn("name", d)
        self.assertIn("started_at", d)
        self.assertIn("status", d)

    def test_to_dict_includes_model_when_set(self):
        span = Span(span_type="generation", model="claude-3", provider="anthropic")
        d = span.to_dict()
        self.assertEqual(d["model"], "claude-3")
        self.assertEqual(d["provider"], "anthropic")

    def test_to_dict_excludes_none_optional_fields(self):
        span = Span(span_type="custom", name="bare")
        d = span.to_dict()
        self.assertNotIn("model", d)
        self.assertNotIn("provider", d)
        self.assertNotIn("input_tokens", d)
        self.assertNotIn("output_tokens", d)
        self.assertNotIn("cost", d)
        self.assertNotIn("latency_ms", d)
        self.assertNotIn("error_message", d)
        self.assertNotIn("parent_span_id", d)

    def test_to_dict_includes_tokens_after_end(self):
        span = Span(span_type="generation")
        span.end(input_tokens=10, output_tokens=20, cost=0.001)
        d = span.to_dict()
        self.assertEqual(d["input_tokens"], 10)
        self.assertEqual(d["output_tokens"], 20)
        self.assertEqual(d["cost"], 0.001)

    def test_to_dict_includes_metadata(self):
        span = Span(span_type="custom", metadata={"temperature": 0.7})
        d = span.to_dict()
        self.assertEqual(d["metadata"], {"temperature": 0.7})

    def test_to_dict_includes_input_and_output(self):
        span = Span(span_type="generation", input="prompt text")
        span.end(output="response text")
        d = span.to_dict()
        self.assertEqual(d["input"], "prompt text")
        self.assertEqual(d["output"], "response text")


class TestSpanContextManager(unittest.TestCase):
    def test_success_auto_ends_with_ok(self):
        span = Span(span_type="generation")
        with span:
            pass
        self.assertEqual(span.status, "ok")
        self.assertIsNotNone(span.latency_ms)

    def test_error_auto_ends_with_error(self):
        span = Span(span_type="generation")
        with self.assertRaises(ValueError):
            with span:
                raise ValueError("bad input")
        self.assertEqual(span.status, "error")
        self.assertEqual(span.error_message, "bad input")

    def test_does_not_suppress_exception(self):
        span = Span(span_type="generation")
        with self.assertRaises(RuntimeError):
            with span:
                raise RuntimeError("boom")

    def test_does_not_double_end(self):
        span = Span(span_type="generation")
        with span:
            span.end(status="ok", output="manual end")
        # Status should remain from the manual end, not overwritten
        self.assertEqual(span.status, "ok")
        self.assertEqual(span.output, "manual end")


class TestTraceCreation(unittest.TestCase):
    def test_trace_has_id(self):
        client = MockClient()
        trace = Trace(client=client, name="my-trace")
        self.assertIsInstance(trace.id, str)
        self.assertTrue(len(trace.id) > 0)

    def test_trace_has_name(self):
        client = MockClient()
        trace = Trace(client=client, name="chatbot-request")
        self.assertEqual(trace.name, "chatbot-request")

    def test_trace_has_started_at(self):
        before = int(time.time() * 1000)
        client = MockClient()
        trace = Trace(client=client, name="t")
        after = int(time.time() * 1000)
        self.assertGreaterEqual(trace.started_at, before)
        self.assertLessEqual(trace.started_at, after)

    def test_trace_status_is_running(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        self.assertEqual(trace.status, "running")

    def test_trace_stores_optional_fields(self):
        client = MockClient()
        trace = Trace(
            client=client,
            name="t",
            session_id="sess-1",
            user_id="user-1",
            input="hello",
            metadata={"key": "val"},
            prompt_name="greeting",
            prompt_version="v2",
        )
        self.assertEqual(trace.session_id, "sess-1")
        self.assertEqual(trace.user_id, "user-1")
        self.assertEqual(trace.input, "hello")
        self.assertEqual(trace.metadata, {"key": "val"})
        self.assertEqual(trace.prompt_name, "greeting")
        self.assertEqual(trace.prompt_version, "v2")


class TestTraceEnd(unittest.TestCase):
    def test_end_sets_ended_at(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        trace.end()
        self.assertIsNotNone(trace.ended_at)

    def test_end_sets_status_completed(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        trace.end()
        self.assertEqual(trace.status, "completed")

    def test_end_sets_custom_status(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        trace.end(status="error")
        self.assertEqual(trace.status, "error")

    def test_end_sets_output(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        trace.end(output="result text")
        self.assertEqual(trace.output, "result text")

    def test_end_enqueues_to_client(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        trace.end()
        self.assertEqual(len(client._trace_buffer), 1)


class TestTraceSpans(unittest.TestCase):
    def test_start_span(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        span = trace.start_span(span_type="generation", model="gpt-4")
        self.assertIsInstance(span, Span)
        self.assertEqual(span.span_type, "generation")
        self.assertEqual(span.model, "gpt-4")
        self.assertEqual(len(trace.spans), 1)

    def test_generation_convenience(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        span = trace.generation(model="claude-3", provider="anthropic")
        self.assertEqual(span.span_type, "generation")
        self.assertEqual(span.model, "claude-3")
        self.assertEqual(span.provider, "anthropic")

    def test_multiple_spans(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        trace.start_span(span_type="retrieval")
        trace.start_span(span_type="generation")
        trace.start_span(span_type="tool")
        self.assertEqual(len(trace.spans), 3)


class TestTraceSerialization(unittest.TestCase):
    def test_to_dict_has_required_keys(self):
        client = MockClient()
        trace = Trace(client=client, name="my-trace")
        trace.end()
        d = trace.to_dict()
        self.assertIn("id", d)
        self.assertIn("name", d)
        self.assertIn("status", d)
        self.assertIn("started_at", d)
        self.assertIn("spans", d)

    def test_to_dict_includes_spans(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        span = trace.start_span(span_type="generation")
        span.end()
        trace.end()
        d = trace.to_dict()
        self.assertEqual(len(d["spans"]), 1)
        self.assertEqual(d["spans"][0]["span_type"], "generation")

    def test_to_dict_includes_optional_fields_when_set(self):
        client = MockClient()
        trace = Trace(
            client=client,
            name="t",
            session_id="s1",
            user_id="u1",
            input="hello",
            metadata={"k": "v"},
            prompt_name="p",
            prompt_version="v1",
        )
        trace.end(output="bye")
        d = trace.to_dict()
        self.assertEqual(d["session_id"], "s1")
        self.assertEqual(d["user_id"], "u1")
        self.assertEqual(d["input"], "hello")
        self.assertEqual(d["output"], "bye")
        self.assertEqual(d["metadata"], {"k": "v"})
        self.assertEqual(d["prompt_name"], "p")
        self.assertEqual(d["prompt_version"], "v1")
        self.assertIn("ended_at", d)

    def test_to_dict_excludes_unset_optional_fields(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        d = trace.to_dict()
        self.assertNotIn("session_id", d)
        self.assertNotIn("user_id", d)
        self.assertNotIn("input", d)
        self.assertNotIn("output", d)
        self.assertNotIn("metadata", d)
        self.assertNotIn("prompt_name", d)
        self.assertNotIn("prompt_version", d)
        self.assertNotIn("ended_at", d)


class TestTraceContextManager(unittest.TestCase):
    def test_success_auto_ends_completed(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        with trace:
            pass
        self.assertEqual(trace.status, "completed")
        self.assertIsNotNone(trace.ended_at)
        self.assertEqual(len(client._trace_buffer), 1)

    def test_error_auto_ends_with_error(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        with self.assertRaises(ValueError):
            with trace:
                raise ValueError("oops")
        self.assertEqual(trace.status, "error")
        self.assertEqual(trace.output, "oops")
        self.assertEqual(len(client._trace_buffer), 1)

    def test_does_not_suppress_exception(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        with self.assertRaises(RuntimeError):
            with trace:
                raise RuntimeError("crash")

    def test_does_not_double_end(self):
        client = MockClient()
        trace = Trace(client=client, name="t")
        with trace:
            trace.end(status="completed", output="manual")
        # Should not overwrite since status is no longer "running"
        self.assertEqual(trace.output, "manual")
        # Should only have been enqueued once
        self.assertEqual(len(client._trace_buffer), 1)


class TestBloopClientTraceIntegration(unittest.TestCase):
    """Test that BloopClient properly manages trace buffer."""

    def test_start_trace_returns_trace(self):
        from bloop import BloopClient

        client = BloopClient("http://localhost:9999", "key", flush_interval=999)
        trace = client.start_trace("test-trace")
        self.assertIsInstance(trace, Trace)
        self.assertEqual(trace.name, "test-trace")
        client.close()

    def test_trace_alias(self):
        from bloop import BloopClient

        client = BloopClient("http://localhost:9999", "key", flush_interval=999)
        trace = client.trace("test-trace", session_id="s1")
        self.assertIsInstance(trace, Trace)
        self.assertEqual(trace.session_id, "s1")
        client.close()

    def test_trace_buffer_populated_after_end(self):
        from bloop import BloopClient

        client = BloopClient("http://localhost:9999", "key", flush_interval=999)
        trace = client.start_trace("t")
        trace.end()
        self.assertEqual(len(client._trace_buffer), 1)
        client.close()

    def test_flush_clears_trace_buffer(self):
        from bloop import BloopClient

        # Use a mock to avoid actual network calls
        client = BloopClient("http://localhost:9999", "key", flush_interval=999)
        trace = client.start_trace("t")
        trace.end()
        self.assertEqual(len(client._trace_buffer), 1)
        # Flush should clear it (even if send fails)
        with client._lock:
            client._flush_traces_locked()
        self.assertEqual(len(client._trace_buffer), 0)
        client.close()


class TraceCaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures trace batch requests."""

    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        TraceCaptureHandler.received.append(
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
        pass


class TestTraceNetworkSend(unittest.TestCase):
    """Test that traces are actually sent to the server."""

    server: HTTPServer
    thread: threading.Thread

    @classmethod
    def setUpClass(cls):
        TraceCaptureHandler.received = []
        cls.server = HTTPServer(("127.0.0.1", 0), TraceCaptureHandler)
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
        TraceCaptureHandler.received.clear()

    def test_trace_sent_to_server(self):
        from bloop import BloopClient

        client = BloopClient(self.base_url, "trace-key", flush_interval=999)
        trace = client.start_trace("chatbot", session_id="s1", user_id="u1")
        span = trace.generation(model="gpt-4", provider="openai", input="hi")
        span.end(input_tokens=10, output_tokens=20, output="hello!")
        trace.end(output="hello!")
        client.flush()
        client.close()

        time.sleep(0.5)

        self.assertGreaterEqual(len(TraceCaptureHandler.received), 1)
        # Find the trace batch request
        trace_reqs = [r for r in TraceCaptureHandler.received if r["path"] == "/v1/traces/batch"]
        self.assertEqual(len(trace_reqs), 1)
        req = trace_reqs[0]
        self.assertIn("traces", req["body"])
        self.assertEqual(len(req["body"]["traces"]), 1)
        t = req["body"]["traces"][0]
        self.assertEqual(t["name"], "chatbot")
        self.assertEqual(t["session_id"], "s1")
        self.assertEqual(len(t["spans"]), 1)
        self.assertEqual(t["spans"][0]["model"], "gpt-4")
        # Verify HMAC
        self.assertIn("X-Signature", req["headers"])
        self.assertIn("X-Project-Key", req["headers"])

    def test_existing_error_tracking_still_works(self):
        """Ensure adding tracing doesn't break existing error capture."""
        from bloop import BloopClient

        client = BloopClient(self.base_url, "test-key", flush_interval=999)
        client.capture("TypeError", "something broke")
        client.flush()
        client.close()

        time.sleep(0.5)

        error_reqs = [r for r in TraceCaptureHandler.received if r["path"] == "/v1/ingest"]
        self.assertEqual(len(error_reqs), 1)
        self.assertEqual(error_reqs[0]["body"]["error_type"], "TypeError")


class TestImports(unittest.TestCase):
    """Verify that tracing types are properly exported."""

    def test_import_trace_from_bloop(self):
        from bloop import Trace
        self.assertTrue(callable(Trace))

    def test_import_span_from_bloop(self):
        from bloop import Span
        self.assertTrue(callable(Span))


if __name__ == "__main__":
    unittest.main()
