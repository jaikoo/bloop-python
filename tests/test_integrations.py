"""Tests for bloop auto-instrumentation integrations."""

import unittest
from unittest.mock import MagicMock

from bloop.client import BloopClient
from bloop.integrations.costs import detect_provider, compute_cost, MODEL_COSTS


class TestProviderDetection(unittest.TestCase):
    def test_detect_openai(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        self.assertEqual(detect_provider(client), "openai")

    def test_detect_minimax(self):
        client = MagicMock()
        client.base_url = "https://api.minimax.io/v1"
        self.assertEqual(detect_provider(client), "minimax")

    def test_detect_minimax_alternate(self):
        client = MagicMock()
        client.base_url = "https://api.minimaxi.com/v1"
        self.assertEqual(detect_provider(client), "minimax")

    def test_detect_kimi(self):
        client = MagicMock()
        client.base_url = "https://api.moonshot.ai/v1"
        self.assertEqual(detect_provider(client), "kimi")

    def test_detect_anthropic(self):
        client = MagicMock()
        client.base_url = "https://api.anthropic.com/v1"
        self.assertEqual(detect_provider(client), "anthropic")

    def test_detect_google(self):
        client = MagicMock()
        client.base_url = "https://generativelanguage.googleapis.com/v1"
        self.assertEqual(detect_provider(client), "google")

    def test_detect_unknown_returns_hostname(self):
        client = MagicMock()
        client.base_url = "https://custom-llm.example.com/v1"
        self.assertEqual(detect_provider(client), "custom-llm.example.com")

    def test_detect_with_no_base_url(self):
        """Client with no base_url attribute should not crash."""
        client = MagicMock(spec=[])
        result = detect_provider(client)
        self.assertIsInstance(result, str)

    def test_detect_with_base_url_attribute(self):
        """Clients may store base_url in _base_url."""
        client = MagicMock(spec=[])
        client._base_url = "https://api.openai.com/v1"
        self.assertEqual(detect_provider(client), "openai")


class TestCostComputation(unittest.TestCase):
    def test_known_model_gpt4o(self):
        cost = compute_cost("gpt-4o", 1000, 500)
        expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        self.assertAlmostEqual(cost, expected)

    def test_known_model_gpt4o_mini(self):
        cost = compute_cost("gpt-4o-mini", 1000, 500)
        expected = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
        self.assertAlmostEqual(cost, expected)

    def test_known_model_claude_sonnet(self):
        cost = compute_cost("claude-sonnet-4-5-20250929", 1000, 500)
        expected = (1000 * 3.00 / 1_000_000) + (500 * 15.00 / 1_000_000)
        self.assertAlmostEqual(cost, expected)

    def test_known_model_minimax_m1(self):
        cost = compute_cost("MiniMax-M1", 1000, 500)
        expected = (1000 * 0.40 / 1_000_000) + (500 * 2.20 / 1_000_000)
        self.assertAlmostEqual(cost, expected)

    def test_known_model_kimi_k2(self):
        cost = compute_cost("kimi-k2", 1000, 500)
        expected = (1000 * 0.60 / 1_000_000) + (500 * 2.50 / 1_000_000)
        self.assertAlmostEqual(cost, expected)

    def test_unknown_model_returns_zero(self):
        cost = compute_cost("unknown-model", 1000, 500)
        self.assertEqual(cost, 0.0)

    def test_custom_costs_override(self):
        custom = {"my-model": {"input": 1.0 / 1_000_000, "output": 2.0 / 1_000_000}}
        cost = compute_cost("my-model", 1000, 500, custom)
        expected = (1000 * 1.0 / 1_000_000) + (500 * 2.0 / 1_000_000)
        self.assertAlmostEqual(cost, expected)

    def test_custom_costs_takes_priority(self):
        """Custom costs for a known model should override built-in pricing."""
        custom = {"gpt-4o": {"input": 99.0 / 1_000_000, "output": 99.0 / 1_000_000}}
        cost = compute_cost("gpt-4o", 1000, 500, custom)
        expected = (1000 * 99.0 / 1_000_000) + (500 * 99.0 / 1_000_000)
        self.assertAlmostEqual(cost, expected)

    def test_zero_tokens(self):
        cost = compute_cost("gpt-4o", 0, 0)
        self.assertEqual(cost, 0.0)

    def test_model_costs_dict_is_populated(self):
        """MODEL_COSTS should contain entries for multiple providers."""
        self.assertIn("gpt-4o", MODEL_COSTS)
        self.assertIn("claude-sonnet-4-5-20250929", MODEL_COSTS)
        self.assertIn("MiniMax-M1", MODEL_COSTS)
        self.assertIn("kimi-k2", MODEL_COSTS)


class TestWrapOpenAI(unittest.TestCase):
    def _make_bloop_client(self):
        return BloopClient("http://localhost:5332", "test-key", flush_interval=999)

    def _make_mock_openai(self, prompt_tokens=100, completion_tokens=50, content="Hello!"):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = prompt_tokens
        mock_response.usage.completion_tokens = completion_tokens
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content

        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client, mock_response

    def test_wrap_returns_same_client(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_openai()
        wrapped = bloop.wrap_openai(mock_client)
        self.assertIs(wrapped, mock_client)
        bloop.close()

    def test_wrap_traces_chat_completion(self):
        bloop = self._make_bloop_client()
        mock_client, mock_response = self._make_mock_openai()

        wrapped = bloop.wrap_openai(mock_client)
        result = wrapped.chat.completions.create(model="gpt-4o", messages=[])

        self.assertEqual(result, mock_response)
        self.assertEqual(len(bloop._trace_buffer), 1)
        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["name"], "openai.chat.completions.create")
        self.assertEqual(len(trace["spans"]), 1)
        self.assertEqual(trace["spans"][0]["model"], "gpt-4o")
        self.assertEqual(trace["spans"][0]["provider"], "openai")
        self.assertEqual(trace["spans"][0]["status"], "ok")
        bloop.close()

    def test_wrap_records_token_usage(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_openai(prompt_tokens=200, completion_tokens=100)

        bloop.wrap_openai(mock_client)
        mock_client.chat.completions.create(model="gpt-4o", messages=[])

        span = bloop._trace_buffer[0]["spans"][0]
        self.assertEqual(span["input_tokens"], 200)
        self.assertEqual(span["output_tokens"], 100)
        bloop.close()

    def test_wrap_records_cost(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_openai(prompt_tokens=1000, completion_tokens=500)

        bloop.wrap_openai(mock_client)
        mock_client.chat.completions.create(model="gpt-4o", messages=[])

        span = bloop._trace_buffer[0]["spans"][0]
        expected_cost = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        self.assertAlmostEqual(span["cost"], expected_cost)
        bloop.close()

    def test_wrap_records_output(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_openai(content="The answer is 42")

        bloop.wrap_openai(mock_client)
        mock_client.chat.completions.create(model="gpt-4o", messages=[])

        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["output"], "The answer is 42")
        self.assertEqual(trace["spans"][0]["output"], "The answer is 42")
        bloop.close()

    def test_wrap_handles_error(self):
        bloop = self._make_bloop_client()
        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"
        mock_client.chat.completions.create.side_effect = RuntimeError("API timeout")

        bloop.wrap_openai(mock_client)
        with self.assertRaises(RuntimeError):
            mock_client.chat.completions.create(model="gpt-4o", messages=[])

        self.assertEqual(len(bloop._trace_buffer), 1)
        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["status"], "error")
        self.assertEqual(trace["spans"][0]["status"], "error")
        self.assertIn("API timeout", trace["spans"][0]["error_message"])
        bloop.close()

    def test_wrap_with_minimax_provider(self):
        """OpenAI-compatible providers like Minimax should be auto-detected."""
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_openai()
        mock_client.base_url = "https://api.minimax.io/v1"

        bloop.wrap_openai(mock_client)
        mock_client.chat.completions.create(model="MiniMax-M1", messages=[])

        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["name"], "minimax.chat.completions.create")
        self.assertEqual(trace["spans"][0]["provider"], "minimax")
        bloop.close()

    def test_wrap_embeddings(self):
        bloop = self._make_bloop_client()
        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"

        # Mock chat
        mock_chat_response = MagicMock()
        mock_chat_response.usage.prompt_tokens = 10
        mock_chat_response.usage.completion_tokens = 5
        mock_chat_response.choices = [MagicMock()]
        mock_chat_response.choices[0].message.content = "hi"
        mock_client.chat.completions.create.return_value = mock_chat_response

        # Mock embeddings
        mock_emb_response = MagicMock()
        mock_emb_response.usage.prompt_tokens = 50
        mock_emb_response.usage.total_tokens = 50
        mock_client.embeddings.create.return_value = mock_emb_response

        bloop.wrap_openai(mock_client)
        mock_client.embeddings.create(model="text-embedding-3-small", input=["test"])

        self.assertEqual(len(bloop._trace_buffer), 1)
        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["name"], "openai.embeddings.create")
        self.assertEqual(trace["spans"][0]["input_tokens"], 50)
        bloop.close()


class TestWrapAnthropic(unittest.TestCase):
    def _make_bloop_client(self):
        return BloopClient("http://localhost:5332", "test-key", flush_interval=999)

    def _make_mock_anthropic(self, input_tokens=200, output_tokens=100, text="Hi there!"):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = input_tokens
        mock_response.usage.output_tokens = output_tokens
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = text

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        return mock_client, mock_response

    def test_wrap_returns_same_client(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_anthropic()
        wrapped = bloop.wrap_anthropic(mock_client)
        self.assertIs(wrapped, mock_client)
        bloop.close()

    def test_wrap_traces_messages_create(self):
        bloop = self._make_bloop_client()
        mock_client, mock_response = self._make_mock_anthropic()

        wrapped = bloop.wrap_anthropic(mock_client)
        result = wrapped.messages.create(model="claude-sonnet-4-5-20250929", messages=[])

        self.assertEqual(result, mock_response)
        self.assertEqual(len(bloop._trace_buffer), 1)
        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["name"], "anthropic.messages.create")
        self.assertEqual(trace["spans"][0]["provider"], "anthropic")
        self.assertEqual(trace["spans"][0]["model"], "claude-sonnet-4-5-20250929")
        bloop.close()

    def test_wrap_records_token_usage(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_anthropic(input_tokens=300, output_tokens=150)

        bloop.wrap_anthropic(mock_client)
        mock_client.messages.create(model="claude-sonnet-4-5-20250929", messages=[])

        span = bloop._trace_buffer[0]["spans"][0]
        self.assertEqual(span["input_tokens"], 300)
        self.assertEqual(span["output_tokens"], 150)
        bloop.close()

    def test_wrap_records_cost(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_anthropic(input_tokens=1000, output_tokens=500)

        bloop.wrap_anthropic(mock_client)
        mock_client.messages.create(model="claude-sonnet-4-5-20250929", messages=[])

        span = bloop._trace_buffer[0]["spans"][0]
        expected_cost = (1000 * 3.00 / 1_000_000) + (500 * 15.00 / 1_000_000)
        self.assertAlmostEqual(span["cost"], expected_cost)
        bloop.close()

    def test_wrap_records_output(self):
        bloop = self._make_bloop_client()
        mock_client, _ = self._make_mock_anthropic(text="The answer is 42")

        bloop.wrap_anthropic(mock_client)
        mock_client.messages.create(model="claude-sonnet-4-5-20250929", messages=[])

        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["output"], "The answer is 42")
        self.assertEqual(trace["spans"][0]["output"], "The answer is 42")
        bloop.close()

    def test_wrap_handles_error(self):
        bloop = self._make_bloop_client()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("Rate limited")

        bloop.wrap_anthropic(mock_client)
        with self.assertRaises(RuntimeError):
            mock_client.messages.create(model="claude-sonnet-4-5-20250929", messages=[])

        trace = bloop._trace_buffer[0]
        self.assertEqual(trace["status"], "error")
        self.assertEqual(trace["spans"][0]["status"], "error")
        self.assertIn("Rate limited", trace["spans"][0]["error_message"])
        bloop.close()


class TestSetModelCosts(unittest.TestCase):
    def test_set_model_costs_stores_values(self):
        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)
        bloop.set_model_costs("my-model", {"input": 0.001, "output": 0.002})
        self.assertEqual(bloop._custom_model_costs["my-model"]["input"], 0.001)
        self.assertEqual(bloop._custom_model_costs["my-model"]["output"], 0.002)
        bloop.close()

    def test_custom_costs_used_in_wrap(self):
        """Custom costs set on BloopClient should be used during auto-tracing."""
        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)
        bloop.set_model_costs("my-fine-tune", {"input": 5.0 / 1_000_000, "output": 15.0 / 1_000_000})

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 500
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"

        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"
        mock_client.chat.completions.create.return_value = mock_response

        bloop.wrap_openai(mock_client)
        mock_client.chat.completions.create(model="my-fine-tune", messages=[])

        span = bloop._trace_buffer[0]["spans"][0]
        expected_cost = (1000 * 5.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
        self.assertAlmostEqual(span["cost"], expected_cost)
        bloop.close()


class TestImports(unittest.TestCase):
    def test_import_model_costs_from_bloop(self):
        from bloop import MODEL_COSTS
        self.assertIsInstance(MODEL_COSTS, dict)
        self.assertIn("gpt-4o", MODEL_COSTS)

    def test_import_bloop_client_still_works(self):
        from bloop import BloopClient
        self.assertTrue(callable(BloopClient))

    def test_bloop_client_has_wrap_openai(self):
        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)
        self.assertTrue(hasattr(bloop, "wrap_openai"))
        self.assertTrue(callable(bloop.wrap_openai))
        bloop.close()

    def test_bloop_client_has_wrap_anthropic(self):
        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)
        self.assertTrue(hasattr(bloop, "wrap_anthropic"))
        self.assertTrue(callable(bloop.wrap_anthropic))
        bloop.close()

    def test_bloop_client_has_set_model_costs(self):
        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)
        self.assertTrue(hasattr(bloop, "set_model_costs"))
        self.assertTrue(callable(bloop.set_model_costs))
        bloop.close()


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing functionality is not broken by new integrations."""

    def test_client_init_has_custom_model_costs(self):
        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)
        self.assertIsInstance(bloop._custom_model_costs, dict)
        self.assertEqual(len(bloop._custom_model_costs), 0)
        bloop.close()

    def test_trace_still_works(self):
        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)
        trace = bloop.start_trace("manual-trace")
        span = trace.start_span(span_type="generation", model="gpt-4")
        span.end(input_tokens=10, output_tokens=5)
        trace.end()
        self.assertEqual(len(bloop._trace_buffer), 1)
        bloop.close()


class TestThreadSafety(unittest.TestCase):
    """Verify wrapping is safe to use from multiple threads."""

    def test_concurrent_calls(self):
        import threading

        bloop = BloopClient("http://localhost:5332", "test-key", flush_interval=999)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"
        mock_client.chat.completions.create.return_value = mock_response

        bloop.wrap_openai(mock_client)

        errors = []

        def call_llm():
            try:
                mock_client.chat.completions.create(model="gpt-4o", messages=[])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_llm) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(bloop._trace_buffer), 10)
        bloop.close()


if __name__ == "__main__":
    unittest.main()
