"""OpenAI-compatible auto-instrumentation wrapper."""

import functools
from typing import Any


def wrap_openai_client(bloop_client: Any, openai_client: Any) -> Any:
    """Wrap an OpenAI-compatible client to auto-trace all LLM calls.

    Works with: openai.OpenAI(), and any OpenAI-compatible provider
    (Minimax, Kimi, etc.) that uses the same API shape.
    """
    from bloop.integrations.costs import compute_cost, detect_provider

    provider = detect_provider(openai_client)

    # Wrap chat.completions.create
    original_chat_create = openai_client.chat.completions.create

    @functools.wraps(original_chat_create)
    def traced_chat_create(*args: Any, **kwargs: Any) -> Any:
        params = kwargs if kwargs else (args[0] if args else {})
        model = (
            params.get("model", "unknown")
            if isinstance(params, dict)
            else getattr(params, "model", "unknown")
        )
        trace_name = f"{provider}.chat.completions.create"

        trace = bloop_client.start_trace(name=trace_name)
        span = trace.start_span(
            span_type="generation",
            name=trace_name,
            model=model,
            provider=provider,
        )

        try:
            response = original_chat_create(*args, **kwargs)

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            cost = compute_cost(
                model,
                input_tokens,
                output_tokens,
                getattr(bloop_client, "_custom_model_costs", None),
            )

            output_text = None
            try:
                output_text = response.choices[0].message.content
            except (IndexError, AttributeError):
                pass

            span.end(
                status="ok",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                output=output_text,
            )
            trace.end(status="completed", output=output_text)
            return response
        except Exception as e:
            span.end(status="error", error_message=str(e))
            trace.end(status="error")
            raise

    # Monkey-patch the create method
    openai_client.chat.completions.create = traced_chat_create

    # Wrap embeddings.create if it exists
    if hasattr(openai_client, "embeddings") and hasattr(
        openai_client.embeddings, "create"
    ):
        original_emb_create = openai_client.embeddings.create

        @functools.wraps(original_emb_create)
        def traced_emb_create(*args: Any, **kwargs: Any) -> Any:
            params = kwargs if kwargs else (args[0] if args else {})
            model = (
                params.get("model", "unknown")
                if isinstance(params, dict)
                else "unknown"
            )
            trace_name = f"{provider}.embeddings.create"

            trace = bloop_client.start_trace(name=trace_name)
            span = trace.start_span(
                span_type="generation",
                name=trace_name,
                model=model,
                provider=provider,
            )

            try:
                response = original_emb_create(*args, **kwargs)
                usage = getattr(response, "usage", None)
                input_tokens = (
                    getattr(usage, "prompt_tokens", 0)
                    or getattr(usage, "total_tokens", 0)
                    or 0
                )
                cost = compute_cost(
                    model,
                    input_tokens,
                    0,
                    getattr(bloop_client, "_custom_model_costs", None),
                )
                span.end(status="ok", input_tokens=input_tokens, cost=cost)
                trace.end(status="completed")
                return response
            except Exception as e:
                span.end(status="error", error_message=str(e))
                trace.end(status="error")
                raise

        openai_client.embeddings.create = traced_emb_create

    return openai_client
