"""Anthropic auto-instrumentation wrapper."""

import functools
from typing import Any


def wrap_anthropic_client(bloop_client: Any, anthropic_client: Any) -> Any:
    """Wrap an Anthropic client to auto-trace messages.create() calls."""
    from bloop.integrations.costs import compute_cost

    original_create = anthropic_client.messages.create

    @functools.wraps(original_create)
    def traced_create(*args: Any, **kwargs: Any) -> Any:
        params = kwargs if kwargs else (args[0] if args else {})
        model = (
            params.get("model", "unknown")
            if isinstance(params, dict)
            else getattr(params, "model", "unknown")
        )

        trace = bloop_client.start_trace(name="anthropic.messages.create")
        span = trace.start_span(
            span_type="generation",
            name="anthropic.messages.create",
            model=model,
            provider="anthropic",
        )

        try:
            response = original_create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cost = compute_cost(
                model,
                input_tokens,
                output_tokens,
                getattr(bloop_client, "_custom_model_costs", None),
            )
            output_text = None
            try:
                output_text = response.content[0].text
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

    anthropic_client.messages.create = traced_create
    return anthropic_client
