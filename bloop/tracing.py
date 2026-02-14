"""LLM trace and span types for bloop SDK."""

import time
import uuid
from typing import Any, Optional


class Span:
    """A span represents a single operation within a trace (e.g., LLM generation, tool call).

    Can be used as a context manager:
        with trace.generation(model="gpt-4") as span:
            result = call_llm(...)
            span.set_usage(input_tokens=10, output_tokens=20)
            span.set_output(result)
    """

    def __init__(
        self,
        span_type: str,
        name: str = "",
        model: str = "",
        provider: str = "",
        input: str = "",
        metadata: Optional[dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.parent_span_id: Optional[str] = parent_span_id
        self.span_type: str = span_type  # "generation", "tool", "retrieval", "custom"
        self.name: str = name
        self.model: str = model
        self.provider: str = provider
        self.input: str = input
        self.metadata: Optional[dict[str, Any]] = metadata
        self.started_at: int = int(time.time() * 1000)

        # Set on end()
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.cost: Optional[float] = None  # dollars
        self.latency_ms: Optional[int] = None
        self.time_to_first_token_ms: Optional[int] = None
        self.status: Optional[str] = None  # "ok" or "error"
        self.error_message: Optional[str] = None
        self.output: Optional[str] = None

    def end(
        self,
        status: str = "ok",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        error_message: Optional[str] = None,
        output: Optional[str] = None,
        time_to_first_token_ms: Optional[int] = None,
    ) -> None:
        """End the span, computing latency and recording final state."""
        self.latency_ms = int(time.time() * 1000) - self.started_at
        self.status = status
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        if cost is not None:
            self.cost = cost
        if error_message is not None:
            self.error_message = error_message
        if output is not None:
            self.output = output
        if time_to_first_token_ms is not None:
            self.time_to_first_token_ms = time_to_first_token_ms

    def set_usage(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        """Set token usage and cost on the span."""
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        if cost is not None:
            self.cost = cost

    def set_output(self, output: str) -> None:
        """Set the output text on the span."""
        self.output = output

    def to_dict(self) -> dict[str, Any]:
        """Serialize the span to a dictionary, omitting unset optional fields."""
        d: dict[str, Any] = {
            "id": self.id,
            "span_type": self.span_type,
            "name": self.name,
            "started_at": self.started_at,
            "status": self.status or "ok",
        }
        if self.parent_span_id:
            d["parent_span_id"] = self.parent_span_id
        if self.model:
            d["model"] = self.model
        if self.provider:
            d["provider"] = self.provider
        if self.input_tokens is not None:
            d["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            d["output_tokens"] = self.output_tokens
        if self.cost is not None:
            d["cost"] = self.cost
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.time_to_first_token_ms is not None:
            d["time_to_first_token_ms"] = self.time_to_first_token_ms
        if self.error_message:
            d["error_message"] = self.error_message
        if self.input:
            d["input"] = self.input
        if self.output:
            d["output"] = self.output
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type:
            self.end(status="error", error_message=str(exc_val) if exc_val else "error")
        elif self.status is None:
            self.end(status="ok")
        return False


class Trace:
    """A trace represents a complete LLM interaction, containing one or more spans.

    Can be used as a context manager:
        with client.trace("chatbot-request") as trace:
            with trace.generation(model="gpt-4") as span:
                ...
    """

    def __init__(
        self,
        client: Any,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        input: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self._client = client
        self.name: str = name
        self.session_id: Optional[str] = session_id
        self.user_id: Optional[str] = user_id
        self.status: str = "running"
        self.input: Optional[str] = input
        self.output: Optional[str] = None
        self.metadata: Optional[dict[str, Any]] = metadata
        self.prompt_name: Optional[str] = prompt_name
        self.prompt_version: Optional[str] = prompt_version
        self.started_at: int = int(time.time() * 1000)
        self.ended_at: Optional[int] = None
        self.spans: list[Span] = []

    def start_span(
        self,
        span_type: str = "custom",
        name: str = "",
        model: str = "",
        provider: str = "",
        input: str = "",
        metadata: Optional[dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """Create and register a new span within this trace."""
        span = Span(
            span_type=span_type,
            name=name,
            model=model,
            provider=provider,
            input=input,
            metadata=metadata,
            parent_span_id=parent_span_id,
        )
        self.spans.append(span)
        return span

    def generation(
        self,
        model: str = "",
        provider: str = "",
        name: str = "",
        input: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        """Convenience: start a generation span (usable as context manager)."""
        return self.start_span(
            span_type="generation",
            name=name,
            model=model,
            provider=provider,
            input=input,
            metadata=metadata,
        )

    def end(self, status: str = "completed", output: Optional[str] = None) -> None:
        """End the trace and enqueue it for sending."""
        self.ended_at = int(time.time() * 1000)
        self.status = status
        if output is not None:
            self.output = output
        self._client._enqueue_trace(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a dictionary, omitting unset optional fields."""
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "spans": [s.to_dict() for s in self.spans],
        }
        if self.session_id:
            d["session_id"] = self.session_id
        if self.user_id:
            d["user_id"] = self.user_id
        if self.input:
            d["input"] = self.input
        if self.output:
            d["output"] = self.output
        if self.metadata:
            d["metadata"] = self.metadata
        if self.prompt_name:
            d["prompt_name"] = self.prompt_name
        if self.prompt_version:
            d["prompt_version"] = self.prompt_version
        if self.ended_at:
            d["ended_at"] = self.ended_at
        return d

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type:
            self.end(status="error", output=str(exc_val) if exc_val else None)
        elif self.status == "running":
            self.end(status="completed")
        return False
