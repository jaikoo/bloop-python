"""Bloop error reporting client for Python."""

import hashlib
import hmac
import json
import sys
import threading
import urllib.request
from typing import Any, Optional


class BloopClient:
    """Captures and sends error events to a bloop server.

    Usage:
        client = BloopClient("https://bloop.example.com", "your-project-key")
        client.capture("TypeError", "something went wrong", stack="...")
        client.close()

    Or as a context manager:
        with BloopClient("https://bloop.example.com", "key") as client:
            client.capture("ValueError", "bad input")
    """

    def __init__(
        self,
        endpoint: str,
        project_key: str,
        *,
        flush_interval: float = 5.0,
        max_buffer_size: int = 100,
        environment: str = "production",
        release: str = "",
        auto_capture: bool = False,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.project_key = project_key
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.environment = environment
        self.release = release

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._closed = False
        self._timer: Optional[threading.Timer] = None
        self._original_excepthook = sys.excepthook

        if auto_capture:
            self._install_excepthook()

        self._schedule_flush()

    def _install_excepthook(self) -> None:
        original = sys.excepthook

        def hook(exc_type, exc_value, exc_tb):
            self.capture(
                error_type=exc_type.__name__,
                message=str(exc_value),
                source="python",
            )
            self.flush()
            original(exc_type, exc_value, exc_tb)

        sys.excepthook = hook

    def capture(
        self,
        error_type: str,
        message: str,
        *,
        source: str = "python",
        stack: str = "",
        route_or_procedure: str = "",
        screen: str = "",
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Buffer an error event for sending."""
        if self._closed:
            return

        import time

        event: dict[str, Any] = {
            "timestamp": int(time.time() * 1000),
            "source": source,
            "environment": self.environment,
            "error_type": error_type,
            "message": message,
        }
        if self.release:
            event["release"] = self.release
        if stack:
            event["stack"] = stack
        if route_or_procedure:
            event["route_or_procedure"] = route_or_procedure
        if screen:
            event["screen"] = screen
        if metadata:
            event["metadata"] = metadata

        # Allow extra kwargs to pass through
        for k, v in kwargs.items():
            if k not in event:
                event[k] = v

        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.max_buffer_size:
                self._flush_locked()

    def flush(self) -> None:
        """Send all buffered events immediately."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush buffer while holding the lock."""
        if not self._buffer:
            return

        events = self._buffer[:]
        self._buffer.clear()

        # Send in background to avoid blocking
        thread = threading.Thread(target=self._send, args=(events,), daemon=True)
        thread.start()

    def _send(self, events: list[dict[str, Any]]) -> None:
        """Send events to the bloop server."""
        if len(events) == 1:
            url = f"{self.endpoint}/v1/ingest"
            body = json.dumps(events[0]).encode("utf-8")
        else:
            url = f"{self.endpoint}/v1/ingest/batch"
            body = json.dumps({"events": events}).encode("utf-8")

        signature = hmac.new(
            self.project_key.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": signature,
                "X-Project-Key": self.project_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception:
            pass  # Fire and forget — don't crash the host app

    def _schedule_flush(self) -> None:
        """Schedule the next periodic flush."""
        if self._closed:
            return
        self._timer = threading.Timer(self.flush_interval, self._periodic_flush)
        self._timer.daemon = True
        self._timer.start()

    def _periodic_flush(self) -> None:
        """Called by the timer to flush and reschedule."""
        self.flush()
        self._schedule_flush()

    def close(self) -> None:
        """Flush remaining events and stop the background timer."""
        self._closed = True
        if self._timer:
            self._timer.cancel()
        self.flush()
        sys.excepthook = self._original_excepthook

    def __enter__(self) -> "BloopClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
