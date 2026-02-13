# @bloop/python

Python SDK for [bloop](https://github.com/your-org/bloop) error observability. Zero dependencies — uses only the Python standard library.

## Install

```bash
pip install bloop-sdk
```

## Usage

```python
from bloop import BloopClient

client = BloopClient(
    "https://bloop.example.com",
    "your-project-api-key",
    environment="production",
    release="1.0.0",
)

# Capture an error
client.capture("ValueError", "invalid input", stack="Traceback ...")

# Or use as context manager for automatic cleanup
with BloopClient("https://bloop.example.com", "key") as client:
    client.capture("TypeError", "oops")

# Auto-capture unhandled exceptions
client = BloopClient("...", "key", auto_capture=True)
```

## API

### `BloopClient(endpoint, project_key, **opts)`

- `flush_interval` (float): Seconds between automatic flushes. Default: `5.0`
- `max_buffer_size` (int): Flush when buffer reaches this size. Default: `100`
- `environment` (str): Environment tag. Default: `"production"`
- `release` (str): Release version tag. Default: `""`
- `auto_capture` (bool): Install `sys.excepthook` to capture unhandled exceptions. Default: `False`

### `client.capture(error_type, message, **kwargs)`

Buffer an error event. Keyword arguments: `source`, `stack`, `route_or_procedure`, `screen`, `metadata`.

### `client.flush()`

Send all buffered events immediately.

### `client.close()`

Flush and stop background timer.
