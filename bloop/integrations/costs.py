"""Model pricing table for cost computation."""

# Costs per token in dollars
MODEL_COSTS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    # Anthropic
    "claude-sonnet-4-5-20250929": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    # Minimax
    "MiniMax-M1": {"input": 0.40 / 1_000_000, "output": 2.20 / 1_000_000},
    "MiniMax-Text-01": {"input": 0.20 / 1_000_000, "output": 1.10 / 1_000_000},
    # Kimi
    "kimi-k2": {"input": 0.60 / 1_000_000, "output": 2.50 / 1_000_000},
    "moonshot-v1-8k": {"input": 0.20 / 1_000_000, "output": 2.00 / 1_000_000},
}

PROVIDER_MAP: dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.minimax.io": "minimax",
    "api.minimaxi.com": "minimax",
    "api.moonshot.ai": "kimi",
    "generativelanguage.googleapis.com": "google",
}


def detect_provider(client) -> str:
    """Auto-detect provider from client's base_url."""
    try:
        base_url = getattr(client, "base_url", None) or getattr(client, "_base_url", "")
        base_url = str(base_url)
        from urllib.parse import urlparse

        hostname = urlparse(base_url).hostname or ""
        return PROVIDER_MAP.get(hostname, hostname)
    except Exception:
        return "unknown"


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    custom_costs: dict = None,
) -> float:
    """Compute cost in dollars for a given model and token counts."""
    rates = (custom_costs or {}).get(model) or MODEL_COSTS.get(model)
    if not rates:
        return 0.0
    return (input_tokens * rates["input"]) + (output_tokens * rates["output"])
