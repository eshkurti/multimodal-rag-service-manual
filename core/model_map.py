"""Registry of LLM models available via OpenRouter."""

MODEL_MAP = {
    "gpt-4o": {
        "provider": "openrouter",
        "model_name": "openai/gpt-4o",
    },
    "gpt-4o-mini": {
        "provider": "openrouter",
        "model_name": "openai/gpt-4o-mini",
    },
    "gemini-2.5-flash": {
        "provider": "openrouter",
        "model_name": "google/gemini-2.5-flash",
    },
    "llama-4-maverick": {
        "provider": "openrouter",
        "model_name": "meta-llama/llama-4-maverick",
    },
    "claude-sonnet-4": {
        "provider": "openrouter",
        "model_name": "anthropic/claude-sonnet-4",
    },
}


def resolve_model(key: str) -> str:
    """Resolve a friendly model key to the OpenRouter model string."""
    if key in MODEL_MAP:
        return MODEL_MAP[key]["model_name"]
    return key
