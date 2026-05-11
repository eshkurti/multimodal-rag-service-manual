"""OpenRouter API interface for chat completions, embeddings, and image encoding."""

import os
import base64
import requests
import numpy as np
from dotenv import load_dotenv
from core.model_map import resolve_model

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def call_llm(messages: list, model: str = "gpt-4o", max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Send a chat completion request to OpenRouter. Supports multimodal messages (text + images)."""
    model_name = resolve_model(model)

    prepared_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", [])
        if isinstance(content, list):
            prepared_messages.append({"role": role, "content": content})
        else:
            prepared_messages.append({"role": role, "content": [{"type": "text", "text": str(content)}]})

    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": prepared_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    message_content = response.json()["choices"][0]["message"].get("content", [])
    if isinstance(message_content, list):
        return "\n".join(c.get("text", "") for c in message_content if c.get("type") == "text")
    return str(message_content)


def call_embedding(texts, model: str = "openai/text-embedding-3-small") -> list[np.ndarray]:
    """Embed one or more texts via OpenRouter. Returns L2-normalized vectors."""
    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": texts},
    )

    if response.status_code != 200:
        raise RuntimeError(f"Embedding API error {response.status_code}: {response.text}")

    embeddings = [np.array(item["embedding"], dtype=np.float32) for item in response.json()["data"]]

    def l2_normalize(vec, eps=1e-12):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > eps else vec

    return [l2_normalize(e) for e in embeddings]


def image_to_base64(path: str) -> str:
    """Convert an image file to a raw base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
