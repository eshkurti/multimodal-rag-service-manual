"""
System A — Multimodal RAG with OCR.

Embeds service manual pages (HTML text + OCR-extracted text from images) into a
vector store, retrieves the top-k most similar pages for a user query, and sends
both the retrieved text and original images to a multimodal LLM for grounded reasoning.
"""

import os
import re
import time
import pickle
import numpy as np

from data.service_manual_data import manual_database, title_to_file, image_map
from core.llm_interface import call_llm, call_embedding, image_to_base64
from core.model_map import MODEL_MAP

EMBEDDING_CACHE = os.path.join("data", "manual_embeddings.pkl")
BATCH_SIZE = 100
SLEEP_PER_BATCH = 0.1

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on the 2013 BMW 335i Convertible (E93) "
    "L6-3.0L Turbo (N55). "
    "If the provided context does not contain enough information to answer the question, "
    'respond only with: "Sorry, I don\'t know the answer to that question. Please consult the service manual."'
)


# ── Vector store ─────────────────────────────────────────────────────────────

def _save_cache(path: str, obj: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_cache(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def build_vector_store(
    database: dict[str, str], cache_path: str = EMBEDDING_CACHE
) -> dict[str, np.ndarray]:
    """Build or load cached embeddings for every page in the manual."""
    keys = list(database.keys())
    cache = _load_cache(cache_path)
    if cache and set(cache.keys()) == set(keys):
        return {k: np.array(v, dtype=np.float32) for k, v in cache.items()}

    print(f"Building embeddings for {len(keys)} pages...")
    texts = [f"{title}\n\n{database[title]}" for title in keys]

    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings.extend(call_embedding(batch))
        time.sleep(SLEEP_PER_BATCH)

    store = {keys[i]: embeddings[i] for i in range(len(keys))}
    _save_cache(cache_path, {k: v.tolist() for k, v in store.items()})
    return store


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_embedding: np.ndarray,
    vector_store: dict[str, np.ndarray],
    k: int = 5,
    min_similarity: float = 0.5,
) -> list[tuple[str, float]]:
    """Return the top-k page titles by cosine similarity above a threshold."""
    keys = list(vector_store.keys())
    matrix = np.vstack([vector_store[k] for k in keys])
    similarities = matrix.dot(query_embedding)

    valid_idx = np.where(similarities >= min_similarity)[0]
    if valid_idx.size == 0:
        return []

    k_eff = min(k, valid_idx.size)
    top_unsorted = valid_idx[np.argpartition(similarities[valid_idx], -k_eff)[-k_eff:]]
    top_sorted = top_unsorted[np.argsort(similarities[top_unsorted])[::-1]]
    return [(keys[i], float(similarities[i])) for i in top_sorted]


# ── Text sanitization ────────────────────────────────────────────────────────

def _sanitize_text(s: str) -> str:
    """Clean common LaTeX artifacts from OCR-extracted text."""
    if not s:
        return s
    s = re.sub(r"\\\(|\\\)", "", s)
    s = s.replace(r"\pm", "±")
    s = re.sub(r"\^\\circ", "°", s)
    s = s.replace(r"\circ", "°")
    s = re.sub(r"\\([^\w])", r"\1", s)
    return s


# ── RAG pipeline ─────────────────────────────────────────────────────────────

def rag_query(
    query: str,
    vector_store: dict[str, np.ndarray],
    model: str = "gpt-4o",
    k: int = 5,
    min_similarity: float = 0.5,
) -> str:
    """Run the full multimodal RAG pipeline: embed → retrieve → generate with images."""
    query_embedding = call_embedding(query)[0]
    hits = retrieve_top_k(query_embedding, vector_store, k=k, min_similarity=min_similarity)

    if not hits:
        return "Sorry, I don't know the answer to that question. Please consult the service manual."

    print(f"\nRetrieved {len(hits)} relevant pages:")
    for i, (title, score) in enumerate(hits, 1):
        filename = title_to_file.get(title, "?")
        print(f"  {i}. {title} (score={score:.4f}) — file: {filename}")

    # Build multimodal messages: text + images for each retrieved page
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    seen_images = set()

    for title, _ in hits:
        content_blocks = []

        # Text block
        text = _sanitize_text(f"Q: {title}\nA: {manual_database.get(title, '')}")
        content_blocks.append({"type": "text", "text": text})

        # Image blocks (deduplicated)
        for img_path in image_map.get(title, []):
            try:
                raw_b64 = image_to_base64(img_path)
                if raw_b64 in seen_images:
                    continue
                seen_images.add(raw_b64)

                ext = os.path.splitext(img_path)[1].lower().lstrip(".")
                mime = "jpeg" if ext == "jpg" else ext if ext in {"png", "jpeg", "gif", "bmp", "tiff"} else "jpeg"
                data_url = f"data:image/{mime};base64,{raw_b64}"
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception as e:
                print(f"Warning: Failed to encode image {img_path}: {e}")

        messages.append({"role": "user", "content": content_blocks})

    # Append user query as final message
    messages.append({"role": "user", "content": [{"type": "text", "text": _sanitize_text(query)}]})

    answer = call_llm(messages, model=model)
    return _sanitize_text(answer)


# ── CLI entry point ──────────────────────────────────────────────────────────

def select_model() -> str:
    """Prompt the user to pick a model from the registry."""
    keys = list(MODEL_MAP.keys())
    print("\nAvailable models:")
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {key}  ({MODEL_MAP[key]['model_name']})")

    while True:
        choice = input("Choose a model by number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        print("Invalid choice. Try again.")


def main():
    vector_store = build_vector_store(manual_database)

    query = input("\nEnter your question: ").strip()
    if not query:
        print("No query provided.")
        return

    model = select_model()
    print(f"\nUsing model: {model}")

    answer = rag_query(query, vector_store, model=model)
    print(f"\n{'─' * 60}\n{answer}\n{'─' * 60}")


if __name__ == "__main__":
    main()
