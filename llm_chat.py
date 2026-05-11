"""
System C — Direct LLM (no retrieval).

Sends the user query directly to the LLM without any retrieved context.
Demonstrates hallucination and approximation when the model has no grounding.
"""

from core.llm_interface import call_llm
from core.model_map import MODEL_MAP

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on the 2013 BMW 335i Convertible (E93) "
    "L6-3.0L Turbo (N55). "
    "If you don't know the answer, say: "
    '"Sorry, I don\'t know the answer to that question. Please consult the service manual."'
)


def llm_query(query: str, model: str = "gpt-4o") -> str:
    """Send a query directly to the LLM without any retrieval context."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return call_llm(messages, model=model)


def select_model() -> str:
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
    query = input("\nEnter your question: ").strip()
    if not query:
        print("No query provided.")
        return

    model = select_model()
    print(f"\nUsing model: {model}")

    answer = llm_query(query, model=model)
    print(f"\n{'─' * 60}\n{answer}\n{'─' * 60}")


if __name__ == "__main__":
    main()
