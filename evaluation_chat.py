"""
Comparative evaluation runner.

Runs all three systems on the same query and displays results side by side:
  System A — Multimodal RAG + OCR  (text + OCR embeddings, images sent to LLM)
  System B — Text-Only RAG         (HTML text only, no OCR, no images)
  System C — Direct LLM            (no retrieval at all)
"""

import os
import sys
import builtins
import io
import re
import contextlib
from bs4 import BeautifulSoup

from data.service_manual_data import PAGES_DIR, title_to_file as orig_title_to_file
from data.service_manual_data import manual_database as manual_database_with_ocr
from data.service_manual_data import image_map as orig_image_map
from rag_chat import build_vector_store, rag_query
import core.llm_interface as llm_iface


def _call_and_capture(func, *args, **kwargs):
    """Call a function while capturing its stdout. Returns (captured_output, result)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = func(*args, **kwargs)
        return buf.getvalue().strip(), result
    except Exception as e:
        return buf.getvalue().strip(), f"Failed with exception: {e}"


def _extract_top_ranked(captured: str) -> tuple[str | None, float | None]:
    """Parse the top-ranked filename and score from captured RAG output."""
    if not captured:
        return None, None
    m = re.search(r"^\s*1\..*score=([0-9.]+).*file:\s*(\S+)", captured, re.MULTILINE)
    if m:
        try:
            return m.group(2), float(m.group(1))
        except ValueError:
            return m.group(2), None
    return None, None


def _build_text_only_db(pages_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    """Re-parse HTML pages with no OCR and no images (System B corpus)."""
    manual_db = {}
    title_map = {}

    for filename in os.listdir(pages_dir):
        if not filename.endswith(".html"):
            continue
        path = os.path.join(pages_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
        except Exception:
            continue

        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else filename

        main_div = soup.find("div", class_="main")
        if main_div:
            for tag in main_div.select("button, script, style, link, ul, li"):
                tag.decompose()
            for span in main_div.select("span.indent-right-align"):
                span.unwrap()
            for br in main_div.find_all("br"):
                br.replace_with("\n")
            for img in main_div.find_all("img"):
                img.decompose()
            text = main_div.get_text(separator="\n", strip=True)
        else:
            text = ""

        if len(text) < 50:
            continue

        manual_db[title] = text
        title_map[title] = filename

    return manual_db, title_map


def _print_result(label: str, filename: str | None, score: float | None, answer: str):
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")
    if filename:
        score_str = f" (score={score:.4f})" if score is not None else ""
        print(f"  Top ranked file: {filename}{score_str}")
    else:
        print("  Top ranked file: (none)")
    print(f"\n  Answer:\n  {answer}")
    print(f"{'─' * 60}")


def run_evaluation(query: str, model: str = "gpt-4o"):
    """Run all three systems on the same query and print comparative results."""
    print(f"Query: {query}")
    print(f"Model: {model}\n")

    # Build vector stores BEFORE capturing (so progress is visible)
    print("Preparing System A (Multimodal RAG + OCR)...")
    cache_a = os.path.join("data", "manual_embeddings_with_ocr.pkl")
    vector_db_a = build_vector_store(manual_database_with_ocr, cache_path=cache_a)

    print("Preparing System B (Text-Only RAG)...")
    manual_db_b, title_map_b = _build_text_only_db(PAGES_DIR)
    cache_b = os.path.join("data", "manual_embeddings_text_only.pkl")
    vector_db_b = build_vector_store(manual_db_b, cache_path=cache_b)

    print("Vector stores ready. Running evaluation...\n")

    # Suppress interactive input() calls inside rag_query
    original_input = builtins.input
    builtins.input = lambda _prompt="": "n"

    try:
        # System A: Multimodal RAG + OCR
        captured_a, answer_a = _call_and_capture(
            rag_query, query, vector_db_a, model=model
        )
        file_a, score_a = _extract_top_ranked(captured_a)
        _print_result("System A — Multimodal RAG + OCR", file_a, score_a, answer_a)

        # System B: Text-Only RAG (no OCR, no images)
        # Temporarily override the global image_map and title_to_file
        import data.service_manual_data as smd
        orig_db = smd.manual_database
        orig_ttf = smd.title_to_file
        orig_im = smd.image_map

        smd.manual_database = manual_db_b
        smd.title_to_file = title_map_b
        smd.image_map = {t: [] for t in title_map_b}

        captured_b, answer_b = _call_and_capture(
            rag_query, query, vector_db_b, model=model
        )
        file_b, score_b = _extract_top_ranked(captured_b)
        _print_result("System B — Text-Only RAG (no OCR, no images)", file_b, score_b, answer_b)

        # Restore originals
        smd.manual_database = orig_db
        smd.title_to_file = orig_ttf
        smd.image_map = orig_im

        # System C: Direct LLM (no retrieval)
        def direct_call(q):
            msgs = [{"role": "user", "content": [{"type": "text", "text": q}]}]
            return llm_iface.call_llm(msgs, model=model)

        _, answer_c = _call_and_capture(direct_call, query)
        _print_result("System C — Direct LLM (no retrieval)", None, None, answer_c)

    finally:
        builtins.input = original_input


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Please provide your query:\n").strip()
        if not query:
            print("No query provided.")
            sys.exit(1)

    run_evaluation(query, model="gpt-4o")
