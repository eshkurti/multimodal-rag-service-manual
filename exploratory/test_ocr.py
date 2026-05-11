"""
Exploratory: OCR embedding similarity analysis.

Tests whether Tesseract OCR text from service manual images produces
reliable cosine similarity scores when embedded alongside text queries.

Conclusion: OCR-extracted text yields cosine similarity > 0.50 for
relevant diagram/table queries, confirming it as the primary retrieval
strategy over CLIP. See the term paper for full analysis.
"""

import sys
import numpy as np
from PIL import Image
import pytesseract
from core.llm_interface import call_embedding


def compute_ocr_similarity(image_path: str, query: str) -> tuple[str, float]:
    """Run OCR on an image, embed both the OCR text and query, return similarity."""
    image = Image.open(image_path).convert("RGB")
    ocr_text = pytesseract.image_to_string(image).strip()
    print(f"OCR text extracted:\n{ocr_text}\n")

    ocr_emb = call_embedding(ocr_text)[0]
    query_emb = call_embedding(query)[0]

    similarity = float(np.dot(ocr_emb, query_emb) / (np.linalg.norm(ocr_emb) * np.linalg.norm(query_emb)))
    return ocr_text, similarity


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_ocr.py <image_path> <query>")
        sys.exit(1)

    image_path = sys.argv[1]
    query = " ".join(sys.argv[2:])
    _, similarity = compute_ocr_similarity(image_path, query)
    print(f"OCR-query cosine similarity: {similarity:.4f}")
