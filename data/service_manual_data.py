"""
ETL pipeline for the BMW service manual.

Parses HTML pages, extracts text content, runs Tesseract OCR on embedded images,
and builds the data structures used by the RAG pipeline.
"""

import os
import json
import shutil
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

# Tesseract path: set via env var, or auto-detect from PATH / common locations
_TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
elif not shutil.which("tesseract"):
    # Common Windows install location
    _win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win_default):
        pytesseract.pytesseract.tesseract_cmd = _win_default

PAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "manual", "pages")
OCR_CACHE_FILE = os.path.join(PAGES_DIR, "ocr_cache.json")
RASTER_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ── OCR cache ────────────────────────────────────────────────────────────────

if os.path.exists(OCR_CACHE_FILE):
    with open(OCR_CACHE_FILE, "r", encoding="utf-8") as _f:
        OCR_CACHE = json.load(_f)
else:
    OCR_CACHE = {}


def _save_ocr_cache():
    with open(OCR_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(OCR_CACHE, f, ensure_ascii=False, indent=2)


def _norm_image_path(page_filename: str, img_src: str) -> str:
    if os.path.isabs(img_src):
        return img_src
    page_dir = os.path.dirname(os.path.join(PAGES_DIR, page_filename))
    return os.path.normpath(os.path.join(page_dir, img_src))


def _extract_text_from_image(image_path: str) -> str:
    """Run Tesseract OCR on an image, with caching to avoid repeat processing."""
    if image_path in OCR_CACHE:
        return OCR_CACHE[image_path]
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            ocr_text = pytesseract.image_to_string(img).strip()
            OCR_CACHE[image_path] = ocr_text
            return ocr_text
    except Exception as e:
        print(f"Warning: OCR failed for {image_path}: {e}")
        OCR_CACHE[image_path] = ""
        return ""


# ── Main loader ──────────────────────────────────────────────────────────────

def load_manual_database() -> tuple[dict, dict, dict]:
    """
    Parse all HTML pages in the service manual.

    Returns:
        manual_db: {title: combined_text_content}
        title_to_file: {title: filename}
        image_map: {title: [absolute_image_paths]}
    """
    manual_db = {}
    title_to_file = {}
    image_map = {}

    for filename in os.listdir(PAGES_DIR):
        if not filename.endswith(".html"):
            continue

        path = os.path.join(PAGES_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else filename

        main_div = soup.find("div", class_="main")
        content_parts = []
        images_for_page = []

        if main_div:
            # Collect raster images before cleaning the DOM
            for img in main_div.find_all("img"):
                src = img.get("src") or ""
                if not src:
                    continue
                abs_img = _norm_image_path(filename, src)
                ext = os.path.splitext(abs_img)[1].lower()
                if os.path.exists(abs_img) and ext in RASTER_IMAGE_EXTS:
                    images_for_page.append(abs_img)

            # Remove non-semantic markup
            for tag in main_div.select("button, script, style, link, ul, li"):
                tag.decompose()
            for span in main_div.select("span.indent-right-align"):
                span.unwrap()
            for br in main_div.find_all("br"):
                br.replace_with("\n")
            for img in main_div.find_all("img"):
                img.decompose()

            text_content = main_div.get_text(separator="\n", strip=True)
            if text_content:
                content_parts.append(text_content)

        # OCR extraction for images
        ocr_texts = []
        for img_path in images_for_page:
            ocr_text = _extract_text_from_image(img_path)
            if len(ocr_text) > 10:
                ocr_texts.append(ocr_text)

        if ocr_texts:
            content_parts.append("\n\n".join(ocr_texts))

        content = "\n\n".join(content_parts) if content_parts else ""

        if len(content) < 50 and not images_for_page:
            continue

        manual_db[title] = content
        title_to_file[title] = filename
        image_map[title] = images_for_page

    _save_ocr_cache()
    return manual_db, title_to_file, image_map


manual_database, title_to_file, image_map = load_manual_database()
