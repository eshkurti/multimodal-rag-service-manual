# Multimodal RAG for Automotive Service Manuals

An OCR-augmented, multimodal Retrieval-Augmented Generation system that answers technical questions about the 2013 BMW 335i (E93, N55) by retrieving relevant pages from an HTML-based service manual and grounding LLM responses in both text and images. Built as a research project for the **Practical Studies Industry 4.0** course at Hochschule Hof.

The full academic term paper is included at [`docs/term_paper.pdf`](docs/term_paper.pdf).

## Key Results

Three systems were compared on a 50-query evaluation set spanning visually dependent, procedural, and factual questions:

| System | Method | Accuracy |
|---|---|---|
| **A — Multimodal RAG + OCR** | Text + OCR embeddings, images sent to LLM | **88%** |
| B — Text-Only RAG | HTML text only, no OCR, no images | 38% |
| C — Direct LLM | No retrieval, query only | 12% |

OCR-augmented multimodal retrieval more than doubled the accuracy of text-only RAG and was 7× more accurate than using the LLM alone.

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │       Service Manual (1400+ pages)  │
                          │  HTML text + diagrams + schematics  │
                          └──────────┬──────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    ETL Pipeline      │
                          │  • DOM cleaning      │
                          │  • Image extraction  │
                          │  • Tesseract OCR     │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Vector Store        │
                          │  (text + OCR → embeddings)
                          └──────────┬──────────┘
                                     │
        User Query ──► Embed ──► Cosine Similarity Search (top-k)
                                     │
                          ┌──────────▼──────────┐
                          │  Retrieved pages     │
                          │  text + original     │
                          │  images (base64)     │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Multimodal LLM      │
                          │  (GPT-4o via         │
                          │   OpenRouter)         │
                          └──────────┬──────────┘
                                     │
                              Grounded Answer
```

**Why OCR + images?** OCR text is used for *retrieval* (embedding and similarity search), while original images are sent to the LLM for *reasoning*. This avoids propagating OCR errors into the answer while still enabling retrieval of pages where critical data is locked in diagrams and tables.

## Project Structure

```
├── rag_chat.py                 # System A: Multimodal RAG + OCR
├── llm_chat.py                 # System C: Direct LLM (no retrieval)
├── evaluation_chat.py          # Comparative evaluation (runs all 3 systems)
├── core/
│   ├── llm_interface.py        # OpenRouter API (chat, embeddings, image encoding)
│   └── model_map.py            # Model registry
├── data/
│   └── service_manual_data.py  # HTML parser + Tesseract OCR + data loader
├── exploratory/
│   ├── test_clip.py            # CLIP cross-modal analysis (showed poor results)
│   └── test_ocr.py             # OCR embedding analysis (confirmed as primary strategy)
├── manual/                     # Service manual dataset
│   └── pages/                  # 1400+ HTML pages with embedded images
├── docs/
│   └── term_paper.pdf          # Full academic term paper with methodology and results
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

```bash
git clone https://github.com/your-username/bmw-service-manual-multimodal-rag.git
cd bmw-service-manual-multimodal-rag

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env.example
# Edit .env.example and add your OpenRouter API key
```

If Tesseract is not in your system PATH, set the path via environment variable:
```bash
export TESSERACT_CMD="/usr/bin/tesseract"          # Linux
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows
```

## Usage

**Multimodal RAG chatbot (System A):**
```bash
python rag_chat.py
```

On first run, the system embeds all ~1400 pages (text + OCR) and caches the result. Subsequent runs load from cache.

**Direct LLM baseline (System C):**
```bash
python llm_chat.py
```

**Run all three systems on the same query:**
```bash
python evaluation_chat.py "What is the tolerance for the front axle total toe adjustment?"
```

### Example Queries

Visually dependent (data locked in diagrams):
- `What is the tolerance for the front axle's total toe adjustment?`
- `What is the expected vacuum range at Waste Gate Actuator Bank at 3K RPM No Load?`

Procedural (multi-step reasoning):
- `What is the Test Plan code that may show errors regarding fuel pressure control?`

Factual (direct lookup):
- `What is the maximum negative camber difference allowed between the left and right front wheels?`

## Why CLIP Didn't Work

Exploratory testing with CLIP showed cosine similarity scores below 0.30 between text queries and technical diagrams/schematics, even when the images contained directly relevant content. CLIP was trained on natural images and captions, not on automotive schematics, torque tables, or alignment specification charts. The [`exploratory/`](exploratory/) directory contains the analysis scripts. Full discussion in the term paper.

## Tech Stack

- **Embeddings**: `text-embedding-3-small` (via OpenRouter)
- **LLM**: GPT-4o multimodal (via OpenRouter) — supports text + image input
- **OCR**: Tesseract (with JSON caching to avoid repeated processing)
- **ETL**: BeautifulSoup for HTML parsing, Pillow for image handling
- **Dataset**: 1400+ HTML pages from a community BMW E93 service manual
- **Evaluation**: 50-query stratified test set (visually dependent, procedural, factual)

## Design Decisions

**Document-level chunking**: Each HTML page is treated as a single retrieval unit rather than splitting into smaller chunks. This preserves spatial and procedural relationships between diagrams, captions, and instructions that would be lost with typical chunk-based approaches.

**OCR for retrieval, images for reasoning**: OCR-extracted text is embedded into the vector store for search, but original images are sent to the multimodal LLM. This separates the retrieval concern (where noisy OCR is acceptable) from the reasoning concern (where visual fidelity matters).

**Zero-shot pipeline**: No fine-tuning or domain-specific training required. The system works with any HTML-based service manual without annotation, making it practical for deployment across different vehicle models and manufacturers.

