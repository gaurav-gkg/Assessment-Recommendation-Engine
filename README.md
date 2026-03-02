# SHL Assessment Recommendation Engine

> **SHL AI Research Engineer — GenAI Assessment Submission**

An intelligent, production-ready recommendation system that maps natural-language hiring queries (or job descriptions / URLs) to relevant **SHL Individual Test Solutions** using Retrieval-Augmented Generation (RAG) with Google Gemini and FAISS.

---

## Architecture Overview

```
User Query / JD text / URL
          │
          ▼
 ┌─────────────────────┐
 │   Query Processor   │  ← URL fetching + Gemini enrichment
 └────────┬────────────┘
          │ enriched query
          ▼
 ┌─────────────────────┐
 │   FAISS Vector DB   │  ← Gemini text-embedding-004 embeddings
 │  (cosine similarity)│     of 377+ Individual Test Solutions
 └────────┬────────────┘
          │ top-30 candidates
          ▼
 ┌─────────────────────┐
 │   Gemini Reranker   │  ← pointwise LLM relevance scoring
 └────────┬────────────┘
          │ top-N re-scored
          ▼
 ┌─────────────────────┐
 │  Category Balancer  │  ← enforces diversity across test types
 └────────┬────────────┘
          │ 5–10 recommendations
          ▼
     FastAPI  /recommend
          │
          ▼
   Streamlit Frontend
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM & Embeddings | Google Gemini (`gemini-2.0-flash`, `text-embedding-004`) |
| Vector Search | FAISS (flat inner-product, cosine normalised) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Scraping | Requests + BeautifulSoup4 + Selenium (fallback) |
| Evaluation | Custom Recall@K, MAP, NDCG implementation |

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd SHL
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

### 3. Prepare Datasets

```bash
python scripts/prepare_datasets.py
# Reads Gen_AI Dataset.xlsx → data/datasets/train.json + test.json
```

### 4. Scrape the SHL Catalogue

```bash
python scripts/scrape_catalogue.py
```

This will:
- Crawl `https://www.shl.com/solutions/products/product-catalog/`
- Filter for **Individual Test Solutions** only
- Enrich each entry with detail-page metadata
- Save to `data/raw/shl_catalogue.json` and `data/processed/assessments.json`

> ⚠️ Requires internet access. Respects a 1.2 s delay between requests.

### 5. Build the FAISS Index

```bash
python scripts/build_index.py
```

Computes Gemini embeddings for all assessments and builds the FAISS index.

### 6. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### 7. Start the Frontend

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
SHL/
├── api/                    # FastAPI application
│   ├── main.py             #   App factory + middleware
│   ├── routes.py           #   /health and /recommend endpoints
│   └── models.py           #   Pydantic request/response models
├── embeddings/             # Embedding & vector search
│   ├── embedding_model.py  #   Google Gemini embeddings wrapper
│   └── vector_store.py     #   FAISS index management
├── recommender/            # RAG pipeline
│   ├── rag_engine.py       #   Orchestrator (main entry point)
│   ├── query_processor.py  #   URL fetching + Gemini query enrichment
│   └── reranker.py         #   LLM reranking + category balancing
├── scraper/                # Data ingestion
│   ├── shl_scraper.py      #   Catalogue scraper (pagination + detail pages)
│   └── data_processor.py   #   Normalisation & search-text construction
├── evaluation/             # Metrics and evaluation runner
│   ├── metrics.py          #   Recall@K, MAP, NDCG
│   └── evaluator.py        #   Evaluation pipeline + CSV generation
├── frontend/
│   └── app.py              # Streamlit UI
├── scripts/                # CLI entry points
│   ├── prepare_datasets.py  #   Convert Excel dataset → JSON
│   ├── scrape_catalogue.py
│   ├── build_index.py
│   ├── evaluate.py
│   └── generate_predictions.py
├── data/
│   ├── raw/                # Scraped JSON (gitignored)
│   ├── processed/          # Embeddings + FAISS index (gitignored)
│   └── datasets/           # train.json, test.json, predictions.csv
├── tests/                  # Pytest test suite
│   ├── test_metrics.py
│   ├── test_api.py
│   └── test_recommender.py
├── config.py               # Centralised configuration (pydantic-settings)
├── requirements.txt
├── .env.example
└── pytest.ini
```

---

## Design Decisions

### Why RAG over fine-tuning?
The SHL catalogue changes over time. RAG allows the system to stay current by simply re-running the scraper and rebuilding the index — no model retraining required.

### Why Gemini for both embeddings and reranking?
Keeping the full pipeline within the Google ecosystem reduces integration complexity and cost. `text-embedding-004` provides 768-dimensional embeddings that are excellent for semantic retrieval.

### Why FAISS over a managed vector DB?
FAISS runs locally with zero infrastructure overhead, making the project fully self-contained and reproducible. It trivially scales to the 377+ assessment corpus.

### Category Balancing
A greedy swap algorithm ensures that no single test-type category exceeds 70 % of the final recommendation list. This directly addresses the *balanced recommendations* requirement in the brief.

---



