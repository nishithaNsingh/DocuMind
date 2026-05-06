# 🧠 DocuMind — AI Document Intelligence

> A production-grade RAG (Retrieval-Augmented Generation) system for financial, legal, and compliance document Q&A. Built with hybrid retrieval, reranking, and hallucination-resistant answer generation.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

- **Hybrid Retrieval** — Dense vector search + BM25 sparse search merged via Reciprocal Rank Fusion (RRF)
- **Cohere Reranking** — Optional cross-encoder reranker for higher precision
- **Title-boosted RRF** — Section headings matched against query keywords for smarter ranking
- **Hallucination Reduction** — LLM grounded strictly in retrieved context, temperature = 0
- **Multi-document Reasoning** — Spans loan agreements, insurance policies, RBI compliance docs, and 10-K filings
- **RAGAS Evaluated** — Faithfulness: 0.90 · Context Precision: 0.78 · Context Recall: 1.00
- **Parallel Search** — Dense + BM25 run concurrently via ThreadPoolExecutor
- **Clean UI** — Dark-mode interface with source attribution and response timing

---

## 🏗️ Architecture

```
User Query
    │
    ├─── Dense Search (fastembed → Qdrant ANN)
    │                                           ──► RRF Merge ──► Title Boost ──► Top 10
    └─── BM25 Search (rank-bm25)
              │
              ▼
         [Optional] Cohere Rerank
              │
              ▼
         DeepSeek LLM (via OpenRouter)
              │
              ▼
         Grounded Answer + Sources
```

---

## 📁 Project Structure

```
documind/
├── main.py              # FastAPI app — serves UI + /query endpoint
├── retrieval.py         # Hybrid search: dense + BM25 + RRF + rerank
├── answer.py            # LLM prompt + DeepSeek generation
├── ingestion.py         # PDF parsing + chunking + embedding + Qdrant upsert
├── eval_data.py         # RAGAS evaluation dataset builder
├── ragas_eval.py        # RAGAS evaluation runner
├── index.html           # Frontend UI (served by FastAPI)
├── Dockerfile           # HF Spaces deployment (port 7860)
├── requirements.txt     # Production dependencies
├── .env                 # API keys (never commit this)
└── data/                # Source PDFs (not committed)
    ├── lending_loan_agreement.pdf
    ├── insurance_policy_sbi.pdf
    ├── compliance_kyc_rbi.pdf
    ├── Compliance-Guidelines.pdf
    └── apple_10k.pdf
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/yourusername/documind.git
cd documind
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the root:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 3. Ingest documents

```bash
python ingestion.py
```

This parses PDFs, chunks them by title, embeds with `all-MiniLM-L6-v2`, and upserts to Qdrant.

### 4. Run locally

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## 🐳 Docker

```bash
docker build -t documind .
docker run -p 8000:8000 --env-file .env documind
```

---

## 🚀 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — choose **Docker** SDK

2. Add secrets in **Settings → Variables and secrets**:
   ```
   QDRANT_URL
   QDRANT_API_KEY
   COHERE_API_KEY
   OPENROUTER_API_KEY
   ```

3. Push your code:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/documind
   # copy your files in (exclude .env and data/)
   git add .
   git commit -m "deploy: initial"
   git push
   ```

HF Spaces free tier provides 16GB RAM and 2 vCPUs — the ONNX model is pre-baked into the Docker image so there is no cold-start download penalty.

---

## 📊 RAGAS Evaluation

Run locally (not on HF — evaluation deps are commented out in requirements.txt):

```bash
pip install ragas datasets
python ragas_eval.py
```

| Metric | Score |
|---|---|
| Faithfulness | 0.90 |
| Context Precision | 0.78 |
| Context Recall | 1.00 |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Vector DB | Qdrant (cloud) |
| Embeddings | fastembed · all-MiniLM-L6-v2 (ONNX) |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| Reranker | Cohere rerank-english-v3.0 |
| LLM | DeepSeek via OpenRouter |
| PDF parsing | Unstructured.io |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| Deployment | Hugging Face Spaces (Docker) |

---

## 📄 Supported Document Types

| Type | Example |
|---|---|
| Lending | Loan agreements, term sheets |
| Insurance | SBI insurance policy |
| Compliance | RBI KYC guidelines, AML handbook |
| Financial | Apple 10-K annual report |


---

## 📝 License

MIT — free to use, fork, and build on.