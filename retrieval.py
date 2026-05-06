
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding
import cohere

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
COLLECTION_NAME = "fintech_docs"

embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# =========================
# LOAD BM25 CORPUS ONCE
# =========================
def _load_corpus():
    print("Loading BM25 corpus...")
    all_points, offset = [], None
    while True:
        batch, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000, offset=offset,
            with_payload=True, with_vectors=False
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    print(f"BM25 ready: {len(all_points)} chunks")
    texts     = [p.payload["text"] for p in all_points]
    tokenized = [t.lower().split() for t in texts]
    return texts, tokenized, all_points


_corpus_texts, _tokenized, _points = _load_corpus()
_bm25 = BM25Okapi(_tokenized)


# =========================
# EMBED
# =========================
def _embed(text: str) -> list:
    return list(next(iter(embedder.embed([text]))))


# =========================
# DENSE SEARCH — uses .search() (stable across all versions)
# =========================
def _dense(query: str, doc_type: str = None, top_k: int = 15) -> list:
    vector = _embed(query)

    query_filter = None
    if doc_type:
        query_filter = Filter(must=[
            FieldCondition(
                key="doc_type",
                match=MatchValue(value=doc_type.lower().strip())
            )
        ])

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True
    )
    return results


# =========================
# BM25 SEARCH
# =========================
def _sparse(query: str, doc_type: str = None, top_k: int = 15) -> list:
    scores = _bm25.get_scores(query.lower().split())
    scored = list(zip(_points, scores))
    if doc_type:
        dt = doc_type.lower().strip()
        scored = [(p, s) for p, s in scored if p.payload.get("doc_type") == dt]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:top_k]]


# =========================
# TITLE-BOOSTED RRF
# =========================
def _title_boost(doc, query_keywords: set) -> float:
    title = doc.payload.get("title", "").lower()
    matches = sum(1 for kw in query_keywords if kw in title)
    return 0.05 * matches


def _rrf(dense: list, sparse: list, query: str, k: int = 60) -> list:
    stopwords = {"what", "is", "are", "the", "a", "an", "of", "for",
                 "in", "to", "and", "or", "does", "do", "how", "which"}
    query_keywords = {w for w in query.lower().split()
                      if w not in stopwords and len(w) > 2}

    scores: dict = {}
    for rank, doc in enumerate(dense):
        base  = 1.0 / (k + rank + 1)
        boost = _title_boost(doc, query_keywords)
        scores[doc.id] = scores.get(doc.id, 0.0) + base + boost

    for rank, doc in enumerate(sparse):
        base  = 1.0 / (k + rank + 1)
        boost = _title_boost(doc, query_keywords)
        scores[doc.id] = scores.get(doc.id, 0.0) + base + boost

    id_map = {d.id: d for d in sparse}
    id_map.update({d.id: d for d in dense})
    return [id_map[i] for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# =========================
# HYBRID SEARCH — PARALLEL
# =========================
def hybrid_search(query: str, doc_type: str = None, top_k: int = 10) -> list:
    with ThreadPoolExecutor(max_workers=2) as ex:
        fd = ex.submit(_dense,  query, doc_type)
        fs = ex.submit(_sparse, query, doc_type)
        dense, sparse = fd.result(), fs.result()
    return _rrf(dense, sparse, query)[:top_k]


# =========================
# RERANK
# =========================
def rerank(query: str, docs: list, top_n: int = 3) -> list:
    if not docs or not co:
        return docs[:top_n]
    texts = [d.payload["text"] for d in docs]
    res = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=min(top_n, len(docs))
    )
    return [docs[r.index] for r in res.results]
