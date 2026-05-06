# import os
# from dotenv import load_dotenv
# from rank_bm25 import BM25Okapi
# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# import cohere

# load_dotenv()

# # =========================
# # CONFIG
# # =========================
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL")
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# if not COHERE_API_KEY:
#     raise ValueError("❌ COHERE_API_KEY missing")

# co = cohere.Client(COHERE_API_KEY)

# COLLECTION_NAME = "fintech_docs"

# # Embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# qdrant = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY
# )

# # =========================
# # EMBEDDING
# # =========================
# def embed_query(text):
#     return model.encode(text).tolist()


# # =========================
# # LOAD CORPUS (for BM25)
# # =========================
# def load_corpus():
#     points, _ = qdrant.scroll(
#         collection_name=COLLECTION_NAME,
#         limit=10000,   # increased
#         with_payload=True
#     )

#     texts = [p.payload["text"] for p in points]
#     tokenized = [t.split() for t in texts]

#     return texts, tokenized, points


# corpus_texts, tokenized_corpus, corpus_points = load_corpus()
# bm25 = BM25Okapi(tokenized_corpus)


# # =========================
# # BM25 SEARCH (with filter)
# # =========================
# def bm25_search(query, doc_type=None, top_k=5):
#     tokenized_query = query.split()

#     scores = bm25.get_scores(tokenized_query)

#     scored_points = list(zip(corpus_points, scores))

#     if doc_type:
#         doc_type = doc_type.lower().strip()
#         scored_points = [
#             (p, s) for p, s in scored_points
#             if p.payload.get("doc_type") == doc_type
#         ]

#     scored_points.sort(key=lambda x: x[1], reverse=True)

#     return [p for p, _ in scored_points[:top_k]]


# # =========================
# # DENSE SEARCH
# # =========================
# def dense_search(query, doc_type=None):
#     vector = embed_query(query)

#     filter_condition = None
#     if doc_type:
#         filter_condition = {
#             "must": [
#                 {"key": "doc_type", "match": {"value": doc_type.lower().strip()}}
#             ]
#         }

#     results = qdrant.query_points(
#         collection_name=COLLECTION_NAME,
#         query=vector,
#         limit=10,
#         query_filter=filter_condition
#     )

#     return results.points


# # =========================
# # RRF MERGE
# # =========================
# def reciprocal_rank_fusion(dense, sparse, k=60):
#     scores = {}

#     for rank, doc in enumerate(dense):
#         scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank)

#     for rank, doc in enumerate(sparse):
#         scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank)

#     id_to_doc = {doc.id: doc for doc in dense + sparse}

#     ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#     return [id_to_doc[i[0]] for i in ranked]


# # =========================
# # HYBRID SEARCH
# # =========================
# def hybrid_search(query, doc_type=None):
#     dense_results = dense_search(query, doc_type)
#     sparse_results = bm25_search(query, doc_type)

#     merged = reciprocal_rank_fusion(dense_results, sparse_results)

#     return merged[:15]


# # =========================
# # RERANK
# # =========================
# def rerank(query, docs):
#     texts = [d.payload["text"] for d in docs]

#     response = co.rerank(
#         model="rerank-english-v3.0",
#         query=query,
#         documents=texts,
#         top_n=3
#     )

#     return [docs[r.index] for r in response.results]

"""
retrieval.py — OPTIMIZED for speed

Key changes:
1. Dense + BM25 search run in PARALLEL (ThreadPoolExecutor)
2. BM25 corpus loaded once at module import, never again
3. Rerank is optional — skippable for max speed
"""

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
def _embed(text: str) -> list[float]:
    return list(next(iter(embedder.embed([text]))))


# =========================
# DENSE SEARCH
# =========================
def _dense(query: str, doc_type: str = None, top_k: int = 10) -> list:
    qf = None
    if doc_type:
        qf = Filter(must=[FieldCondition(key="doc_type",
                    match=MatchValue(value=doc_type.lower().strip()))])
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=_embed(query), limit=top_k,
        query_filter=qf, with_payload=True
    ).points


# =========================
# BM25 SEARCH
# =========================
def _sparse(query: str, doc_type: str = None, top_k: int = 10) -> list:
    scores = _bm25.get_scores(query.lower().split())
    scored = list(zip(_points, scores))
    if doc_type:
        dt = doc_type.lower().strip()
        scored = [(p, s) for p, s in scored if p.payload.get("doc_type") == dt]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:top_k]]


# =========================
# RRF MERGE
# =========================
def _rrf(dense: list, sparse: list, k: int = 60) -> list:
    scores: dict = {}
    for r, d in enumerate(dense):
        scores[d.id] = scores.get(d.id, 0.0) + 1.0 / (k + r + 1)
    for r, d in enumerate(sparse):
        scores[d.id] = scores.get(d.id, 0.0) + 1.0 / (k + r + 1)
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
    return _rrf(dense, sparse)[:top_k]


# =========================
# RERANK (optional ~0.8s extra)
# =========================
def rerank(query: str, docs: list, top_n: int = 3) -> list:
    if not docs or not co:
        return docs[:top_n]
    texts = [d.payload["text"] for d in docs]
    res = co.rerank(model="rerank-english-v3.0", query=query,
                    documents=texts, top_n=min(top_n, len(docs)))
    return [docs[r.index] for r in res.results]