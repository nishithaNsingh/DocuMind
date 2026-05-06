import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="FinTech RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Import retrieval at startup so BM25 corpus loads ONCE
# (not on first request)
from retrieval import hybrid_search, rerank
from answer import generate_answer


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/query")
def query(
    q: str = Query(...),
    doc_type: str = Query(None),
    use_rerank: bool = Query(False)   # toggle — False = faster, True = more accurate
):
    docs = hybrid_search(q, doc_type)

    if use_rerank:
        docs = rerank(q, docs)         
    else:
        docs = docs[:3]                 # just take top 3 from RRF

    answer  = generate_answer(q, docs)
    sources = list(set([d.payload["source"] for d in docs]))

    return JSONResponse({"answer": answer, "sources": sources})


@app.get("/health")
def health():
    return {"status": "ok"}