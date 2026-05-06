import uuid
import os
from dotenv import load_dotenv

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

COLLECTION_NAME = "fintech_docs"

# ✅ FREE embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# =========================
# CREATE COLLECTION
# =========================
def create_collection():
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,   # IMPORTANT
            distance=Distance.COSINE
        )
    )

    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_type",
        field_schema=PayloadSchemaType.KEYWORD
    )

    print("✅ Collection created")


# =========================
# PARSE PDF
# =========================
def extract_elements(file_path):
    return partition_pdf(
        filename=file_path,
        strategy="fast",
        infer_table_structure=True
    )


# =========================
# CHUNKING
# =========================
def chunk_by_title(elements):
    chunks = []
    current_chunk = []
    current_title = "Unknown"

    for el in elements:
        text = str(el).strip()
        category = getattr(el, "category", "Text")

        if not text:
            continue

        if category == "Title":
            if current_chunk:
                chunks.append({
                    "title": current_title,
                    "text": " ".join(current_chunk)
                })
                current_chunk = []

            current_title = text
        else:
            current_chunk.append(text)

    if current_chunk:
        chunks.append({
            "title": current_title,
            "text": " ".join(current_chunk)
        })

    return chunks


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)


def refine_chunks(chunks):
    final_chunks = []

    for chunk in chunks:
        splits = splitter.split_text(chunk["text"])
        for s in splits:
            final_chunks.append({
                "title": chunk["title"],
                "text": s
            })

    return final_chunks


# =========================
# EMBEDDING
# =========================
def embed_text(text):
    return model.encode(text).tolist()


# =========================
# INGEST
# =========================
def ingest_document(file_path, doc_type):
    print(f"\n📄 Processing: {file_path}")

    elements = extract_elements(file_path)

    if not elements:
        print("⚠️ No elements extracted, skipping")
        return

    title_chunks = chunk_by_title(elements)

    if not title_chunks:
    # fallback: treat full doc as one chunk
        text = " ".join([str(el) for el in elements if str(el).strip()])
        title_chunks = [{"title": "Full Document", "text": text}]

    # if not title_chunks:
    #     print("⚠️ No title chunks, skipping")
    #     return

    final_chunks = refine_chunks(title_chunks)

    if not final_chunks:
        print("⚠️ No final chunks, skipping")
        return

    points = []

    for chunk in final_chunks:
        if not chunk["text"].strip():
            continue

        vector = embed_text(chunk["text"])

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "source": os.path.basename(file_path),
                    "doc_type": doc_type.lower().strip()
                }
            )
        )

    if not points:
        print("⚠️ No valid points, skipping upload")
        return

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Inserted {len(points)} chunks")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    create_collection()

    ingest_document("data/lending_loan_agreement.pdf", "lending")
    ingest_document("data/insurance_policy_sbi.pdf", "insurance")
    ingest_document("data/compliance_kyc_rbi.pdf", "compliance")
    #ingest_document("data/ADB_AML_Handbook.pdf", "compliance")
    ingest_document("data/Compliance-Guidelines.pdf", "compliance")
    ingest_document("data/apple_10k.pdf", "financial")


    
    # add more docs here

    print("\n🎉 Ingestion complete!")