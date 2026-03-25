from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

router = APIRouter()

# =========================
# Embedding
# =========================
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-base"
)

# =========================
# reranker
# =========================
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, docs):
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

# =========================
# DB
# =========================
client = Client(
    Settings(
        persist_directory=r"C:\Users\T1233\Desktop\ollma\data\chroma",
        is_persistent=True
    )
)



collection = client.get_collection(
    name="test",
    embedding_function=embedding_function
)


THRESHOLD = 0.5  # ← 適宜調整

# =========================
# Schema
# =========================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

# =========================
# Models API
# =========================
@router.get("/models")
def models():
    return {
        "data": [
            {"id": "local-rag-model"}
        ]
    }

# =========================
# Chat API
# =========================
@router.post("/chat/completions")
def chat(req: ChatRequest):

    query = req.messages[-1].content

    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # ===== rerank =====
    reranked = rerank(query, docs)

    # ===== filter =====
    filtered = [
        (doc, score)
        for doc, score in reranked
        if score >= THRESHOLD
    ]

    # ===== no result =====
    if not filtered:
        answer = "該当する情報が見つかりませんでした"
        sources = []
    else:
        # ===== context生成 =====
        top_docs = filtered[:3]
        context = "\n".join([doc for doc, _ in top_docs])

        # ===== sources =====
        sources = []
        for doc, score in top_docs:
            idx = docs.index(doc)
            sources.append({
                "content": doc,
                "rerank_score": float(score),
                "distance": float(dists[idx]),
                "metadata": metas[idx]
            })

        # ===== answer =====
        answer = (
            "【検索結果】\n"
            + context
            + "\n\n【回答】\n"
            + f"{query} に関する情報を上記から取得しました。"
        )

    # ===== OpenAI互換レスポンス =====
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer.strip()
                },
                "finish_reason": "stop"
            }
        ],
        # ⚠️ ここ重要：extra情報はトップに置かない
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        # 👉 extensionsとしてぶら下げる（安全）
        "extensions": {
            "sources": sources
        }
    }