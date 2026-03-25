"""
search_test.py
==============
ChromaDBに登録されたベクトルデータの検索動作確認スクリプト。

実行:
    python search_test.py
"""

import os

# ===== プロキシ完全無効化 =====
for key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "NO_PROXY", "no_proxy"
]:
    os.environ.pop(key, None)

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()

# ===========================
# 設定
# ===========================
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "../../../data/chroma")
CHROMA_COLLECTION  = os.environ.get("CHROMA_COLLECTION_NAME", "knowledge_base")
EMBEDDING_MODEL    = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
TOP_K              = int(os.environ.get("TOP_K", "3"))

# ===========================
# ChromaDB 接続
# ===========================
client = Client(
    Settings(
        persist_directory=CHROMA_PERSIST_DIR,
        is_persistent=True
    )
)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

collection = client.get_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_function
)

print(f"\n✅ ChromaDB 接続成功")
print(f"   コレクション : {CHROMA_COLLECTION}")
print(f"   登録チャンク数: {collection.count()}")
print(f"   検索件数(K)  : {TOP_K}")

# ===========================
# 検索ループ
# ===========================
print("\n" + "="*60)
print("検索テスト開始（exit で終了）")
print("="*60)

while True:
    query = input("\n🔍 質問 > ").strip()

    if query.lower() in ["exit", "quit", "q"]:
        print("終了します")
        break

    if not query:
        continue

    # ===== ベクトル検索 =====
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\n{'─'*60}")
    print(f"検索結果（上位{len(docs)}件）")
    print(f"{'─'*60}")

    if not docs:
        print("❌ 該当する情報が見つかりませんでした")
        continue

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        similarity = 1 - dist  # コサイン距離 → 類似度に変換
        print(f"\n[{i+1}] 類似度: {similarity:.4f}  距離: {dist:.4f}")
        print(f"     タイトル  : {meta.get('title', '-')}")
        print(f"     カテゴリ  : {meta.get('category', '-')}")
        print(f"     タグ      : {meta.get('tags', '-')}")
        print(f"     チャンクNo: {meta.get('chunk_index', '-')}")
        print(f"     内容 ↓")
        print(f"     {doc[:200].replace(chr(10), ' ')}")
        if len(doc) > 200:
            print(f"     ...（{len(doc)}文字）")