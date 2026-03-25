"""
ingest.py
=========
ETLエントリーポイント。
NotionデータベースのコンテンツをChromaDBにベクトル化して格納する。

実行:
    python ingest.py              # 通常実行
    python ingest.py --reset      # ChromaDBをリセットしてから再取り込み
    python ingest.py --dry-run    # 取得のみ（ChromaDBへの書き込みなし）
"""

import os
import sys
import argparse
import logging

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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import shutil
from loaders.notion_loader import NotionLoader
load_dotenv() 


# ===========================
# ログ設定
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ===========================
# 設定（環境変数 or デフォルト値）
# ===========================
CHROMA_PERSIST_DIR    = os.environ.get("CHROMA_PERSIST_DIR", "../../data/chroma")
CHROMA_COLLECTION     = os.environ.get("CHROMA_COLLECTION_NAME", "knowledge_base")
EMBEDDING_MODEL       = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE            = int(os.environ.get("CHUNK_SIZE", "300"))
CHUNK_OVERLAP         = int(os.environ.get("CHUNK_OVERLAP", "50"))

# ===========================
# ChromaDB 初期化
# ===========================
def build_chroma_client():
    return Client(
        Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            is_persistent=True
        )
    )

def get_collection(client, reset: bool = False):
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            logger.info(f"[ChromaDB] コレクション削除: {CHROMA_COLLECTION}")
        except Exception:
            pass  # 存在しない場合は無視

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_function
    )
    logger.info(f"[ChromaDB] コレクション: {CHROMA_COLLECTION} (既存件数: {collection.count()})")
    return collection

# ===========================
# チャンク分割
# ===========================
def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )

def split_document(doc: dict, splitter) -> tuple[list[str], list[dict], list[str]]:
    """
    1ドキュメントをチャンク分割し、
    (documents, metadatas, ids) のタプルを返す。
    """
    chunks = splitter.split_text(doc["text"])

    documents = []
    metadatas = []
    ids       = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc['page_id']}_{i}"

        documents.append(chunk)
        metadatas.append({
            "page_id":    doc["page_id"],
            "title":      doc["title"],
            "category":   doc["category"],
            "tags":       ",".join(doc["tags"]),
            "source_url": doc["source_url"],
            "chunk_index": i,
            "source":     "notion",
        })
        ids.append(chunk_id)

    return documents, metadatas, ids

# ===========================
# ChromaDB 書き込み
# ===========================
def upsert_chunks(collection, documents, metadatas, ids):
    """
    チャンクをChromaDBにupsertする。
    同一IDが既に存在する場合は上書きされる（再実行に対して冪等）。
    """
    if not documents:
        return

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

# ===========================
# メイン処理
# ===========================
def run(reset: bool = False, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("ETL 開始")
    logger.info(f"  Embedding モデル : {EMBEDDING_MODEL}")
    logger.info(f"  ChromaDB         : {CHROMA_PERSIST_DIR}")
    logger.info(f"  コレクション名   : {CHROMA_COLLECTION}")
    logger.info(f"  チャンクサイズ   : {CHUNK_SIZE} / オーバーラップ: {CHUNK_OVERLAP}")
    logger.info(f"  dry-run          : {dry_run}")
    logger.info("=" * 60)

    # ----- Step 1: Notionからデータ取得 -----
    loader = NotionLoader()
    docs = loader.load()

    if not docs:
        logger.warning("取得できたドキュメントが0件。処理を終了する。")
        return

    # ----- Step 2: チャンク分割 -----
    splitter = build_splitter()

    all_documents = []
    all_metadatas = []
    all_ids       = []

    for doc in docs:
        documents, metadatas, ids = split_document(doc, splitter)
        all_documents.extend(documents)
        all_metadatas.extend(metadatas)
        all_ids.extend(ids)
        logger.info(f"  チャンク分割: '{doc['title']}' → {len(documents)} チャンク")

    logger.info(f"\n合計チャンク数: {len(all_documents)}")

    if dry_run:
        logger.info("[dry-run] ChromaDBへの書き込みをスキップ")
        _print_sample_chunks(all_documents, all_metadatas, all_ids)
        return

    # ----- Step 3: ChromaDBへ書き込み -----
    chroma_client = build_chroma_client()
    collection    = get_collection(chroma_client, reset=reset)

    upsert_chunks(collection, all_documents, all_metadatas, all_ids)

    logger.info(f"\n[完了] ChromaDB 登録件数: {collection.count()} チャンク")

def _print_sample_chunks(documents, metadatas, ids, n=3):
    """dry-run 時のサンプル表示"""
    print(f"\n{'='*60}")
    print(f"サンプルチャンク（先頭{n}件）")
    print(f"{'='*60}")
    for i in range(min(n, len(documents))):
        print(f"\n[{ids[i]}]")
        print(f"  title    : {metadatas[i]['title']}")
        print(f"  category : {metadatas[i]['category']}")
        print(f"  tags     : {metadatas[i]['tags']}")
        print(f"  chunk    : {documents[i][:120].replace(chr(10), ' ')}...")

# ===========================
# CLI
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Notion → ChromaDB ETL")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="ChromaDBのコレクションをリセットしてから再取り込みする"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Notionからの取得とチャンク分割のみ行い、ChromaDBへの書き込みは行わない"
    )
    args = parser.parse_args()

    run(reset=args.reset, dry_run=args.dry_run)