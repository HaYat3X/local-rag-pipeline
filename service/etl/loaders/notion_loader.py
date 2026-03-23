"""
notion_loader.py
================
NotionのデータベースからレコードとページコンテンツをETL用に取得するローダー。

notion-client ライブラリは使用しない。
httpx で Notion REST API を直接叩くことでバージョン依存を排除している。
"""

import os
import time
import logging
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ===========================
# 設定（.env から取得）
# ===========================
NOTION_API_KEY  = os.environ.get("NOTION_API_KEY", "")
DATABASE_ID     = os.environ.get("NOTION_DATABASE_ID", "")
FILTER_STATUS   = os.environ.get("NOTION_FILTER_STATUS", "公開中")
API_SLEEP_SEC   = float(os.environ.get("NOTION_API_SLEEP", "0.4"))
NOTION_VERSION  = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"


# ===========================
# ブロック → テキスト変換
# ===========================
BLOCK_TYPE_MAP = {
    "heading_1":          lambda t: f"# {t}",
    "heading_2":          lambda t: f"## {t}",
    "heading_3":          lambda t: f"### {t}",
    "bulleted_list_item": lambda t: f"- {t}",
    "numbered_list_item": lambda t: f"1. {t}",
    "quote":              lambda t: f"> {t}",
    "code":               lambda t: f"```\n{t}\n```",
    "divider":            lambda _: "---",
    "paragraph":          lambda t: t,
}


def _extract_rich_text(rich_text_list: list) -> str:
    return "".join(rt.get("plain_text", "") for rt in rich_text_list)


def block_to_text(block: dict) -> Optional[str]:
    block_type = block.get("type")
    if not block_type:
        return None

    block_data = block.get(block_type, {})

    if block_type == "divider":
        return "---"

    if block_type == "code":
        text = _extract_rich_text(block_data.get("rich_text", []))
        lang = block_data.get("language", "")
        return f"```{lang}\n{text}\n```"

    rich_text = block_data.get("rich_text", [])
    if not rich_text:
        return None

    text = _extract_rich_text(rich_text)
    converter = BLOCK_TYPE_MAP.get(block_type)

    if converter:
        return converter(text)

    logger.debug(f"未対応ブロックタイプ: {block_type}")
    return text if text else None


# ===========================
# Notion API クライアント
# ===========================
class NotionLoader:

    def __init__(self):
        if not NOTION_API_KEY:
            raise ValueError(
                "NOTION_API_KEY が未設定。\n"
                ".env に NOTION_API_KEY=secret_xxx を追加すること。"
            )
        if not DATABASE_ID:
            raise ValueError(
                "NOTION_DATABASE_ID が未設定。\n"
                ".env に NOTION_DATABASE_ID=xxxxxxxx を追加すること。"
            )

        # httpx で直接 REST API を叩く（notion-client不使用）
        self._http = httpx.Client(
            verify=False,           # 社内SSL証明書対策
            timeout=30.0,
            headers={
                "Authorization":  f"Bearer {NOTION_API_KEY}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type":   "application/json",
            }
        )
        self.database_id = DATABASE_ID

    # ------------------------------------------------------------------
    # パブリックメソッド
    # ------------------------------------------------------------------

    def load(self) -> list[dict]:
        logger.info(f"[NotionLoader] DB取得開始: {self.database_id}")

        pages = self._query_database()
        logger.info(f"[NotionLoader] 取得レコード数: {len(pages)}")

        documents = []
        for page in pages:
            try:
                doc = self._build_document(page)
                if doc and doc["text"].strip():
                    documents.append(doc)
                    logger.info(f"  ✓ {doc['title']}")
                else:
                    logger.warning(f"  ✗ コンテンツが空のためスキップ: {page['id']}")
            except Exception as e:
                logger.error(f"  ✗ ページ取得エラー ({page['id']}): {e}")

            time.sleep(API_SLEEP_SEC)

        logger.info(f"[NotionLoader] 完了: {len(documents)} 件取得")
        return documents

    # ------------------------------------------------------------------
    # プライベートメソッド
    # ------------------------------------------------------------------

    def _query_database(self) -> list[dict]:
        """DBをクエリしてステータスが対象のページ一覧を返す"""
        pages = []
        payload = {
            "filter": {
                "property": "ステータス",
                "select": {"equals": FILTER_STATUS}
            }
        }

        while True:
            response = self._http.post(
                f"{NOTION_BASE_URL}/databases/{self.database_id}/query",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            pages.extend(data.get("results", []))

            if not data.get("has_more"):
                break

            payload["start_cursor"] = data.get("next_cursor")

        return pages

    def _get_page_blocks(self, page_id: str) -> list[dict]:
        """ページ内の全ブロックを取得する（ページネーション対応）"""
        blocks = []
        url    = f"{NOTION_BASE_URL}/blocks/{page_id}/children"
        params = {"page_size": 100}

        while True:
            response = self._http.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            blocks.extend(data.get("results", []))

            if not data.get("has_more"):
                break

            params["start_cursor"] = data.get("next_cursor")

        return blocks

    def _blocks_to_text(self, page_id: str) -> str:
        blocks = self._get_page_blocks(page_id)
        lines = []
        for block in blocks:
            text = block_to_text(block)
            if text and text.strip():
                lines.append(text)
        return "\n\n".join(lines)

    def _extract_properties(self, page: dict) -> dict:
        props = page.get("properties", {})

        title_prop = props.get("名前", {}).get("title", [])
        title = _extract_rich_text(title_prop) if title_prop else "（タイトルなし）"

        category_prop = props.get("カテゴリ", {}).get("select")
        category = category_prop["name"] if category_prop else ""

        tags_prop = props.get("タグ", {}).get("multi_select", [])
        tags = [t["name"] for t in tags_prop]

        status_prop = props.get("ステータス", {}).get("select")
        status = status_prop["name"] if status_prop else ""

        source_url = props.get("参照URL", {}).get("url", "")

        return {
            "title":      title,
            "category":   category,
            "tags":       tags,
            "status":     status,
            "source_url": source_url,
        }

    def _build_document(self, page: dict) -> dict:
        page_id = page["id"]
        props   = self._extract_properties(page)
        text    = self._blocks_to_text(page_id)

        return {
            "page_id":    page_id,
            "title":      props["title"],
            "category":   props["category"],
            "tags":       props["tags"],
            "status":     props["status"],
            "source_url": props["source_url"],
            "text":       text,
        }


# ===========================
# 単体実行（動作確認用）
# ===========================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    loader = NotionLoader()
    docs = loader.load()

    print(f"\n{'='*60}")
    print(f"取得ドキュメント数: {len(docs)}")
    print(f"{'='*60}")

    for doc in docs:
        print(f"\n--- {doc['title']} ---")
        print(f"  カテゴリ : {doc['category']}")
        print(f"  タグ     : {', '.join(doc['tags'])}")
        print(f"  文字数   : {len(doc['text'])} 文字")
        print(f"  冒頭     : {doc['text'][:100].replace(chr(10), ' ')}...")