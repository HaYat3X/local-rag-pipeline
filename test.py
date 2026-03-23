import os

# ===== プロキシ完全無効化 =====
for key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "NO_PROXY", "no_proxy"
]:
    os.environ.pop(key, None)

from chromadb import Client
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# ===== Embedding =====
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="all-MiniLM-L6-v2"
# )
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-base"
)
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, docs):
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked

# ===== DB =====
client = Client()
collection = client.get_or_create_collection(
    name="test",
    embedding_function=embedding_function
)



# ===== Notion Blockデータ =====
notion_blocks = [
    {"type": "heading_1", "text": "RAGとGISの基礎"},
    {"type": "paragraph", "text": "RAG（Retrieval Augmented Generation）は、外部データを検索してからLLMに渡すことで回答精度を向上させる手法です。特にGISのような専門領域では、既存ドキュメントの活用が重要になります。"},
    {"type": "paragraph", "text": "従来のLLMは学習データに依存していましたが、RAGではリアルタイムに情報を取得できるため、最新情報や社内ナレッジの活用が可能になります。"},
    {"type": "heading_2", "text": "RAGの処理フロー"},
    {"type": "code", "text": "ユーザー質問 → Embedding → ベクトル検索 → 関連チャンク取得 → LLMに入力 → 回答生成"},
    {"type": "paragraph", "text": "この流れの中で重要なのは検索部分です。どれだけ適切なチャンクを取得できるかで最終的な回答品質が決まります。"},
    {"type": "heading_2", "text": "Chunkの重要性"},
    {"type": "paragraph", "text": "Chunkとは文章を小さな単位に分割することを指します。適切なサイズで分割することで検索精度が向上します。一般的には200〜500文字が推奨されます。"},
    {"type": "bulleted_list_item", "text": "小さすぎる → 文脈が失われる"},
    {"type": "bulleted_list_item", "text": "大きすぎる → 無関係な情報が混ざる"},
    {"type": "bulleted_list_item", "text": "適切なサイズ → 意味と精度のバランスが良い"},
    {"type": "paragraph", "text": "また、チャンク分割の際には文単位で区切ることが重要です。文字数だけで分割すると意味が壊れる可能性があります。"},
    {"type": "heading_2", "text": "Embeddingの役割"},
    {"type": "paragraph", "text": "Embeddingは文章を数値ベクトルに変換する技術です。意味が近い文章はベクトル空間でも近くなります。これによりキーワード一致ではなく意味検索が可能になります。"},
    {"type": "paragraph", "text": "例えば「GISとは何か」と「地理情報システムとは？」は異なる表現ですが、Embeddingでは近い位置に配置されます。"},
    {"type": "code", "text": "[0.12, -0.98, 0.33, ...] ← ベクトル表現"},
    {"type": "heading_2", "text": "Embeddingモデルの種類"},
    {"type": "bulleted_list_item", "text": "MiniLM → 軽量・高速・PoC向け"},
    {"type": "bulleted_list_item", "text": "BGE → 高精度・多言語対応"},
    {"type": "bulleted_list_item", "text": "OpenAI → 高精度だが外部API依存"},
    {"type": "paragraph", "text": "用途によってモデルを選択する必要があります。ローカル環境では軽量モデルが扱いやすいですが、精度はやや落ちる場合があります。"},
    {"type": "heading_2", "text": "GISでのRAG活用"},
    {"type": "paragraph", "text": "GISでは大量のドキュメントが存在します。例えば操作手順書、データ仕様書、解析アルゴリズムの説明などです。これらをRAGで検索可能にすることで業務効率が大幅に向上します。"},
    {"type": "paragraph", "text": "具体例としては、バッファ解析、空間結合、座標変換などの処理手順を自然言語で検索できるようになります。"},
    {"type": "heading_3", "text": "具体的なユースケース"},
    {"type": "bulleted_list_item", "text": "QGISの操作方法を検索"},
    {"type": "bulleted_list_item", "text": "ArcGISの設定手順を検索"},
    {"type": "bulleted_list_item", "text": "PythonでのGIS処理コードを検索"},
    {"type": "heading_2", "text": "よくある失敗パターン"},
    {"type": "paragraph", "text": "RAGがうまく動かない場合、多くはChunk設計かEmbeddingの問題です。LLM自体の問題であるケースは少ないです。"},
    {"type": "bulleted_list_item", "text": "Chunkが大きすぎる"},
    {"type": "bulleted_list_item", "text": "Chunkが小さすぎる"},
    {"type": "bulleted_list_item", "text": "Embeddingモデルが弱い"},
    {"type": "bulleted_list_item", "text": "検索件数（top_k）が少ない"},
    {"type": "heading_2", "text": "まとめ"},
    {"type": "paragraph", "text": "RAGの精度はChunkとEmbeddingでほぼ決まります。LLMは最後の仕上げに過ぎません。まずは検索精度を改善することが重要です。"}
]

icp_blocks = [

    {"type": "heading_1", "text": "ICP（Iterative Closest Point）"},

    {"type": "heading_2", "text": "概要"},
    {"type": "paragraph", "text": "ICPは3D点群の位置合わせに使用されるアルゴリズムで、注目する点に対して最も近い点を対応付けることで2つの点群の位置関係を最適化する。"},

    {"type": "heading_2", "text": "特徴"},
    {"type": "bulleted_list_item", "text": "2つの点群の間で最適な位置関係を求める"},
    {"type": "bulleted_list_item", "text": "対応点間の距離を最小化する"},
    {"type": "bulleted_list_item", "text": "大きなズレや回転には弱い"},

    {"type": "heading_2", "text": "仕組み"},

    {"type": "heading_3", "text": "対応点探索"},
    {"type": "paragraph", "text": "移動させる点群の各点に対して、基準点群から最も近い点を探索し対応関係を決定する。"},

    {"type": "heading_3", "text": "変換計算"},
    {"type": "paragraph", "text": "対応点ペアをもとに、距離が最小になるような回転と並進を計算する。"},

    {"type": "heading_3", "text": "反復処理"},
    {"type": "paragraph", "text": "変換と対応点探索を繰り返し、収束するまで精度を向上させる。"},

    {"type": "heading_2", "text": "注意点"},
    {"type": "paragraph", "text": "初期位置が大きく異なる場合は誤った対応付けが発生しやすく、精度が低下する。"},

    {"type": "heading_2", "text": "Python実装"},
    {"type": "code", "text": """reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())"""},

    {"type": "heading_3", "text": "主要パラメータ"},
    {"type": "paragraph", "text": "source_pcdは移動させる点群、target_pcdは基準点群、thresholdは対応点の最大距離、trans_initは初期変換行列を表す。"},

    {"type": "heading_3", "text": "出力結果"},
    {"type": "bulleted_list_item", "text": "transformation：最適な変換行列"},
    {"type": "bulleted_list_item", "text": "fitness：一致率"},
    {"type": "bulleted_list_item", "text": "inlier_rmse：誤差指標"},

    {"type": "heading_2", "text": "ユースケース"},
    {"type": "paragraph", "text": "位置ズレした点群データを既存の正しい点群に合わせるために使用される。例えば異なる年度のLASデータの位置補正に利用される。"}
]

# ===== 通常テキスト =====
raw_documents = [
    """GIS（地理情報システム）は空間データを扱う技術です。
データの可視化や分析に利用されます。
WebGISの普及によりブラウザ上での活用も進んでいます。"""
]

# ===== Block → テキスト変換 =====
def block_to_text(block):
    t = block["type"]
    text = block["text"]

    if t == "heading_1":
        return f"# {text}"
    elif t == "heading_2":
        return f"## {text}"
    elif t == "heading_3":
        return f"### {text}"
    elif t == "bulleted_list_item":
        return f"- {text}"
    elif t == "code":
        return f"```\n{text}\n```"
    else:
        return text

# ===== TextSplitter =====
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "、", " ", ""]
)

documents = []
metadatas = []
ids = []

doc_id = 0

# ===== Notion Blocks処理 =====
for i, block in enumerate(icp_blocks):
    text = block_to_text(block)

    chunks = splitter.split_text(text)

    for chunk in chunks:
        documents.append(chunk)

        metadatas.append({
            "source": "notion",
            "block_type": block["type"],
            "block_index": i
        })

        ids.append(str(doc_id))
        doc_id += 1

# ===== 通常テキスト処理 =====
for i, doc in enumerate(raw_documents):
    chunks = splitter.split_text(doc)

    for chunk in chunks:
        documents.append(chunk)

        metadatas.append({
            "source": "raw_text",
            "block_index": i
        })

        ids.append(str(doc_id))
        doc_id += 1

# ===== デバッグ =====
print(f"チャンク数: {len(documents)}")
for i, d in enumerate(documents):
    print(f"\n--- chunk {i} ---")
    print(d)
    print("meta:", metadatas[i])

# ===== DB登録（初回のみ） =====
if collection.count() == 0:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

# ===== 検索ループ =====
print("=== RAG検索 ===")

THRESHOLD = 0.7  # ← ここ調整ポイント 厳しい

while True:
    query = input("\n質問してください > ").strip()

    if query.lower() in ["exit", "quit"]:
        print("終了します")
        break

    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # ===== rerank =====
    reranked = rerank(query, docs)

    # ===== フィルタ（ここ重要）=====
    filtered = [
        (doc, score)
        for doc, score in reranked
        if score >= THRESHOLD
    ]

    print("\n--- 検索結果（rerank + filter後） ---")

    if not filtered:
        print("該当する情報が見つかりませんでした")
        continue

    # 上位3件まで表示
    for i, (doc, score) in enumerate(filtered[:3]):
        idx = docs.index(doc)

        print(f"{i+1}. {doc}")
        print(f"   rerank_score: {score:.4f}")
        print(f"   original_distance: {dists[idx]:.4f}")
        print(f"   meta: {metas[idx]}")