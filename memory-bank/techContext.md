# 技術コンテキスト

## 使用技術の詳細

### 1. markitdown

**概要**:
- Microsoftが開発したPythonユーティリティ
- 様々なファイル形式をマークダウンに変換するためのツール
- LLMや関連するテキスト分析パイプラインでの使用に最適化

**主要機能**:
- PDF、PowerPoint、Word、Excel、画像、音声などの多様なファイル形式をサポート
- 重要な文書構造（見出し、リスト、表、リンクなど）をマークダウンとして保持
- プラグインシステムによる拡張性

**使用方法**:
```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("document.pdf")
markdown_text = result.markdown
```

**依存関係**:
- 基本機能: beautifulsoup4, requests, markdownify, magika, charset-normalizer
- オプション機能: python-pptx, mammoth, pandas, openpyxl, xlrd, lxml, pdfminer.six, olefile, pydub, SpeechRecognition, youtube-transcript-api, azure-ai-documentintelligence, azure-identity

### 2. LangChain

**概要**:
- LLMアプリケーション開発のためのフレームワーク
- 複数のコンポーネントを組み合わせて高度なAIアプリケーションを構築可能

**主要機能**:
- ドキュメント処理（ローディング、分割、変換）
- 埋め込み生成と管理
- ベクトルストアとの統合
- プロンプトエンジニアリング
- LLMとの統合

**使用方法**:
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document

# 埋め込み生成
embeddings = OpenAIEmbeddings()

# テキスト分割
text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(text)

# ベクトルストア
db = Chroma(persist_directory="./data", embedding_function=embeddings)
db.add_documents([Document(page_content=chunk) for chunk in chunks])

# 検索
docs = db.similarity_search("query", k=4)
```

### 3. Streamlit

**概要**:
- Pythonベースのウェブアプリケーションフレームワーク
- データサイエンスやAIアプリケーションの迅速な開発に最適化

**主要機能**:
- インタラクティブなUIコンポーネント
- データの可視化
- セッション状態管理
- キャッシュ機能
- ファイルアップロード/ダウンロード

**使用方法**:
```python
import streamlit as st

st.title("MarkItDown RAG アプリケーション")

uploaded_file = st.file_uploader("ドキュメントをアップロード", type=["pdf", "docx", "xlsx"])
if uploaded_file is not None:
    if st.button("処理"):
        with st.spinner("処理中..."):
            # 処理ロジック
            st.success("処理完了")

query = st.text_input("質問を入力してください")
if query:
    with st.spinner("検索中..."):
        # 検索ロジック
        st.write("回答: ...")
```

### 4. ChromaDB

**概要**:
- オープンソースの埋め込みベクトルデータベース
- 高速な類似度検索を提供

**主要機能**:
- 埋め込みベクトルの保存と検索
- メタデータのフィルタリング
- 永続化オプション
- LangChainとの統合

**使用方法**:
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

# ドキュメント追加
collection.add(
    documents=["document content"],
    metadatas=[{"source": "file.pdf"}],
    ids=["doc1"]
)

# 検索
results = collection.query(
    query_texts=["search query"],
    n_results=2
)
```

### 5. OpenAI API

**概要**:
- OpenAIが提供するAPIサービス
- テキスト埋め込み、テキスト生成などの機能を提供

**主要機能**:
- テキスト埋め込み生成
- テキスト生成
- チャット対話

**使用方法**:
```python
from openai import OpenAI

client = OpenAI()

# 埋め込み生成
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="テキスト"
)
embedding = response.data[0].embedding

# テキスト生成
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
```

## 開発環境

### 1. Python環境

- Python 3.12+
- 仮想環境管理: venv または Conda
- パッケージ管理: pip

### 2. 開発ツール

- **コードエディタ**: VS Code
- **バージョン管理**: Git
- **リンティング**: Ruff
- **型チェック**: mypy
- **テスト**: pytest, pytest-cov

### 3. 環境変数

```
OPENAI_API_KEY=your_openai_api_key
```

## 技術的制約

### 1. パフォーマンス制約

- 大きなファイルの処理には時間がかかる可能性がある
- 埋め込みモデルのAPIリクエストにはレート制限がある
- ベクトルデータベースのサイズはメモリ制約を受ける

### 2. セキュリティ制約

- OpenAI APIキーの安全な管理が必要
- アップロードされたドキュメントの安全な処理と保存
- ユーザークエリとレスポンスの適切な検証

### 3. スケーラビリティ制約

- Streamlitアプリケーションは大規模な同時アクセスには最適化されていない
- ローカルChromaDBは大量のドキュメントには制限がある
- 処理能力はホストマシンのリソースに依存

## 依存関係管理

### 1. 必須依存関係

```toml
[project]
dependencies = [
    "markitdown[all]>=0.1.1",
    "langchain>=0.1.0",
    "langchain-openai>=0.0.2",
    "streamlit>=1.32.0",
    "chromadb>=0.4.22",
    "python-dotenv>=1.0.0",
]
```

### 2. 開発依存関係

```toml
[project.optional-dependencies]
dev = [
    "pytest==8.3.5",
    "pytest-cov==6.0.0",
    "ruff==0.11.2",
    "mypy==1.14.1",
]
```

### 3. ツール設定

```toml
[tool.ruff]
line-length = 88
target-version = "py312"
select = ["E", "F", "I", "N", "W", "B", "A", "C4", "UP", "ANN", "RUF"]
ignore = ["ANN101"]  # self の型アノテーションは不要

[tool.ruff.format]
quote-style = "single"
indent-style = "space"
line-ending = "auto"

[tool.mypy]
python_version = "3.12"
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
strict_optional = true
warn_redundant_casts = true
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "--cov=src"
