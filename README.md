# MarkItDown RAG サンプルアプリケーション

Microsoftの「markitdown」リポジトリを活用して、あらゆるデータをマークダウンに変換し、RAG（Retrieval-Augmented Generation）機能とドキュメント管理UIを備えた検証用サンプルアプリケーションです。

## 主要機能

- **ドキュメント変換機能**
  - markitdownを使用して様々な形式のファイルをマークダウンに変換
  - 対応形式:
    - PDF (.pdf)
    - Word (.docx)
    - PowerPoint (.pptx)
    - Excel (.xlsx)
    - テキスト (.txt)
    - Markdown (.md)
    - HTML (.html, .htm)
    - テキストベースの形式 (.csv, .json, .xml)
    - 画像ファイル (.jpg, .jpeg, .png)
  - 変換したマークダウンの保存と管理

- **RAG機能**
  - 変換したマークダウンドキュメントを検索可能なデータベースに格納
  - OpenAIの埋め込みモデルを使用してドキュメントのベクトル化
  - クエリに基づいて関連情報を取得
  - LangChainを使用したRAG機能の実装

- **ドキュメント管理UI**
  - Streamlitを使用したウェブインターフェース
  - ドキュメントのアップロード、検索、閲覧機能

## 技術スタック

- **言語**: Python 3.12+
- **フレームワーク**:
  - Streamlit: UIとバックエンドロジックを統合
  - LangChain: RAG機能の実装
- **ライブラリ**:
  - markitdown: ドキュメント変換
  - OpenAI API: 埋め込みモデル
  - ChromaDB: ベクトルデータベース
- **開発ツール**:
  - pytest: テスト
  - ruff: リンティングとフォーマット
  - mypy: 静的型チェック

## セットアップ

1. リポジトリをクローン
   ```bash
   git clone https://github.com/nichiki/markitdown-rag-sample.git
   cd markitdown-rag-sample
   ```

2. 仮想環境を作成して有効化
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linuxの場合
   # または
   venv\Scripts\activate  # Windowsの場合
   ```

3. 依存関係をインストール
   ```bash
   # 依存関係をインストール
   uv sync
   ```

4. 環境変数を設定
   ```bash
   cp .env.example .env
   # .envファイルを編集してOPENAI_API_KEYを設定
   ```

## 使用方法

### アプリケーションの実行

```bash
streamlit run src/app.py
```

### ドキュメントの変換と検索

1. Streamlit UIを通じてドキュメントをアップロード
2. 「処理」ボタンをクリックしてドキュメントを変換
3. 検索ボックスにクエリを入力して関連情報を検索

### コード品質管理

- **自動フォーマット**: コードは Ruff によって保存時に自動的にフォーマットされます
  ```bash
  # 手動でフォーマットする場合
  ruff format .
  ```

- **静的型チェック**: mypy による厳格な型チェックが適用されます
  ```bash
  mypy src
  ```

- **テスト実行**: pytest でテストを実行し、カバレッジレポートを生成
  ```bash
  pytest
  # または詳細なカバレッジレポートを表示
  pytest --cov=src --cov-report=term-missing
  ```

### コーディング規約

このプロジェクトでは、以下のコーディング規約に従います：

- すべてのファイル、クラス、関数、メソッドに docstring を記述
- すべての関数とメソッドに型アノテーションを使用
- インポートは標準ライブラリ、サードパーティ、ローカルの順に整理
- 可能な限り関数型アプローチを採用（純粋関数、不変データ構造）
- クラスを使用する場合は単一責任の原則を守る

### テスト要件

テストは `tests/` ディレクトリに配置し、以下の原則に従います：

- テスト駆動開発（TDD）のアプローチを推奨
- すべての公開関数とメソッドにはユニットテストを作成
- コードカバレッジは 80% 以上を目標
- 外部依存はモックを使用

## トラブルシューティング

### よくある問題と解決方法

1. **OpenAI APIキーの設定**
   - 問題: `OpenAI API key not found` というエラーが表示される
   - 解決: `.env` ファイルに `OPENAI_API_KEY=your_api_key` を設定してください

2. **ファイルのアップロードエラー**
   - 問題: ファイルのアップロードに失敗する
   - 解決: ファイルサイズが200MB以下であることを確認してください

3. **ベクトルデータベースのエラー**
   - 問題: `ChromaDB` 関連のエラーが表示される
   - 解決: `data/embeddings` ディレクトリを削除して再試行してください

## ディレクトリ構造

```
markitdown-rag-sample/
├── src/                   # メインのソースコード
│   ├── core/              # コアロジック
│   │   ├── __init__.py
│   │   ├── document_processor.py  # markitdownを使用したドキュメント処理
│   │   └── rag.py         # LangChainを使用したRAG機能
│   ├── db/                # データベース関連
│   │   ├── __init__.py
│   │   └── vector_store.py  # ベクトルストア操作（ChromaDB）
│   └── app.py             # Streamlitアプリケーション
├── tests/                 # テストコード
├── data/                  # データ保存ディレクトリ（.gitignoreに含まれる）
│   ├── raw/               # 元のドキュメント
│   ├── processed/         # 処理済みマークダウン
│   └── embeddings/        # 埋め込みデータ
├── docs/                  # ドキュメント
│   └── references/        # 参考資料
├── .env.example           # 環境変数の例
├── .gitignore             # Git の除外設定
├── README.md              # このファイル
├── mypy.ini               # mypy設定
└── pyproject.toml         # プロジェクト設定
