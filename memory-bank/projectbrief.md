# プロジェクト概要

## 目的

Microsoftの「markitdown」リポジトリを活用して、あらゆるデータをマークダウンに変換し、RAG（Retrieval-Augmented Generation）機能とドキュメント管理UIを備えた検証用アプリケーションを開発する。

## 主要機能

1. **ドキュメント変換機能**
   - markitdownを使用して様々な形式のファイル（PDF、Word、Excel、画像、音声など）をマークダウンに変換
   - 変換したマークダウンの保存と管理

2. **RAG機能**
   - 変換したマークダウンドキュメントを検索可能なデータベースに格納
   - OpenAIの埋め込みモデルを使用してドキュメントのベクトル化
   - クエリに基づいて関連情報を取得
   - LangChainを使用したRAG機能の実装

3. **ドキュメント管理UI**
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

## 開発アプローチ

- テスト駆動開発（TDD）を採用
- 段階的な開発と継続的な検証
- 小さな成功を積み重ねるアプローチ
- Googleスタイルのdocstring形式を使用
- 型アノテーションを徹底
- 関数型アプローチを優先（純粋関数、不変データ構造）

## プロジェクト構造

```
/
├── src/
│   ├── core/                 # コアロジック
│   │   ├── __init__.py
│   │   ├── document_processor.py  # markitdownを使用したドキュメント処理
│   │   ├── embeddings.py     # 埋め込み生成（OpenAI経由）
│   │   └── rag.py            # LangChainを使用したRAG機能
│   ├── db/                   # データベース関連
│   │   ├── __init__.py
│   │   └── vector_store.py   # ベクトルストア操作（ChromaDB）
│   └── app.py                # Streamlitアプリケーション
├── tests/                    # テストコード
├── data/                     # データ保存ディレクトリ
│   ├── raw/                  # 元のドキュメント
│   ├── processed/            # 処理済みマークダウン
│   └── embeddings/           # 埋め込みデータ
├── pyproject.toml            # プロジェクト設定
└── README.md                 # プロジェクト説明
