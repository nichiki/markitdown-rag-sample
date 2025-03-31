"""ベクトルデータベース操作モジュール。

このモジュールは、ドキュメントのベクトル化と検索機能を提供します。
"""

import os
import uuid
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from pydantic import SecretStr


class VectorStoreError(Exception):
    """ベクトルストア操作中に発生するエラー。"""

    pass


class VectorStore:
    """ベクトルストアクラス。

    このクラスは、ドキュメントのチャンク分割、埋め込み生成、
    ベクトルデータベースへの保存、検索機能を提供します。
    """

    def __init__(
        self,
        persist_directory: str = './data/embeddings',
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        api_key: str | None = None,
    ) -> None:
        """VectorStoreのインスタンスを初期化します。

        Args:
            persist_directory: ベクトルデータベースの保存先ディレクトリ
            chunk_size: テキスト分割時のチャンクサイズ
            chunk_overlap: テキスト分割時のチャンクオーバーラップ
            api_key: OpenAI APIキー(Noneの場合は環境変数から取得)
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key

        # 保存先ディレクトリが存在しない場合は作成
        os.makedirs(persist_directory, exist_ok=True)

        # テキスト分割器の初期化
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # 埋め込みモデルとベクトルデータベースは遅延初期化
        self._embedding_model: OpenAIEmbeddings | None = None
        self._db: Chroma | None = None

    @property
    def embedding_model(self) -> OpenAIEmbeddings:
        """埋め込みモデルを取得します。初回アクセス時に初期化します。"""
        if self._embedding_model is None:
            # api_keyがstrの場合はSecretStrに変換
            secret_key = SecretStr(self.api_key) if self.api_key else None
            self._embedding_model = OpenAIEmbeddings(api_key=secret_key)
        return self._embedding_model

    @property
    def db(self) -> Chroma:
        """ベクトルデータベースを取得します。初回アクセス時に初期化します。"""
        if self._db is None:
            self._db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
            )
        return self._db

    def split_text(self, text: str) -> list[str]:
        """テキストをチャンクに分割します。

        Args:
            text: 分割するテキスト

        Returns:
            List[str]: 分割されたテキストチャンクのリスト
        """
        return self.text_splitter.split_text(text)

    def generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """テキストチャンクの埋め込みベクトルを生成します。

        Args:
            chunks: 埋め込みを生成するテキストチャンクのリスト

        Returns:
            List[List[float]]: 埋め込みベクトルのリスト
        """
        # OpenAIEmbeddingsのembed_documentsメソッドは実際にはlist[list[float]]を返す
        embeddings = self.embedding_model.embed_documents(chunks)
        return embeddings

    def add_document(self, markdown_content: str, metadata: dict[str, Any]) -> None:
        """ドキュメントをベクトルデータベースに追加します。

        Args:
            markdown_content: マークダウン形式のドキュメント内容
            metadata: ドキュメントのメタデータ

        Raises:
            VectorStoreError: ドキュメントの追加に失敗した場合
        """
        try:
            # テキストをチャンクに分割
            chunks = self.split_text(markdown_content)

            # 各チャンクにメタデータを追加
            metadatas = []
            for i, _ in enumerate(chunks):
                # 元のメタデータをコピー
                chunk_metadata = metadata.copy()
                # チャンクIDを追加
                chunk_metadata['chunk_id'] = str(uuid.uuid4())
                # チャンクインデックスを追加
                chunk_metadata['chunk_index'] = i
                metadatas.append(chunk_metadata)

            # ベクトルデータベースに追加
            self.db.add_texts(texts=chunks, metadatas=metadatas)
        except Exception as e:
            raise VectorStoreError(f'ドキュメントの追加に失敗しました: {e}') from e

    def search(
        self, query: str, k: int = 4, filter_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """クエリに基づいて関連ドキュメントを検索します。

        Args:
            query: 検索クエリ
            k: 返す結果の数
            filter_metadata: 検索結果をフィルタリングするためのメタデータ

        Returns:
            List[Dict[str, Any]]: 検索結果のリスト。各結果は以下の形式:
                {
                    'content': str,  # ドキュメントの内容
                    'metadata': Dict[str, Any],  # ドキュメントのメタデータ
                    'score': float  # 関連性スコア
                }

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        try:
            # 検索を実行
            if filter_metadata:
                results = self.db.similarity_search_with_relevance_scores(
                    query, k=k, filter=filter_metadata
                )
            else:
                results = self.db.similarity_search_with_relevance_scores(query, k=k)

            # 結果を整形
            formatted_results = []
            for doc, score in results:
                formatted_results.append(
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': score,
                    }
                )

            return formatted_results
        except Exception as e:
            raise VectorStoreError(f'検索に失敗しました: {e}') from e


if __name__ == '__main__':
    import argparse
    import json
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    # .envファイルから環境変数を読み込む
    load_dotenv()

    def main() -> int:
        """メイン関数。

        コマンドライン引数を解析し、ベクトルストア操作を実行します。

        Returns:
            int: 終了コード(0: 成功、1: 失敗)
        """
        parser = argparse.ArgumentParser(description='ベクトルデータベース操作ツール')
        subparsers = parser.add_subparsers(dest='command', help='サブコマンド')

        # addコマンド
        add_parser = subparsers.add_parser('add', help='ドキュメントを追加')
        add_parser.add_argument('file_path', help='追加するマークダウンファイルのパス')
        add_parser.add_argument(
            '-d',
            '--db-dir',
            default='./data/embeddings',
            help='ベクトルデータベースのディレクトリ',
        )
        add_parser.add_argument(
            '-m', '--metadata', default='{}', help='JSONフォーマットのメタデータ'
        )

        # searchコマンド
        search_parser = subparsers.add_parser('search', help='ドキュメントを検索')
        search_parser.add_argument('query', help='検索クエリ')
        search_parser.add_argument(
            '-d',
            '--db-dir',
            default='./data/embeddings',
            help='ベクトルデータベースのディレクトリ',
        )
        search_parser.add_argument(
            '-k', '--top-k', type=int, default=4, help='返す結果の数'
        )
        search_parser.add_argument(
            '-f', '--filter', default='{}', help='JSONフォーマットのフィルタメタデータ'
        )

        # demoコマンド(動作確認用)
        demo_parser = subparsers.add_parser('demo', help='動作確認用デモ')
        demo_parser.add_argument(
            '-d',
            '--db-dir',
            default='./data/embeddings',
            help='ベクトルデータベースのディレクトリ',
        )

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return 0

        try:
            # ベクトルストアのインスタンスを作成
            vector_store = VectorStore(persist_directory=args.db_dir)

            if args.command == 'add':
                # ファイルを読み込む
                file_path = args.file_path
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()

                # メタデータを解析
                metadata = json.loads(args.metadata)

                # ファイル名をメタデータに追加
                if 'source' not in metadata:
                    metadata['source'] = Path(file_path).name

                # ドキュメントを追加
                print(f'ドキュメントを追加中: {file_path}')
                vector_store.add_document(content, metadata)
                # 注: Chroma 0.4.x以降は自動的に永続化されるため、persist()は不要
                print('ドキュメントの追加が完了しました')

            elif args.command == 'search':
                # フィルタを解析
                filter_metadata = json.loads(args.filter) if args.filter else None

                # 検索を実行
                print(f'検索クエリ: {args.query}')
                results = vector_store.search(
                    args.query, k=args.top_k, filter_metadata=filter_metadata
                )

                # 結果を表示
                print(f'\n検索結果 ({len(results)}件):\n')
                for i, result in enumerate(results):
                    print(f'結果 {i + 1} (スコア: {result["score"]:.4f}):')
                    print(f'ソース: {result["metadata"].get("source", "Unknown")}')
                    print(f'内容: {result["content"][:200]}...')
                    print()

            elif args.command == 'demo':
                # デモ用のサンプルマークダウンを作成
                sample_markdown = """# ベクトルデータベースのデモ

## はじめに

このデモでは、ベクトルデータベースの基本的な機能を紹介します。

## ベクトルデータベースとは

ベクトルデータベースは、テキストや画像などのデータを高次元ベクトルとして保存し、
類似度検索を効率的に行うためのデータベースです。

## 主な用途

1. 意味検索
2. レコメンデーションシステム
3. 異常検出
4. 画像検索

## LangChainとChromaDB

LangChainとChromaDBを組み合わせることで、簡単にベクトルデータベースを構築できます。
"""

                # サンプルメタデータ
                sample_metadata = {
                    'source': 'demo_document.md',
                    'author': 'VectorStore Demo',
                    'created_at': '2025-03-29',
                }

                # ドキュメントを追加
                print('サンプルドキュメントを追加中...')
                vector_store.add_document(sample_markdown, sample_metadata)
                # 注: Chroma 0.4.x以降は自動的に永続化されるため、persist()は不要
                print('サンプルドキュメントの追加が完了しました')

                # 検索クエリのサンプル
                sample_queries = [
                    'ベクトルデータベースとは何ですか',
                    'ベクトルデータベースの用途を教えてください',
                    'LangChainとChromaDBの関係は',
                ]

                # 各クエリで検索を実行
                for query in sample_queries:
                    print(f'\n検索クエリ: {query}')
                    results = vector_store.search(query, k=2)

                    # 結果を表示
                    print(f'検索結果 ({len(results)}件):')
                    for i, result in enumerate(results):
                        print(f'結果 {i + 1} (スコア: {result["score"]:.4f}):')
                        print(f'内容: {result["content"]}')
                        print()

            return 0
        except VectorStoreError as e:
            print(f'エラー: {e}', file=sys.stderr)
            return 1
        except Exception as e:
            print(f'予期しないエラー: {e}', file=sys.stderr)
            return 1

    sys.exit(main())
