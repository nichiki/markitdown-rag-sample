"""RAG (Retrieval-Augmented Generation) モジュール。

このモジュールは、ドキュメントの検索と生成を組み合わせた
RAG機能を提供します。
"""

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from db.vector_store import VectorStore, VectorStoreError


class RAGError(Exception):
    """RAG処理中に発生するエラー。"""

    pass


class SearchResult(BaseModel):
    """検索結果を表すモデル。"""

    content: str = Field(..., description='検索結果のコンテンツ')
    metadata: dict[str, Any] = Field(..., description='検索結果のメタデータ')
    score: float = Field(..., description='検索結果のスコア')


class RAGResponse(BaseModel):
    """RAGレスポンスを表すモデル。"""

    answer: str = Field(..., description='生成された回答')
    sources: list[SearchResult] = Field(..., description='回答の情報源')


class RAG:
    """RAG (Retrieval-Augmented Generation) クラス。

    このクラスは、ドキュメントの検索と生成を組み合わせた
    RAG機能を提供します。
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embeddings_dir: str = './data/embeddings',
        api_key: str | None = None,
        model: str = 'gpt-3.5-turbo',
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> None:
        """RAGのインスタンスを初期化します。

        Args:
            vector_store: 使用するVectorStoreインスタンス (Noneの場合は新規作成)
            embeddings_dir: 埋め込みデータの保存ディレクトリ
            api_key: OpenAI APIキー (Noneの場合は環境変数から取得)
            model: 使用するモデル名
            temperature: 生成時の温度パラメータ
            max_tokens: 生成する最大トークン数
        """
        # ベクトルストアの初期化
        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore(persist_directory=embeddings_dir, api_key=api_key)
        )

        # APIキーの設定
        self.api_key = api_key

        # モデルパラメータの設定
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # LLMの初期化 (遅延初期化)
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """LLMを取得します。初回アクセス時に初期化します。"""
        if self._llm is None:
            # api_keyがstrの場合はSecretStrに変換
            secret_key = SecretStr(self.api_key) if self.api_key else None
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=secret_key,
            )
        return self._llm

    def search(
        self, query: str, k: int = 4, filter_metadata: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """クエリに基づいて関連ドキュメントを検索します。

        Args:
            query: 検索クエリ
            k: 返す結果の数
            filter_metadata: 検索結果をフィルタリングするためのメタデータ

        Returns:
            List[SearchResult]: 検索結果のリスト

        Raises:
            RAGError: 検索に失敗した場合
        """
        try:
            # ベクトルストアで検索を実行
            results = self.vector_store.search(
                query, k=k, filter_metadata=filter_metadata
            )

            # 検索結果をSearchResultモデルに変換
            return [SearchResult(**result) for result in results]
        except VectorStoreError as e:
            raise RAGError(f'検索に失敗しました: {e}') from e
        except Exception as e:
            raise RAGError(f'予期しない検索エラー: {e}') from e

    def generate(self, query: str, context: list[str]) -> str:
        """コンテキストを使用して回答を生成します。

        Args:
            query: ユーザーのクエリ
            context: 回答生成に使用するコンテキスト情報のリスト

        Returns:
            str: 生成された回答

        Raises:
            RAGError: 回答生成に失敗した場合
        """
        try:
            # コンテキストを結合
            combined_context = '\n\n'.join(context)

            # プロンプトの作成
            system_prompt = f"""あなたは情報提供を行うアシスタントです。
以下のコンテキスト情報を使用して、ユーザーのクエリに応答してください。
ユーザーのクエリがキーワードのみの場合は、そのキーワードに関連する情報を要約して提供してください。
質問形式の場合は、質問に直接答えてください。
コンテキスト情報に含まれていない場合は、「その情報はコンテキストに含まれていません」と答えてください。

コンテキスト情報:
{combined_context}
"""

            # LLMで回答を生成
            response = self.llm.invoke(
                [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query},
                ]
            )

            # AIMessageのcontentはstr型だが、型チェッカーはより広い型を推論するため
            # 明示的にキャスト
            return str(response.content)
        except Exception as e:
            raise RAGError(f'回答生成に失敗しました: {e}') from e

    def query(
        self,
        query: str,
        k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """クエリに対してRAGを実行します。

        Args:
            query: ユーザーのクエリ
            k: 検索結果の数
            filter_metadata: 検索結果をフィルタリングするためのメタデータ

        Returns:
            RAGResponse: 生成された回答と情報源

        Raises:
            RAGError: RAG処理に失敗した場合
        """
        try:
            # 関連ドキュメントを検索
            search_results = self.search(query, k=k, filter_metadata=filter_metadata)

            # 検索結果がない場合
            if not search_results:
                return RAGResponse(
                    answer='関連する情報が見つかりませんでした。', sources=[]
                )

            # 検索結果からコンテキストを抽出
            context = [result.content for result in search_results]

            # 回答を生成
            answer = self.generate(query, context)

            # 結果を返す
            return RAGResponse(answer=answer, sources=search_results)
        except Exception as e:
            raise RAGError(f'RAG処理に失敗しました: {e}') from e


if __name__ == '__main__':
    import argparse
    import json
    import sys

    from dotenv import load_dotenv

    # .envファイルから環境変数を読み込む
    load_dotenv()

    def main() -> int:
        """メイン関数。

        コマンドライン引数を解析し、RAG処理を実行します。

        Returns:
            int: 終了コード(0: 成功、1: 失敗)
        """
        parser = argparse.ArgumentParser(description='RAG処理を実行します。')
        parser.add_argument('query', help='検索クエリ')
        parser.add_argument(
            '-d',
            '--db-dir',
            default='./data/embeddings',
            help='ベクトルデータベースのディレクトリ',
        )
        parser.add_argument('-k', '--top-k', type=int, default=4, help='返す結果の数')
        parser.add_argument(
            '-f', '--filter', default='{}', help='JSONフォーマットのフィルタメタデータ'
        )
        parser.add_argument(
            '-m',
            '--model',
            default='gpt-3.5-turbo',
            help='使用するOpenAIモデル',
        )
        parser.add_argument(
            '-t',
            '--temperature',
            type=float,
            default=0.0,
            help='生成時の温度パラメータ',
        )
        args = parser.parse_args()

        try:
            # フィルタを解析
            filter_metadata = json.loads(args.filter) if args.filter else None

            # RAGインスタンスを作成
            rag = RAG(
                embeddings_dir=args.db_dir,
                model=args.model,
                temperature=args.temperature,
            )

            # RAG処理を実行
            print(f'クエリ: {args.query}')
            response = rag.query(
                args.query, k=args.top_k, filter_metadata=filter_metadata
            )

            # 結果を表示
            print('\n回答:')
            print(response.answer)

            print('\n情報源:')
            for i, source in enumerate(response.sources):
                print(f'情報源 {i + 1} (スコア: {source.score:.4f}):')
                print(f'ソース: {source.metadata.get("source", "Unknown")}')
                print(f'内容: {source.content[:200]}...')
                print()

            return 0
        except RAGError as e:
            print(f'エラー: {e}', file=sys.stderr)
            return 1
        except Exception as e:
            print(f'予期しないエラー: {e}', file=sys.stderr)
            return 1

    sys.exit(main())
