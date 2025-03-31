"""RAGクラスのテスト。

このモジュールは、RAGクラスの機能をテストします。
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from src.core.rag import RAG, RAGError, RAGResponse
from src.db.vector_store import VectorStore


class TestRAG:
    """RAGクラスのテストケース。"""

    def setup_method(self) -> None:
        """各テストメソッドの前に実行される。"""
        # モックのVectorStoreを作成
        self.mock_vector_store = MagicMock(spec=VectorStore)

        # RAGインスタンスを作成 (モックのVectorStoreを使用)
        self.rag = RAG(vector_store=self.mock_vector_store, api_key='dummy_key')

    @patch('src.core.rag.ChatOpenAI')
    def test_search(self, mock_chat_openai_class: Mock) -> None:
        """検索機能のテスト。"""
        # モックの検索結果を設定
        mock_search_results = [
            {
                'content': '検索結果1',
                'metadata': {'source': 'doc1.md'},
                'score': 0.95,
            },
            {
                'content': '検索結果2',
                'metadata': {'source': 'doc2.md'},
                'score': 0.85,
            },
        ]
        self.mock_vector_store.search.return_value = mock_search_results

        # テスト対象メソッドの実行
        results = self.rag.search('テストクエリ', k=2)

        # 検証
        assert len(results) == 2
        assert results[0].content == '検索結果1'
        assert results[0].metadata == {'source': 'doc1.md'}
        assert results[0].score == 0.95
        assert results[1].content == '検索結果2'
        assert results[1].metadata == {'source': 'doc2.md'}
        assert results[1].score == 0.85

        # モックの呼び出しを検証
        self.mock_vector_store.search.assert_called_once_with(
            'テストクエリ', k=2, filter_metadata=None
        )

    @patch('src.core.rag.ChatOpenAI')
    def test_search_with_filter(self, mock_chat_openai_class: Mock) -> None:
        """フィルタ付き検索機能のテスト。"""
        # モックの検索結果を設定
        mock_search_results = [
            {
                'content': '検索結果1',
                'metadata': {'source': 'doc1.md', 'category': 'テスト'},
                'score': 0.95,
            }
        ]
        self.mock_vector_store.search.return_value = mock_search_results

        # テスト対象メソッドの実行
        filter_metadata = {'category': 'テスト'}
        results = self.rag.search('テストクエリ', k=1, filter_metadata=filter_metadata)

        # 検証
        assert len(results) == 1
        assert results[0].content == '検索結果1'
        assert results[0].metadata == {'source': 'doc1.md', 'category': 'テスト'}
        assert results[0].score == 0.95

        # モックの呼び出しを検証
        self.mock_vector_store.search.assert_called_once_with(
            'テストクエリ', k=1, filter_metadata=filter_metadata
        )

    @patch('src.core.rag.ChatOpenAI')
    def test_search_error(self, mock_chat_openai_class: Mock) -> None:
        """検索エラーのテスト。"""
        # モックの検索エラーを設定
        self.mock_vector_store.search.side_effect = Exception('検索エラー')

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(RAGError) as excinfo:
            self.rag.search('テストクエリ')

        # 例外メッセージの検証
        assert '予期しない検索エラー' in str(excinfo.value)

    @patch('src.core.rag.ChatOpenAI')
    def test_generate(self, mock_chat_openai_class: Mock) -> None:
        """回答生成機能のテスト。"""
        # モックのLLMを設定
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = 'テスト回答'
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai_class.return_value = mock_llm

        # テスト用のコンテキスト
        context = ['コンテキスト1', 'コンテキスト2']

        # テスト対象メソッドの実行
        answer = self.rag.generate('テストクエリ', context)

        # 検証
        assert answer == 'テスト回答'

        # モックの呼び出しを検証
        mock_llm.invoke.assert_called_once()
        # 詳細な引数の検証は省略 (プロンプトの内容が複雑なため)

    @patch('src.core.rag.ChatOpenAI')
    def test_generate_error(self, mock_chat_openai_class: Mock) -> None:
        """回答生成エラーのテスト。"""
        # モックのLLMを設定
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception('生成エラー')
        mock_chat_openai_class.return_value = mock_llm

        # テスト用のコンテキスト
        context = ['コンテキスト1', 'コンテキスト2']

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(RAGError) as excinfo:
            self.rag.generate('テストクエリ', context)

        # 例外メッセージの検証
        assert '回答生成に失敗しました' in str(excinfo.value)

    @patch('src.core.rag.ChatOpenAI')
    def test_query(self, mock_chat_openai_class: Mock) -> None:
        """クエリ実行機能のテスト。"""
        # モックの検索結果を設定
        mock_search_results = [
            {
                'content': '検索結果1',
                'metadata': {'source': 'doc1.md'},
                'score': 0.95,
            },
            {
                'content': '検索結果2',
                'metadata': {'source': 'doc2.md'},
                'score': 0.85,
            },
        ]
        self.mock_vector_store.search.return_value = mock_search_results

        # モックのLLMを設定
        mock_llm = MagicMock()
        mock_response = AIMessage(content='テスト回答')
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai_class.return_value = mock_llm

        # テスト対象メソッドの実行
        response = self.rag.query('テストクエリ')

        # 検証
        assert isinstance(response, RAGResponse)
        assert response.answer == 'テスト回答'
        assert len(response.sources) == 2
        assert response.sources[0].content == '検索結果1'
        assert response.sources[0].metadata == {'source': 'doc1.md'}
        assert response.sources[0].score == 0.95
        assert response.sources[1].content == '検索結果2'
        assert response.sources[1].metadata == {'source': 'doc2.md'}
        assert response.sources[1].score == 0.85

    @patch('src.core.rag.ChatOpenAI')
    def test_query_no_results(self, mock_chat_openai_class: Mock) -> None:
        """検索結果がない場合のクエリ実行機能のテスト。"""
        # 空の検索結果を設定
        self.mock_vector_store.search.return_value = []

        # テスト対象メソッドの実行
        response = self.rag.query('テストクエリ')

        # 検証
        assert isinstance(response, RAGResponse)
        assert response.answer == '関連する情報が見つかりませんでした。'
        assert len(response.sources) == 0

    @patch('src.core.rag.ChatOpenAI')
    def test_query_error(self, mock_chat_openai_class: Mock) -> None:
        """クエリ実行エラーのテスト。"""
        # モックの検索エラーを設定
        self.mock_vector_store.search.side_effect = Exception('検索エラー')

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(RAGError) as excinfo:
            self.rag.query('テストクエリ')

        # 例外メッセージの検証
        assert 'RAG処理に失敗しました' in str(excinfo.value)
