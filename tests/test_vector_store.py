"""VectorStoreクラスのテスト。

このモジュールは、VectorStoreクラスの機能をテストします。
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.db.vector_store import VectorStore, VectorStoreError


class TestVectorStore:
    """VectorStoreクラスのテストケース。"""

    def setup_method(self) -> None:
        """各テストメソッドの前に実行される。"""
        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()
        self.persist_directory = os.path.join(self.temp_dir, 'chroma_db')

        # テスト用のマークダウンコンテンツ
        self.markdown_content = """# テストドキュメント

これはテスト用のマークダウンドキュメントです。

## セクション1

これはセクション1の内容です。テストのためのテキストが含まれています。

## セクション2

これはセクション2の内容です。別のテキストが含まれています。
"""

        # テスト用のメタデータ
        self.metadata = {'source': 'test_document.md', 'created_at': '2025-03-29'}

        # VectorStoreのインスタンスを作成(APIキーなし)
        # テスト時はモックを使用するため、実際のAPIキーは不要
        self.vector_store = VectorStore(
            persist_directory=self.persist_directory, api_key='dummy_key'
        )

    def teardown_method(self) -> None:
        """各テストメソッドの後に実行される。"""
        # 一時ディレクトリを削除
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch('src.db.vector_store.MarkdownTextSplitter')
    def test_split_text(self, mock_splitter_class: Mock) -> None:
        """テキスト分割機能のテスト。"""
        # モックの設定
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = [
            '# テストドキュメント\n\nこれはテスト用のマークダウンドキュメントです。',
            (
                '## セクション1\n\nこれはセクション1の内容です。'
                'テストのためのテキストが含まれています。'
            ),
            (
                '## セクション2\n\nこれはセクション2の内容です。'
                '別のテキストが含まれています。'
            ),
        ]
        mock_splitter_class.return_value = mock_splitter

        # text_splitterプロパティをモックに置き換え
        self.vector_store.text_splitter = mock_splitter

        # テスト対象メソッドの実行
        chunks = self.vector_store.split_text(self.markdown_content)

        # 検証
        assert len(chunks) == 3
        assert (
            chunks[0]
            == '# テストドキュメント\n\nこれはテスト用のマークダウンドキュメントです。'
        )
        assert (
            chunks[1] == '## セクション1\n\nこれはセクション1の内容です。'
            'テストのためのテキストが含まれています。'
        )
        assert (
            chunks[2] == '## セクション2\n\nこれはセクション2の内容です。'
            '別のテキストが含まれています。'
        )

        # モックの呼び出しを検証
        mock_splitter.split_text.assert_called_once_with(self.markdown_content)

    @patch('src.db.vector_store.OpenAIEmbeddings')
    def test_generate_embeddings(self, mock_embeddings_class: Mock) -> None:
        """埋め込み生成機能のテスト。"""
        # モックの設定
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        mock_embeddings_class.return_value = mock_embeddings

        # テスト用のテキストチャンク
        chunks = ['チャンク1', 'チャンク2', 'チャンク3']

        # テスト対象メソッドの実行
        embeddings = self.vector_store.generate_embeddings(chunks)

        # 検証
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert embeddings[2] == [0.7, 0.8, 0.9]

        # モックの呼び出しを検証
        mock_embeddings.embed_documents.assert_called_once_with(chunks)

    @patch('src.db.vector_store.Chroma')
    @patch('src.db.vector_store.OpenAIEmbeddings')
    def test_add_document(
        self, mock_embeddings_class: Mock, mock_chroma_class: Mock
    ) -> None:
        """ドキュメント追加機能のテスト。"""
        # モックの設定
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings

        mock_chroma = MagicMock()
        mock_chroma_class.return_value = mock_chroma

        # テスト対象メソッドの実行
        with patch.object(self.vector_store, 'split_text') as mock_split_text:
            # テキスト分割のモック
            chunks = ['チャンク1', 'チャンク2', 'チャンク3']
            mock_split_text.return_value = chunks

            # テスト実行
            self.vector_store.add_document(self.markdown_content, self.metadata)

        # 検証
        mock_split_text.assert_called_once_with(self.markdown_content)
        mock_chroma.add_texts.assert_called_once()

        # add_textsが呼び出されたことを確認
        mock_chroma.add_texts.assert_called_once()
        # 詳細な引数の検証は省略

    @patch('src.db.vector_store.Chroma')
    @patch('src.db.vector_store.OpenAIEmbeddings')
    def test_search(self, mock_embeddings_class: Mock, mock_chroma_class: Mock) -> None:
        """検索機能のテスト。"""
        # モックの設定
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings

        mock_chroma = MagicMock()
        mock_chroma.similarity_search_with_relevance_scores.return_value = [
            (MagicMock(page_content='検索結果1', metadata={'source': 'doc1.md'}), 0.95),
            (MagicMock(page_content='検索結果2', metadata={'source': 'doc2.md'}), 0.85),
            (MagicMock(page_content='検索結果3', metadata={'source': 'doc3.md'}), 0.75),
        ]
        mock_chroma_class.return_value = mock_chroma

        # テスト対象メソッドの実行
        results = self.vector_store.search('テスト検索クエリ', k=3)

        # 検証
        assert len(results) == 3
        assert results[0]['content'] == '検索結果1'
        assert results[0]['metadata']['source'] == 'doc1.md'
        assert results[0]['score'] == 0.95

        assert results[1]['content'] == '検索結果2'
        assert results[1]['metadata']['source'] == 'doc2.md'
        assert results[1]['score'] == 0.85

        assert results[2]['content'] == '検索結果3'
        assert results[2]['metadata']['source'] == 'doc3.md'
        assert results[2]['score'] == 0.75

        # モックの呼び出しを検証
        mock_chroma.similarity_search_with_relevance_scores.assert_called_once()
        # 詳細な引数の検証は省略

    @pytest.mark.skip(
        reason='persist()メソッドは削除されました。'
        'Chroma 0.4.x以降は自動的に永続化されます。'
    )
    @patch('src.db.vector_store.Chroma')
    @patch('src.db.vector_store.OpenAIEmbeddings')
    def test_persist(
        self, mock_embeddings_class: Mock, mock_chroma_class: Mock
    ) -> None:
        """永続化機能のテスト。"""
        # このテストはスキップされます
        pass

    @patch('src.db.vector_store.Chroma')
    @patch('src.db.vector_store.OpenAIEmbeddings')
    def test_add_document_error(
        self, mock_embeddings_class: Mock, mock_chroma_class: Mock
    ) -> None:
        """ドキュメント追加エラーのテスト。"""
        # モックの設定
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings

        mock_chroma = MagicMock()
        mock_chroma.add_texts.side_effect = Exception('DB error')
        mock_chroma_class.return_value = mock_chroma

        # テスト対象メソッドの実行と例外の検証
        with patch.object(self.vector_store, 'split_text') as mock_split_text:
            # テキスト分割のモック
            mock_split_text.return_value = ['チャンク1', 'チャンク2', 'チャンク3']

            # テスト実行と例外の検証
            with pytest.raises(VectorStoreError) as excinfo:
                self.vector_store.add_document(self.markdown_content, self.metadata)

        # 例外メッセージの検証
        assert 'ドキュメントの追加に失敗しました' in str(excinfo.value)

    @patch('src.db.vector_store.Chroma')
    @patch('src.db.vector_store.OpenAIEmbeddings')
    def test_search_error(
        self, mock_embeddings_class: Mock, mock_chroma_class: Mock
    ) -> None:
        """検索エラーのテスト。"""
        # モックの設定
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings

        mock_chroma = MagicMock()
        mock_chroma.similarity_search_with_relevance_scores.side_effect = Exception(
            'Search error'
        )
        mock_chroma_class.return_value = mock_chroma

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(VectorStoreError) as excinfo:
            self.vector_store.search('テスト検索クエリ')

        # 例外メッセージの検証
        assert '検索に失敗しました' in str(excinfo.value)
