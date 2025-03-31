"""DocumentProcessorクラスのテスト。

このモジュールは、DocumentProcessorクラスの機能をテストします。
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.core.document_processor import DocumentProcessingError, DocumentProcessor


class TestDocumentProcessor:
    """DocumentProcessorクラスのテストケース。"""

    def setup_method(self) -> None:
        """各テストメソッドの前に実行される。"""
        self.processor = DocumentProcessor()
        self.test_file_path = os.path.join(tempfile.gettempdir(), 'test_file.txt')
        self.test_output_dir = os.path.join(tempfile.gettempdir(), 'test_output')

        # テスト用のファイルを作成
        with open(self.test_file_path, 'w') as f:
            f.write('This is a test file.')

        # テスト用の出力ディレクトリを作成
        os.makedirs(self.test_output_dir, exist_ok=True)

    def teardown_method(self) -> None:
        """各テストメソッドの後に実行される。"""
        # テスト用のファイルを削除
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

        # テスト用の出力ディレクトリを削除
        if os.path.exists(self.test_output_dir):
            import shutil

            shutil.rmtree(self.test_output_dir)

    @patch('src.core.document_processor.MarkItDown')
    def test_convert_file_success(self, mock_markitdown: Mock) -> None:
        """ファイル変換が成功した場合のテスト。"""
        # モックの設定
        mock_instance = MagicMock()
        mock_instance.convert.return_value.markdown = '# Converted Markdown'
        mock_markitdown.return_value = mock_instance

        # テスト対象メソッドの実行
        result = self.processor.convert_file(self.test_file_path)

        # 検証
        assert result == '# Converted Markdown'
        mock_instance.convert.assert_called_once_with(self.test_file_path)

    @patch('src.core.document_processor.MarkItDown')
    def test_convert_file_failure(self, mock_markitdown: Mock) -> None:
        """ファイル変換が失敗した場合のテスト。"""
        # モックの設定
        mock_instance = MagicMock()
        mock_instance.convert.side_effect = Exception('Conversion failed')
        mock_markitdown.return_value = mock_instance

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(DocumentProcessingError) as excinfo:
            self.processor.convert_file(self.test_file_path)

        # 例外メッセージの検証
        assert 'ファイルの変換に失敗しました' in str(excinfo.value)

    def test_save_markdown_success(self) -> None:
        """マークダウンの保存が成功した場合のテスト。"""
        markdown_content = '# Test Markdown'
        filename = 'test_document'

        # テスト対象メソッドの実行
        output_path = self.processor.save_markdown(
            markdown_content, filename, self.test_output_dir
        )

        # 検証
        expected_path = os.path.join(self.test_output_dir, f'{filename}.md')
        assert output_path == expected_path
        assert os.path.exists(output_path)

        # ファイルの内容を検証
        with open(output_path) as f:
            content = f.read()
        assert content == markdown_content

    def test_save_markdown_failure(self) -> None:
        """マークダウンの保存が失敗した場合のテスト。"""
        markdown_content = '# Test Markdown'
        filename = 'test_document'

        # 存在しないディレクトリを指定
        non_existent_dir = os.path.join(tempfile.gettempdir(), 'non_existent_dir')

        # ディレクトリが存在しないことを確認
        if os.path.exists(non_existent_dir):
            import shutil

            shutil.rmtree(non_existent_dir)

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(DocumentProcessingError) as excinfo:
            # 書き込み権限がないディレクトリを模倣するためにパッチを使用
            with patch(
                'builtins.open', side_effect=PermissionError('Permission denied')
            ):
                self.processor.save_markdown(
                    markdown_content, filename, non_existent_dir
                )

        # 例外メッセージの検証
        assert 'マークダウンの保存に失敗しました' in str(excinfo.value)

    @patch('src.core.document_processor.DocumentProcessor.convert_file')
    @patch('src.core.document_processor.DocumentProcessor.save_markdown')
    def test_process_document_success(
        self, mock_save: Mock, mock_convert: Mock
    ) -> None:
        """ドキュメント処理が成功した場合のテスト。"""
        # モックの設定
        mock_convert.return_value = '# Converted Markdown'
        mock_save.return_value = os.path.join(self.test_output_dir, 'test_document.md')

        # テスト対象メソッドの実行
        result = self.processor.process_document(
            self.test_file_path, 'test_document', self.test_output_dir
        )

        # 検証
        assert result == os.path.join(self.test_output_dir, 'test_document.md')
        mock_convert.assert_called_once_with(self.test_file_path)
        mock_save.assert_called_once_with(
            '# Converted Markdown', 'test_document', self.test_output_dir
        )

    @patch('src.core.document_processor.DocumentProcessor.convert_file')
    def test_process_document_conversion_failure(self, mock_convert: Mock) -> None:
        """ドキュメント変換が失敗した場合のテスト。"""
        # モックの設定
        mock_convert.side_effect = DocumentProcessingError(
            'ファイルの変換に失敗しました'
        )

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(DocumentProcessingError) as excinfo:
            self.processor.process_document(
                self.test_file_path, 'test_document', self.test_output_dir
            )

        # 例外メッセージの検証
        assert 'ファイルの変換に失敗しました' in str(excinfo.value)

    @patch('src.core.document_processor.DocumentProcessor.convert_file')
    @patch('src.core.document_processor.DocumentProcessor.save_markdown')
    def test_process_document_save_failure(
        self, mock_save: Mock, mock_convert: Mock
    ) -> None:
        """マークダウンの保存が失敗した場合のテスト。"""
        # モックの設定
        mock_convert.return_value = '# Converted Markdown'
        mock_save.side_effect = DocumentProcessingError(
            'マークダウンの保存に失敗しました'
        )

        # テスト対象メソッドの実行と例外の検証
        with pytest.raises(DocumentProcessingError) as excinfo:
            self.processor.process_document(
                self.test_file_path, 'test_document', self.test_output_dir
            )

        # 例外メッセージの検証
        assert 'マークダウンの保存に失敗しました' in str(excinfo.value)

    def test_process_document_with_callback(self) -> None:
        """コールバック付きのドキュメント処理テスト。"""
        # モックのコールバック関数
        mock_callback = MagicMock()

        # テスト対象メソッドの実行
        with patch(
            'src.core.document_processor.DocumentProcessor.convert_file'
        ) as mock_convert:
            with patch(
                'src.core.document_processor.DocumentProcessor.save_markdown'
            ) as mock_save:
                # モックの設定
                mock_convert.return_value = '# Converted Markdown'
                mock_save.return_value = os.path.join(
                    self.test_output_dir, 'test_document.md'
                )

                # テスト対象メソッドの実行
                self.processor.process_document(
                    self.test_file_path,
                    'test_document',
                    self.test_output_dir,
                    progress_callback=mock_callback,
                )

        # コールバックが呼び出されたことを検証
        assert mock_callback.call_count == 3
        # 開始時のコール
        mock_callback.assert_any_call(0.0, '処理を開始しています...')
        # 変換後のコール
        mock_callback.assert_any_call(0.5, 'ファイルを変換しました')
        # 完了時のコール
        mock_callback.assert_any_call(1.0, '処理が完了しました')
