"""ドキュメント処理モジュール。

このモジュールは、様々な形式のドキュメントをマークダウンに変換し、
保存するための機能を提供します。
"""

import os
from collections.abc import Callable

from markitdown import MarkItDown


class DocumentProcessingError(Exception):
    """ドキュメント処理中に発生するエラー。"""

    pass


class DocumentProcessor:
    """ドキュメント処理クラス。

    このクラスは、markitdownライブラリを使用して様々な形式のドキュメントを
    マークダウンに変換し、保存する機能を提供します。
    """

    def __init__(self) -> None:
        """DocumentProcessorのインスタンスを初期化します。"""
        # テスト時にモック可能にするために、初期化時ではなく遅延初期化する
        self._markitdown = None

    def convert_file(self, file_path: str) -> str:
        """ファイルをマークダウンに変換します。

        Args:
            file_path: 変換するファイルのパス

        Returns:
            str: 変換されたマークダウンテキスト

        Raises:
            DocumentProcessingError: ファイルの変換に失敗した場合
        """
        try:
            # 遅延初期化
            markitdown = MarkItDown()
            result = markitdown.convert(file_path)
            return str(result.markdown)
        except Exception as e:
            raise DocumentProcessingError(f'ファイルの変換に失敗しました: {e}') from e

    def save_markdown(self, markdown: str, filename: str, output_dir: str) -> str:
        """マークダウンをファイルに保存します。

        Args:
            markdown: 保存するマークダウンテキスト
            filename: 保存するファイル名(拡張子なし)
            output_dir: 保存先ディレクトリのパス

        Returns:
            str: 保存されたファイルのパス

        Raises:
            DocumentProcessingError: マークダウンの保存に失敗した場合
        """
        try:
            # 出力ディレクトリが存在しない場合は作成
            os.makedirs(output_dir, exist_ok=True)

            # ファイルパスを生成
            output_path = os.path.join(output_dir, f'{filename}.md')

            # ファイルに書き込み
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            return output_path
        except Exception as e:
            raise DocumentProcessingError(
                f'マークダウンの保存に失敗しました: {e}'
            ) from e

    def process_document(
        self,
        file_path: str,
        output_filename: str,
        output_dir: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        """ドキュメントを処理します。

        ファイルを変換し、マークダウンとして保存します。

        Args:
            file_path: 処理するファイルのパス
            output_filename: 出力ファイル名(拡張子なし)
            output_dir: 出力ディレクトリのパス
            progress_callback: 進捗状況を通知するコールバック関数(オプション)
                第1引数: 進捗率(0.0〜1.0)
                第2引数: 状態メッセージ

        Returns:
            str: 保存されたファイルのパス

        Raises:
            DocumentProcessingError: 処理中にエラーが発生した場合
        """
        # 進捗コールバックが指定されている場合は呼び出す
        if progress_callback:
            progress_callback(0.0, '処理を開始しています...')

        # ファイルを変換
        markdown = self.convert_file(file_path)

        # 進捗コールバックが指定されている場合は呼び出す
        if progress_callback:
            progress_callback(0.5, 'ファイルを変換しました')

        # マークダウンを保存
        output_path = self.save_markdown(markdown, output_filename, output_dir)

        # 進捗コールバックが指定されている場合は呼び出す
        if progress_callback:
            progress_callback(1.0, '処理が完了しました')

        return output_path


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    def main() -> int:
        """メイン関数。

        コマンドライン引数を解析し、ドキュメント処理を実行します。

        Returns:
            int: 終了コード(0: 成功、1: 失敗)
        """
        parser = argparse.ArgumentParser(
            description='ドキュメントをマークダウンに変換します。'
        )
        parser.add_argument('file_path', help='変換するファイルのパス')
        parser.add_argument(
            '-o',
            '--output-dir',
            default='./data/processed',
            help='出力ディレクトリのパス',
        )
        parser.add_argument(
            '-n',
            '--output-name',
            help='出力ファイル名(拡張子なし、指定しない場合は入力ファイル名を使用)',
        )
        args = parser.parse_args()

        try:
            # 入力ファイルのパスを取得
            file_path = args.file_path

            # 出力ファイル名を取得(指定されていない場合は入力ファイル名を使用)
            if args.output_name:
                output_name = args.output_name
            else:
                output_name = Path(file_path).stem

            # 出力ディレクトリを取得
            output_dir = args.output_dir

            # 進捗コールバック関数
            def progress_callback(progress: float, status: str) -> None:
                print(f'進捗: {progress:.0%} - {status}')

            # ドキュメント処理を実行
            processor = DocumentProcessor()
            output_path = processor.process_document(
                file_path, output_name, output_dir, progress_callback
            )

            print(f'変換完了: {output_path}')
            return 0
        except DocumentProcessingError as e:
            print(f'エラー: {e}', file=sys.stderr)
            return 1
        except Exception as e:
            print(f'予期しないエラー: {e}', file=sys.stderr)
            return 1

    sys.exit(main())
