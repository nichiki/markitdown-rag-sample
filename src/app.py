"""Streamlit UI for RAG application.

このモジュールは、RAG機能を備えたStreamlit UIを提供します。
ドキュメントのアップロード、一覧表示、検索機能を実装しています。
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.document_processor import DocumentProcessor  # type: ignore
from core.rag import RAG  # type: ignore
from db.vector_store import VectorStore  # type: ignore

# .envファイルから環境変数を読み込む
load_dotenv()

# 定数
UPLOAD_DIR = './data/uploads'
PROCESSED_DIR = './data/processed'
EMBEDDINGS_DIR = './data/embeddings'
SUPPORTED_TYPES = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/plain',
    'text/markdown',
    'text/html',
    'text/csv',
    'application/json',
    'application/xml',
    'image/jpeg',
    'image/png',
]
SUPPORTED_EXTENSIONS = [
    '.pdf',
    '.docx',
    '.pptx',
    '.xlsx',
    '.txt',
    '.md',
    '.html',
    '.htm',
    '.csv',
    '.json',
    '.xml',
    '.jpg',
    '.jpeg',
    '.png',
]


def setup_directories() -> None:
    """必要なディレクトリを作成します。"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def get_uploaded_files() -> list[Path]:
    """アップロードされたファイルの一覧を取得します。"""
    return sorted(Path(UPLOAD_DIR).glob('*'))


def get_processed_files() -> list[Path]:
    """処理済みファイルの一覧を取得します。"""
    return sorted(Path(PROCESSED_DIR).glob('*.md'))


def init_session_state() -> None:
    """セッション状態を初期化します。"""
    if 'processing_file' not in st.session_state:
        st.session_state.processing_file = False
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'progress' not in st.session_state:
        st.session_state.progress = 0.0
    if 'status' not in st.session_state:
        st.session_state.status = ''
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore(persist_directory=EMBEDDINGS_DIR)
    if 'rag' not in st.session_state:
        st.session_state.rag = RAG(
            vector_store=st.session_state.vector_store,
            embeddings_dir=EMBEDDINGS_DIR,
        )


def upload_file() -> None:
    """ファイルアップロード機能を提供します。"""
    st.subheader('ドキュメントのアップロード')

    uploaded_file = st.file_uploader(
        'ファイルを選択してください',
        type=[
            'pdf',
            'docx',
            'pptx',
            'xlsx',
            'txt',
            'md',
            'html',
            'htm',
            'csv',
            'json',
            'xml',
            'jpg',
            'jpeg',
            'png',
        ],
        help=(
            'PDF, Word, PowerPoint, Excel, テキスト, Markdown, HTML, CSV, JSON, XML, '
            '画像ファイルがサポートされています'
        ),
    )

    if uploaded_file is not None:
        # ファイル名を取得
        file_name = uploaded_file.name
        file_path = os.path.join(UPLOAD_DIR, file_name)

        # ファイルを保存
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"ファイル '{file_name}' がアップロードされました。")

        # 処理ボタン
        if st.button('ファイルを処理'):
            st.session_state.processing_file = True
            st.session_state.current_file = file_path
            st.rerun()


def process_file(file_path: str) -> str | None:
    """ファイルを処理します。

    Args:
        file_path: 処理するファイルのパス

    Returns:
        Optional[str]: 処理されたファイルのパス、エラー時はNone
    """
    try:
        # 進捗コールバック関数
        def progress_callback(progress: float, status: str) -> None:
            st.session_state.progress = progress
            st.session_state.status = status

        # ファイル名を取得
        file_name = os.path.basename(file_path)
        output_name = Path(file_name).stem

        # ドキュメント処理を実行
        processor = DocumentProcessor()
        output_path: str = processor.process_document(
            file_path, output_name, PROCESSED_DIR, progress_callback
        )

        # ベクトルストアに追加
        with open(output_path, encoding='utf-8') as f:
            content = f.read()

        st.session_state.vector_store.add_document(
            markdown_content=content,
            metadata={'source': output_name + '.md'},
        )

        return output_path
    except Exception as e:
        st.error(f'ファイル処理中にエラーが発生しました: {e}')
        return None


def display_file_list() -> None:
    """処理済みファイルの一覧を表示します。"""
    st.subheader('処理済みドキュメント')

    processed_files = get_processed_files()
    if not processed_files:
        st.info('処理済みのドキュメントはありません。')
        return

    for file_path in processed_files:
        file_name = file_path.name
        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f'📄 {file_name}')

        with col2:
            if st.button('表示', key=f'view_{file_name}'):
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                st.markdown('---')
                st.markdown(f'### {file_name}')
                st.markdown(content)
                st.markdown('---')


def search_interface() -> None:
    """検索インターフェースを提供します。"""
    st.subheader('ドキュメント検索')

    query = st.text_input('検索クエリを入力してください')

    col1, col2 = st.columns([1, 3])
    with col1:
        k = st.number_input('結果数', min_value=1, max_value=10, value=3)

    if st.button('検索', key='search_button'):
        if not query:
            st.warning('検索クエリを入力してください。')
            return

        with st.spinner('検索中...'):
            try:
                # RAGを実行
                response = st.session_state.rag.query(query, k=k)

                # 結果を表示
                st.markdown('### 回答')
                st.markdown(response.answer)

                # 情報源を表示
                st.markdown('### 情報源')
                for i, source in enumerate(response.sources):
                    with st.expander(f'情報源 {i + 1} (スコア: {source.score:.4f})'):
                        st.write(f'ソース: {source.metadata.get("source", "Unknown")}')
                        st.write('内容:')
                        st.markdown(
                            source.content[:500] + '...'
                            if len(source.content) > 500
                            else source.content
                        )
            except Exception as e:
                st.error(f'検索中にエラーが発生しました: {e}')


def main() -> None:
    """メイン関数。"""
    # ディレクトリの設定
    setup_directories()

    # セッション状態の初期化
    init_session_state()

    # アプリケーションのタイトル
    st.title('RAG検証アプリケーション')
    st.markdown(
        """
        このアプリケーションは、ドキュメントをアップロードし、マークダウンに変換して、
        検索可能なベクトルデータベースに保存します。
        その後、自然言語でドキュメントの内容を検索できます。
        """
    )

    # タブの設定
    tab1, tab2, tab3 = st.tabs(['アップロード', 'ドキュメント一覧', '検索'])

    with tab1:
        upload_file()

        # ファイル処理中の場合
        if st.session_state.processing_file and st.session_state.current_file:
            st.markdown('---')
            st.subheader('ファイル処理中')

            # プログレスバーの表示
            progress_bar = st.progress(0)
            status_text = st.empty()

            # ファイル処理の実行
            output_path = process_file(st.session_state.current_file)

            # プログレスバーの更新
            progress_bar.progress(st.session_state.progress)
            status_text.text(st.session_state.status)

            if output_path:
                st.success(
                    f'ファイルの処理が完了しました: {os.path.basename(output_path)}'
                )

            # 処理完了後にフラグをリセット
            st.session_state.processing_file = False
            st.session_state.current_file = None

    with tab2:
        display_file_list()

    with tab3:
        search_interface()


if __name__ == '__main__':
    main()
