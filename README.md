# Python Boilerplate

Python プロジェクトのボイラープレート（テンプレート）です。devcontainer を使用した開発環境の標準化と、コード品質ツール、AI開発ツールの設定が含まれています。このテンプレートを使用して、高品質な Python プロジェクトを素早く開始できます。

## 特徴

- **devcontainer による開発環境の標準化**
  - Python 3.12
  - uv パッケージマネージャー（高速な依存関係管理）
  - Chromium ブラウザと日本語フォントの事前インストール

- **コード品質ツール**
  - Ruff によるリンティングとフォーマット（保存時に自動適用）
  - mypy による厳格な静的型チェック
  - pytest と pytest-cov によるテストとカバレッジ測定

- **AI開発ツール**
  - Cline 用のルール定義（.clinerules/）
  - browser_action 対応（Web アプリケーションのテスト用）
  - clinerules-bank による言語・フレームワーク固有のガイドライン

- **コーディング規約**
  - Google スタイルの docstring
  - 厳格な型アノテーション
  - 関数型アプローチの推奨
  - 単一責任の原則などのベストプラクティス

## セットアップ

このリポジトリは VS Code の devcontainer 機能を使用しています。VS Code と Docker がインストールされていれば、自動的に開発環境がセットアップされます。

1. リポジトリをクローンまたはテンプレートとして使用
   ```bash
   git clone https://github.com/yourusername/python-boilerplate.git
   cd python-boilerplate
   ```

2. VS Code で開く
   ```bash
   code .
   ```

3. VS Code が devcontainer を検出し、「Reopen in Container」を提案するので、それをクリックします。
   - または、コマンドパレット（F1）から「Remote-Containers: Reopen in Container」を選択

4. 開発用依存関係をインストール
   ```bash
   uv sync --dev
   ```

## 使用方法

### 新しいプロジェクトの作成

1. このリポジトリをテンプレートとして使用
2. プロジェクト名や説明を変更（pyproject.toml）
3. 必要な依存関係を追加（初期状態では空になっています）
   ```bash
   uv add package_name
   ```

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

このプロジェクトでは、`clinerules-bank/languages/python/` に定義された詳細なコーディング規約に従います：

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

## ディレクトリ構造

```
python-boilerplate/
├── .clinerules/           # Cline のルール定義
│   ├── 00-core-principles.md
│   ├── 01-development-workflow.md
│   └── ...
├── .devcontainer/         # devcontainer の設定
│   ├── Dockerfile
│   └── devcontainer.json
├── clinerules-bank/       # 言語・フレームワーク固有のガイドライン
│   ├── languages/
│   │   └── python/
│   │       ├── coding-style.md
│   │       └── testing-guidelines.md
│   └── ...
├── src/                   # メインのソースコード
├── tests/                 # テストコード
├── .clineignore           # Cline の除外設定
├── .env.example           # 環境変数の例
├── .gitignore             # Git の除外設定
├── .python-version        # Python バージョン指定（3.12）
├── README.md              # このファイル
└── pyproject.toml         # プロジェクト設定
