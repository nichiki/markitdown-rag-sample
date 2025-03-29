# Python コーディングスタイル

このプロジェクトでは、以下のコーディングスタイルルールに従ってください。

## 1. ドキュメント

### ファイルレベルのドキュメント
- すべての Python ファイルは、ファイルの先頭に docstring を含める必要があります
- docstring には、ファイルの目的と内容の概要を記述してください

例:
```python
"""
ユーザー認証モジュール。

このモジュールは、ユーザーの認証と権限管理に関する機能を提供します。
"""
```

### 関数とメソッドのドキュメント
- すべての関数とメソッドには docstring を含める必要があります
- docstring には、関数の目的、引数、戻り値、例外を記述してください
- Google スタイルの docstring 形式を使用してください

例:
```python
def authenticate_user(username: str, password: str) -> bool:
    """
    ユーザー名とパスワードを検証します。

    Args:
        username: 認証するユーザー名
        password: ユーザーのパスワード

    Returns:
        認証が成功した場合は True、それ以外の場合は False

    Raises:
        ValueError: ユーザー名または パスワードが空の場合
    """
```

### クラスのドキュメント
- すべてのクラスには docstring を含める必要があります
- docstring には、クラスの目的、属性、メソッドの概要を記述してください

## 2. 型アノテーション

- すべての関数とメソッドの引数と戻り値には型アノテーションを含める必要があります
- 変数の型が明確でない場合は、変数にも型アノテーションを含めてください
- 複合型には `typing` モジュールを使用してください

例:
```python
from typing import Dict, List, Optional

def get_user_data(user_id: int) -> Optional[Dict[str, any]]:
    """ユーザーデータを取得します。"""
    ...
```

## 3. インポート

- インポートは以下の順序でグループ化してください:
  1. 標準ライブラリ
  2. サードパーティライブラリ
  3. ローカルアプリケーション/ライブラリ
- 各グループ内では、アルファベット順に並べてください
- `import` と `from ... import` は別々のブロックにしてください

例:
```python
# 標準ライブラリ
import os
import sys
from datetime import datetime

# サードパーティライブラリ
import numpy as np
import pandas as pd

# ローカルアプリケーション
from myapp.models import User
from myapp.utils import format_date
```

## 4. 命名規則

- クラス名: UpperCamelCase
- 関数、メソッド、変数名: snake_case
- 定数: UPPER_SNAKE_CASE
- プライベート属性とメソッド: _leading_underscore
- 内部使用の属性とメソッド: __double_leading_underscore

## 5. コードフォーマット

- インデントには 4 スペースを使用してください
- 行の最大長は 88 文字です
- 関数やクラスの定義の間には 2 行の空行を入れてください
- メソッドの定義の間には 1 行の空行を入れてください
- リスト、辞書、タプルの要素間にはスペースを入れてください

例:
```python
# 良い例
my_list = [1, 2, 3, 4]
my_dict = {"key1": "value1", "key2": "value2"}

# 悪い例
my_list = [1,2,3,4]
my_dict = {"key1":"value1","key2":"value2"}
```

## 6. コメント

- コードの「なぜ」を説明するコメントを書いてください
- 複雑なロジックや非自明な実装には必ずコメントを付けてください
- TODO コメントには担当者と課題番号を含めてください

例:
```python
# TODO(username): このアルゴリズムをより効率的なものに置き換える (#123)
```

## 7. エラー処理

- 例外は具体的な型で捕捉してください（`except Exception:` は避ける）
- エラーメッセージは明確で具体的にしてください
- リソースの確保と解放には `with` ステートメントを使用してください

## 8. テスト

- すべての公開関数とメソッドにはユニットテストを書いてください
- テストファイル名は `test_` で始めてください
- テスト関数名は `test_` で始めてください

## 9. Ruff の設定

このプロジェクトでは、Ruff を使用してコードのリンティングとフォーマットを行います。以下の設定が適用されます：

```toml
[tool.ruff]
line-length = 88
target-version = "py312"
select = ["E", "F", "I", "N", "W", "B", "A", "C4", "UP", "ANN", "RUF"]
ignore = ["ANN101"]  # self の型アノテーションは不要

[tool.ruff.format]
quote-style = "single"
indent-style = "space"
line-ending = "auto"
```

## 10. mypy の設定

このプロジェクトでは、mypy を使用して静的型チェックを行います。以下の設定が適用されます：

```toml
[tool.mypy]
python_version = "3.12"
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
strict_optional = true
warn_redundant_casts = true
warn_return_any = true
warn_unused_ignores = true
```

## 11. 関数型アプローチ

可能な限り、以下の原則に従ってください：

- 純粋関数を優先する（同じ入力に対して常に同じ出力を返し、副作用がない）
- 不変データ構造を使用する
- 副作用を分離する
- 型安全性を確保する

## 12. オブジェクト指向プログラミング

クラスを使用する場合は、以下の原則に従ってください：

- 単一責任の原則を守る
- インターフェースを明確に定義する
- 継承よりもコンポジションを優先する
- データと振る舞いをカプセル化する
