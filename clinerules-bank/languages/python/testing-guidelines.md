# テスト要件

このプロジェクトでは、以下のテスト要件に従ってください。

## 1. テスト構造

- すべてのテストは `tests` ディレクトリに配置してください
- テストファイル名は `test_` で始めてください
- テスト関数名は `test_` で始めてください
- テストは pytest フレームワークを使用してください

## 2. テストカバレッジ

- すべての公開関数とメソッドにはユニットテストを書いてください
- コードカバレッジは 80% 以上を目標としてください
- 重要なビジネスロジックは 100% のカバレッジを目指してください

## 3. テストの種類

### ユニットテスト
- 個々の関数やメソッドの動作を検証します
- 外部依存関係はモックしてください
- 各テストは独立して実行できるようにしてください

例:
```python
def test_add_numbers():
    """add_numbers 関数のテスト。"""
    # 準備
    a = 1
    b = 2
    expected = 3
    
    # 実行
    result = add_numbers(a, b)
    
    # 検証
    assert result == expected
```

### 統合テスト
- 複数のコンポーネントの連携を検証します
- 実際の依存関係を使用することがあります
- `tests/integration` ディレクトリに配置してください

### エンドツーエンドテスト
- システム全体の動作を検証します
- `tests/e2e` ディレクトリに配置してください

## 4. テスト環境

- テスト用の環境変数は `.env.test` ファイルに定義してください
- テスト用のデータベースは本番環境とは別のものを使用してください
- テストフィクスチャは `tests/fixtures` ディレクトリに配置してください

## 5. モックとスタブ

- 外部サービスへの依存はモックしてください
- データベースへのアクセスはテスト用のデータベースを使用するか、モックしてください
- 時間に依存するテストは、時間をモックしてください

例:
```python
@pytest.fixture
def mock_database():
    """データベース接続のモック。"""
    with patch("myapp.database.connect") as mock:
        yield mock

def test_get_user(mock_database):
    """get_user 関数のテスト。"""
    # モックの設定
    mock_database.return_value.query.return_value = {"id": 1, "name": "Test User"}
    
    # 実行
    result = get_user(1)
    
    # 検証
    assert result["name"] == "Test User"
```

## 6. パラメータ化テスト

- 複数のケースをテストする場合は、パラメータ化テストを使用してください

例:
```python
@pytest.mark.parametrize("input_value,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input_value, expected):
    """square 関数のパラメータ化テスト。"""
    assert square(input_value) == expected
```

## 7. テスト実行

- テストは CI/CD パイプラインで自動的に実行されるようにしてください
- ローカルでは以下のコマンドでテストを実行できます:

```bash
# すべてのテストを実行
pytest

# カバレッジレポートを生成
pytest --cov=src

# 特定のテストファイルを実行
pytest tests/test_specific.py

# 特定のテスト関数を実行
pytest tests/test_specific.py::test_function
```

## 8. テスト駆動開発 (TDD)

可能な限り、テスト駆動開発のアプローチを採用してください:

1. **Red**: まず失敗するテストを書く
2. **Green**: テストが通るように最小限の実装をする
3. **Refactor**: コードをリファクタリングして改善する

## 9. テストの命名

テスト名は以下の形式で記述してください:

- `test_<テスト対象の関数名>_<テストする状況>_<期待される結果>`

例:
```python
def test_validate_email_with_valid_input_returns_true():
    """有効なメールアドレスの場合に True を返すことをテスト。"""
    assert validate_email("user@example.com") is True

def test_validate_email_with_invalid_input_returns_false():
    """無効なメールアドレスの場合に False を返すことをテスト。"""
    assert validate_email("invalid-email") is False
```

## 10. アサーションのベストプラクティス

- 具体的なアサーションメッセージを提供してください
- 複数の条件を検証する場合は、個別のアサーションを使用してください
- 複雑なオブジェクトの比較には `assert_that` や `pytest.approx` などのヘルパーを使用してください

例:
```python
def test_create_user():
    """ユーザー作成のテスト。"""
    user = create_user("test_user", "test@example.com")
    
    # 個別のアサーション
    assert user.username == "test_user", "ユーザー名が正しくありません"
    assert user.email == "test@example.com", "メールアドレスが正しくありません"
    assert user.created_at is not None, "作成日時が設定されていません"
