# 大学の講義課題用に公開したコード
※見やすいようにGeminiで修正

# 仮想環境
> TODO: 時間があればuvにする

仮想環境作成(初回)
```
python3 -m venv .venv
```
仮想環境切り替え
```
. .venv/bin/activate
```
pipをアップデート
```
python -m pip install --upgrade pip
```
パッケージインストール(初回)
```
python -m pip install -r requirements.txt
```
仮想環境終了
```
deactivate
```

# 実行方法
1. `exp/data`に以下のデータを入れる
    ```
    sample_submission.csv
    test.csv
    train.csv
    ```

2. `exp/exp19/` に移動し、以下を実行
    ```
    (.venv) python run.py
    ```
    `exp/exp19/exp19_YYYY-MM-DD_TT-TT-TT` に結果などが保存される
