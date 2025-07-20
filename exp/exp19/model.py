import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score


def lgb_f1_score(preds, data):
    """
    LightGBMのカスタム評価指標としてのF1スコア（マクロ平均）。
    複数クラス分類問題に対応。

    Args:
        preds (np.array): LightGBMが返す予測確率。
        data (lightgbm.Dataset): 評価セットのデータ。

    Returns:
        tuple: (評価指標名, スコア値, is_higher_better(Trueなら高い方が良い))
    """
    y_true = data.get_label() # 正解ラベルを取得

    # 予測確率から最も確信度の高いクラスを選択
    y_pred = np.argmax(preds, axis=1)

    # マクロ平均F1スコアを計算
    score = f1_score(y_true, y_pred, average='macro')
    return 'custom', score, True # カスタム指標名とスコア、高い方が良いことを返す


def LIGHTGBM(X_train, y_train, X_valid, y_valid, X_test, yml):
    """
    LightGBMモデルを学習し、予測を行う。
    この関数は単一のモデル学習と予測を行う（K-Foldは外部で管理される）。

    Args:
        X_train (pd.DataFrame): 学習用の特徴量。
        y_train (pd.Series): 学習用のターゲット変数。
        X_valid (pd.DataFrame): バリデーション用の特徴量。
        y_valid (pd.Series): バリデーション用のターゲット変数。
        X_test (pd.DataFrame): テストデータの特徴量（最終予測用）。
        yml (dict): 設定パラメータを含む辞書。

    Returns:
        tuple:
            - test_preds (np.array): テストデータに対する予測確率。
            - oof_preds (np.array): バリデーションデータに対するOOF予測確率。
            - score (float): バリデーションセットでの最終スコア。
            - logs (list): 学習履歴を含むリスト。
            - model (lightgbm.Booster): 学習済みLightGBMモデル。
    """
    logs = []

    # LightGBMのDataset形式に変換
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    history = {} # 学習履歴を記録するための辞書
    model = lgb.train(
            params = yml["params"], # LightGBMの学習パラメータ
            train_set = lgb_train, # 学習データセット
            valid_sets = [lgb_train, lgb_valid], # 評価データセット（学習中とバリデーション）
            feval=lgb_f1_score, # カスタム評価指標
            callbacks = [
                lgb.callback.record_evaluation(history), # 評価履歴を記録
                lgb.early_stopping(
                    stopping_rounds=yml["train_params"]["early_stopping_rounds"], # 早期打ち切りラウンド数
                    verbose=True), # 早期打ち切りのログを出力
                lgb.log_evaluation(yml["train_params"]["verbose_eval"]) # 評価ログの出力頻度
            ]
            )

    # バリデーションデータに対するOOF予測
    oof_preds = model.predict(X_valid)
    # テストデータに対する予測
    test_preds = model.predict(X_test)

    # バリデーションセットでの最終スコアを取得
    score = history["valid_1"][yml['params']['metric']][-1]
    logs.append(history) # ログをリストに追加

    print(f"Fold Score: {score}")

    return test_preds, oof_preds, score, logs, model # 学習済みモデルも返す
