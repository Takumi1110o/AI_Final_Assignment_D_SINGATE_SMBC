import pandas as pd
import numpy as np
import category_encoders as ce

class LeaveOneOut:
    """
    Leave-One-Out (LOO) Target Encoder クラス。
    K-Fold交差検証内で使用するために、fit/transformメソッドを持つように設計されています。
    """
    def __init__(self, cols, target_col='health'):
        """
        コンストラクタ。エンコード対象の列とターゲット列を指定します。

        Args:
            cols (list): ターゲットエンコーディングを適用するカテゴリカル列名のリスト。
            target_col (str): ターゲット列の名前。デフォルトは 'health'。
        """
        self.cols = cols
        self.target_col = target_col
        self.encoder = None # エンコーダインスタンスを保持

    def fit(self, X, y):
        """
        トレーニングデータにエンコーダを学習させます。

        Args:
            X (pd.DataFrame): 学習用の特徴量データフレーム。
            y (pd.Series): 学習用のターゲット系列。
        """
        # LOOエンコーダを初期化し、指定された列にfit_transformを適用して学習します。
        # ここでは学習のみを行い、変換結果は使用しません。
        self.encoder = ce.LeaveOneOutEncoder(cols=self.cols, random_state=42)
        # fit_transform は内部的に fit して transform するため、fit だけが必要な場合は
        # ダミーで変換結果を受け取ります。
        _ = self.encoder.fit_transform(X[self.cols], y)
        return self

    def transform(self, X):
        """
        データフレームを変換します。

        Args:
            X (pd.DataFrame): 変換する特徴量データフレーム。

        Returns:
            pd.DataFrame: ターゲットエンコードされた新しい列が追加されたデータフレーム。
        """
        X_copy = X.copy()
        # 学習済みのエンコーダを使ってデータを変換します。
        # ターゲット列は変換に不要なので、特徴量のみを渡します。
        transformed_data = self.encoder.transform(X_copy[self.cols])

        # 変換された列を元のデータフレームに追加します。
        for col_idx, original_col in enumerate(self.cols):
            cate_col = f'target_{original_col}'
            # transformed_dataはDataFrameなので、正しい列を選択して代入
            X_copy[cate_col] = transformed_data.iloc[:, col_idx]

        return X_copy


class CATBOOST:
    """
    CatBoost Target Encoder クラス。
    K-Fold交差検証内で使用するために、fit/transformメソッドを持つように設計されています。
    """
    def __init__(self, cols, target_col='health', random_state=None):
        """
        コンストラクタ。エンコード対象の列、ターゲット列、乱数シードを指定します。

        Args:
            cols (list): ターゲットエンコーディングを適用するカテゴリカル列名のリスト。
            target_col (str): ターゲット列の名前。デフォルトは 'health'。
            random_state (int, optional): 乱数シード。yml['params']['seed']から渡されます。
        """
        self.cols = cols
        self.target_col = target_col
        self.random_state = random_state
        self.encoder = None # エンコーダインスタンスを保持

    def fit(self, X, y):
        """
        トレーニングデータにエンコーダを学習させます。

        Args:
            X (pd.DataFrame): 学習用の特徴量データフレーム。
            y (pd.Series): 学習用のターゲット系列。
        """
        self.encoder = ce.CatBoostEncoder(cols=self.cols, random_state=self.random_state)
        # fit_transform でエンコーダを学習
        _ = self.encoder.fit_transform(X[self.cols], y)
        return self

    def transform(self, X):
        """
        データフレームを変換します。

        Args:
            X (pd.DataFrame): 変換する特徴量データフレーム。

        Returns:
            pd.DataFrame: ターゲットエンコードされた新しい列が追加されたデータフレーム。
        """
        X_copy = X.copy()
        transformed_data = self.encoder.transform(X_copy[self.cols])

        for col_idx, original_col in enumerate(self.cols):
            cate_col = f'target_{original_col}'
            X_copy[cate_col] = transformed_data.iloc[:, col_idx]

        return X_copy


class JAMESSTEIN:
    """
    James-Stein Target Encoder クラス。
    K-Fold交差検証内で使用するために、fit/transformメソッドを持つように設計されています。
    """
    def __init__(self, cols, target_col='health', random_state=None):
        """
        コンストラクタ。エンコード対象の列、ターゲット列、乱数シードを指定します。

        Args:
            cols (list): ターゲットエンコーディングを適用するカテゴリカル列名のリスト。
            target_col (str): ターゲット列の名前。デフォルトは 'health'。
            random_state (int, optional): 乱数シード。yml['params']['seed']から渡されます。
        """
        self.cols = cols
        self.target_col = target_col
        self.random_state = random_state
        self.encoder = None # エンコーダインスタンスを保持

    def fit(self, X, y):
        """
        トレーニングデータにエンコーダを学習させます。

        Args:
            X (pd.DataFrame): 学習用の特徴量データフレーム。
            y (pd.Series): 学習用のターゲット系列。
        """
        self.encoder = ce.JamesSteinEncoder(cols=self.cols, random_state=self.random_state)
        # fit_transform でエンコーダを学習
        _ = self.encoder.fit_transform(X[self.cols], y)
        return self

    def transform(self, X):
        """
        データフレームを変換します。

        Args:
            X (pd.DataFrame): 変換する特徴量データフレーム。

        Returns:
            pd.DataFrame: ターゲットエンコードされた新しい列が追加されたデータフレーム。
        """
        X_copy = X.copy()
        transformed_data = self.encoder.transform(X_copy[self.cols])

        for col_idx, original_col in enumerate(self.cols):
            cate_col = f'target_{original_col}'
            X_copy[cate_col] = transformed_data.iloc[:, col_idx]

        return X_copy
