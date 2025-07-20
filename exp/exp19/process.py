import datetime
import re
import os

import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

from fill import Fill # Assuming these are utility classes
from shape import ChangeTheShape # Assuming these are utility classes


def process(yml):
    """
    生データの前処理、結合、基本的な特徴量エンジニアリングを行う。
    ターゲットエンコーディングやモデル固有の特徴量エンジニアリングは含まない。

    Args:
        yml (dict): 設定パラメータを含む辞書。

    Returns:
        tuple: 前処理されたトレーニングデータ (pd.DataFrame) とテストデータ (pd.DataFrame)。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    file_path = os.path.join(parent_dir, 'data')
    train = pd.read_csv(file_path + '/train.csv', index_col=0)
    test = pd.read_csv(file_path + '/test.csv', index_col=0)

    # テストデータに存在しないトレーニングデータの特定の行を削除
    train = train[train['spc_common']!='Himalayan cedar'] #1つ
    train = train[train['spc_common']!='Chinese chestnut'] # 3つ
    train = train[train['nta']!='MN20'] # 1つ
    train = train[train['nta']!='MN21'] # 5つ
    train = train[train['nta']!='BK27'] # 2つ
    train_co_list = train['boro_ct'].unique()
    test_co_list = test['boro_ct'].unique()
    for co in train_co_list:
        if co not in test_co_list:
            train = train[train['boro_ct']!=co]
    train.reset_index(drop=True, inplace=True)

    # 訓練データとテストデータを結合し、一括で前処理
    len_train = len(train)
    data = pd.concat([train, test], ignore_index=True)

    # 欠損値補完
    data['steward'].fillna(0, inplace=True)
    data['problems'].fillna('0', inplace=True)

    shape = ChangeTheShape(data.copy()) # ヘルパークラスのインスタンス化
    encoder = LabelEncoder()

    # 特徴量エンジニアリング
    data['created_at'] = data['created_at'].apply(lambda x: shape.change_created_at(x)) # 日数データに変換
    data.replace({'curb_loc': {'OnCurb': 1, 'OffsetFromCurb': 0}}, inplace=True) # 1,0に変換
    data.replace({'steward': {'1or2': 1, '3or4': 2, '4orMore': 3}}, inplace=True) # 順序付け変換
    data['guards'] = encoder.fit_transform(data['guards']) # ラベルエンコーディング
    data.replace({'sidewalk': {'Damage': 1, 'NoDamage': 0}}, inplace=True) # 0,1に変換
    data['user_type'] = encoder.fit_transform(data['user_type']) # ラベルエンコーディング
    data['problems'] = data['problems'].apply(lambda x: shape.change_problems(x)) # 問題数でエンコード

    # ターゲットエンコーディングの対象となるカテゴリカル列（'spc_common', 'nta'）はそのまま残す
    # 後続の `run.py` 内のK-Foldループでターゲットエンコーディングが適用される

    # 結合したデータを再度、訓練データとテストデータに分割
    train_processed = data.iloc[:len_train].copy()
    test_processed = data.iloc[len_train:].copy()
    if yml['target_col'] in test_processed.columns:
        test_processed = test_processed.drop(columns=[yml['target_col']])

    return train_processed, test_processed


def under_sampling(train_df, zero_num, one_num, two_num):
    """
    指定された戦略に基づいてアンダーサンプリングを実行する。

    Args:
        train_df (pd.DataFrame): アンダーサンプリングを行うデータフレーム。ターゲット列を含む。
        zero_num (int): クラス0の目標サンプル数。
        one_num (int): クラス1の目標サンプル数。
        two_num (int): クラス2の目標サンプル数。

    Returns:
        tuple: アンダーサンプリング後のデータフレーム (pd.DataFrame) とリサンプリングされたターゲット配列 (np.array)。
    """
    # アンダーサンプリング戦略を辞書で定義
    strategy = {0:zero_num, 1:one_num, 2:two_num}
    rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)

    # fit_resample は特徴量とターゲットを分離して渡す必要がある
    # train_df からターゲット列を分離し、残りを特徴量として使用
    data_resampled, y_resampled = rus.fit_resample(train_df.drop('health', axis=1), train_df['health'])

    # アンダーサンプリングされた特徴量とターゲットを結合して新しいデータフレームを生成
    data_resampled = pd.DataFrame(data_resampled, columns=train_df.drop('health', axis=1).columns)
    data_resampled['health'] = y_resampled # ターゲット列を再追加

    return data_resampled, y_resampled


def submit(pred, score:str, logs, yml) -> str:
    """
    提出ファイルとログ、パラメータを保存する。

    Args:
        pred (np.array): テストデータに対する最終予測クラス。
        score (str): スコア文字列（ファイル名に含める）。
        logs (dict): ログ情報（通常はスコアや履歴）。
        yml (dict): 設定パラメータを含む辞書。

    Returns:
        str: 保存先のディレクトリパス。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    file_path = os.path.join(parent_dir, 'data')
    submission = pd.read_csv(file_path + '/sample_submission.csv', header=None)
    submission.drop(1, axis=1, inplace=True) # 2列目を削除
    submission["pred"] = pred # 予測値を新しい列として追加

    now = datetime.datetime.now()
    # 保存ディレクトリの名前を生成
    save_path = f'{yml["name"]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}'

    # ディレクトリが存在しない場合は作成
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # 提出ファイルをCSV形式で保存
    submission.to_csv(save_path + f'/{yml["name"]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False, header=False)

    # 使用したパラメータをYAML形式で保存
    with open(save_path + '/params.yaml', 'w') as file:
        yaml.dump(yml, file)

    # ログ情報をテキストファイルに保存
    f = open(save_path + f'/{score}_' + 'logging.txt', 'w')
    f.write(str(logs))
    f.close()

    return save_path

def fair_process(df):
    """
    Fairモデルに特化した特徴量エンジニアリングを行う。

    Args:
        df (pd.DataFrame): 特徴量エンジニアリングを適用するデータフレーム。

    Returns:
        pd.DataFrame: 特徴量エンジニアリング後のデータフレーム。
    """
    df_copy = df.copy() # オリジナルデータフレームを変更しないようにコピー

    # 新しい特徴量の追加
    df_copy['created_at*st_assem'] = df_copy['created_at'] * df_copy['st_assem']
    df_copy['st_assem*cncldist'] = df_copy['st_assem'] * df_copy['cncldist']
    df_copy['problems/cb_num'] = df_copy['problems'] / df_copy['cb_num']
    df_copy['cb_num/cncldist'] = df_copy['cb_num'] / df_copy['cncldist']
    df_copy['st_assem/st_senate'] = df_copy['st_assem'] / df_copy['st_senate']

    df_copy['tree_dbh*boro_ct'] = df_copy['tree_dbh'] * df_copy['boro_ct']
    df_copy['tree_dbh/cb_num'] = df_copy['tree_dbh'] / df_copy['cb_num']
    df_copy['boro_ct/steward'] = df_copy['boro_ct'] / df_copy['steward']

    # object型の列を削除
    drop_cols = ['spc_common', 'spc_latin', 'nta', 'nta_name', 'boroname', 'zip_city']
    for col in drop_cols:
        if col in df_copy.columns:
            df_copy = df_copy.drop(col, axis=1)

    return df_copy

def poor_process(df):
    """
    Poorモデルに特化した特徴量エンジニアリングを行う。

    Args:
        df (pd.DataFrame): 特徴量エンジニアリングを適用するデータフレーム。

    Returns:
        pd.DataFrame: 特徴量エンジニアリング後のデータフレーム。
    """
    df_copy = df.copy() # オリジナルデータフレームを変更しないようにコピー

    # 新しい特徴量の追加
    df_copy['created_at*st_assem'] = df_copy['created_at'] * df_copy['st_assem']
    df_copy['tree_dbh*boro_ct'] = df_copy['tree_dbh'] * df_copy['boro_ct']
    df_copy['cb_num*cncldist'] = df_copy['cb_num'] / df_copy['cncldist'] # 元のコードでは cb_num*cncldist でしたが、ここでは cb_num / cncldist に修正
    df_copy['created_at/borocode'] = df_copy['created_at'] / df_copy['borocode']
    df_copy['created_at/cb_num'] = df_copy['created_at'] / df_copy['cb_num']
    df_copy['tree_dbh/created_at'] = df_copy['tree_dbh'] / df_copy['created_at']
    df_copy['tree_dbh/borocode'] = df_copy['tree_dbh'] / df_copy['borocode']
    df_copy['tree_dbh/st_senate'] = df_copy['tree_dbh'] / df_copy['st_senate']
    df_copy['borocode/boro_ct'] = df_copy['borocode'] / df_copy['boro_ct']
    df_copy['boro_ct/tree_dbh'] = df_copy['boro_ct'] / df_copy['tree_dbh']
    df_copy['boro_ct/cb_num'] = df_copy['boro_ct'] / df_copy['cb_num']
    df_copy['cb_num/created_at'] = df_copy['cb_num'] / df_copy['created_at']
    df_copy['cb_num/boro_ct'] = df_copy['cb_num'] / df_copy['boro_ct']

    df_copy['tree_dbh/cb_num'] = df_copy['tree_dbh'] / df_copy['cb_num']
    df_copy['boro_ct/steward'] = df_copy['boro_ct'] / df_copy['steward']

    # object型の列を削除
    drop_cols = ['spc_common', 'spc_latin', 'nta', 'nta_name', 'boroname', 'zip_city']
    for col in drop_cols:
        if col in df_copy.columns:
            df_copy = df_copy.drop(col, axis=1)

    return df_copy
