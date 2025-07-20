import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

from process import process, under_sampling, submit, fair_process, poor_process
from model import LIGHTGBM
from visualize import visualize_importance, visualize_oof_gt, visualize_oof_pred
from target_encording import LeaveOneOut, CATBOOST, JAMESSTEIN # ターゲットエンコーディングモジュールをインポート


# --- グローバル変数の初期化 ---
# 各フォールドからのテストデータに対する予測値を格納するリスト
all_test_preds_us_fair = []
all_test_preds_us_poor = []
# 各フォールドのスコアを格納するリスト
all_fold_scores_us_fair = []
all_fold_scores_us_poor = []

# OOF (Out-Of-Fold) 予測値を格納する配列（トレーニングデータ全体の長さで初期化）
# 各要素には、そのサンプルがバリデーションセットになった際の予測値が入る
oof_preds_us_fair = None
oof_preds_us_poor = None

# 特徴量重要度可視化のために、各フォールドで学習したモデルを格納するリスト
all_models_us_fair_for_importance = []
all_models_us_poor_for_importance = []

# --- パラメータの読み込み ---
with open(R'params.yaml') as file:
    yml = yaml.safe_load(file)

# --- ステップ1: 初期データ処理（ターゲットエンコーディング以外の共通前処理） ---
# train_df_initial: ターゲット列を含む前処理済みトレーニングデータ
# test_df_initial: ターゲット列を含まない前処理済みテストデータ
train_df_initial, test_df_initial = process(yml)

# OOF予測値配列の初期化 (train_df_initialのサイズに合わせて)
oof_preds_us_fair = np.zeros((len(train_df_initial), yml['params']['num_class']), dtype=np.float64)
oof_preds_us_poor = np.zeros((len(train_df_initial), yml['params']['num_class']), dtype=np.float64)


# --- K分割交差検証の準備 ---
# StratifiedKFold を使用して、ターゲット変数の分布を各フォールドで均等に保つ
kf = StratifiedKFold(n_splits=yml["n_splits"], shuffle=True, random_state=yml["params"]["seed"])

# ターゲット列がトレーニングデータに存在するか確認
if yml["target_col"] not in train_df_initial.columns:
    raise ValueError(f"Target column '{yml['target_col']}' not found in initial training data.")

# ターゲットエンコーディングの対象となるカテゴリカル列を特定
object_columns_for_te = ['spc_common', 'nta'] # process.py から引き継ぎ

# --- K分割交差検証ループ ---
for fold, (train_index, val_index) in enumerate(kf.split(train_df_initial, train_df_initial[yml["target_col"]])):
    print(f"\n--- Fold {fold+1}/{yml['n_splits']} ---")

    # 現在のフォールドのトレーニングセットとバリデーションセットを分割
    fold_train_initial = train_df_initial.iloc[train_index].copy()
    fold_val_initial = train_df_initial.iloc[val_index].copy()

    # --- ターゲットエンコーディングの適用（各フォールド内で行うことでデータリークを防ぐ） ---
    if yml['target_encoding_type'] == 'jame':
        # JAMESSTEIN エンコーダを初期化
        te_encoder = JAMESSTEIN(object_columns_for_te)
        # 現在のフォールドのトレーニングデータでエンコーダを学習
        te_encoder.fit(fold_train_initial, fold_train_initial[yml['target_col']])

        # 学習したエンコーダでトレーニング、バリデーション、テストデータを変換
        fold_train_te = te_encoder.transform(fold_train_initial)
        fold_val_te = te_encoder.transform(fold_val_initial)
        test_df_te = te_encoder.transform(test_df_initial) # 全体テストデータもこのフォールドのエンコーダで変換

    elif yml['target_encoding_type'] == 'cat':
        # CATBOOST エンコーダを初期化
        te_encoder = CATBOOST(object_columns_for_te)
        # 現在のフォールドのトレーニングデータでエンコーダを学習
        te_encoder.fit(fold_train_initial, fold_train_initial[yml['target_col']])

        # 学習したエンコーダでトレーニング、バリデーション、テストデータを変換
        fold_train_te = te_encoder.transform(fold_train_initial)
        fold_val_te = te_encoder.transform(fold_val_initial)
        test_df_te = te_encoder.transform(test_df_initial)
    else:
        # ターゲットエンコーディングを行わない場合、初期の処理済みデータを使用
        fold_train_te = fold_train_initial
        fold_val_te = fold_val_initial
        test_df_te = test_df_initial

    # --- モデル固有の特徴量エンジニアリングの適用 ---
    # `fair_process` と `poor_process` は、ターゲットエンコード後のデータに適用される
    fair_fold_train = fair_process(fold_train_te)
    fair_fold_val = fair_process(fold_val_te)
    fair_test_data = fair_process(test_df_te) # テストデータにも同様に適用

    poor_fold_train = poor_process(fold_train_te)
    poor_fold_val = poor_process(fold_val_te)
    poor_test_data = poor_process(test_df_te) # テストデータにも同様に適用

    # --- モデル1 (Fairモデル) の学習と推論 ---
    # アンダーサンプリングの戦略を決定（現在のフォールドのトレーニングデータのクラス分布に基づいて）
    pi_fair = {}
    for c in [0, 1, 2]: # 想定されるターゲットクラスが0, 1, 2の場合
        # 各クラスのアンダーサンプリング後の目標数を計算
        pi_fair[c] = int(fold_train_initial[yml["target_col"]].value_counts().get(c, 0) * yml['run_params']['model1_under_sampling'][f'{"zero" if c == 0 else "one" if c == 1 else "twe"}'])

    # アンダーサンプリングの実行（特徴量エンジニアリング後のトレーニングデータに対して）
    train_under_sampling_fair, y_resampled_fair = under_sampling(fair_fold_train, pi_fair[0], pi_fair[1], pi_fair[2])

    # LightGBM モデルの入力データを準備
    X_train_fair = train_under_sampling_fair.drop(columns=[yml["target_col"]])
    y_train_fair = train_under_sampling_fair[yml["target_col"]]
    X_val_fair = fair_fold_val.drop(columns=[yml["target_col"]])
    y_val_fair = fair_fold_val[yml["target_col"]]

    # LightGBM モデルの学習と推論
    test_preds_fair_fold, oof_preds_fair_fold, score_us_fair_fold, logs_fair, model_us_fair = \
        LIGHTGBM(X_train_fair, y_train_fair, X_val_fair, y_val_fair, fair_test_data, yml)

    # 各フォールドの結果を収集
    all_test_preds_us_fair.append(test_preds_fair_fold)
    all_fold_scores_us_fair.append(score_us_fair_fold)
    oof_preds_us_fair[val_index] = oof_preds_fair_fold # 現在のフォールドのバリデーションセットに対するOOF予測値を格納
    all_models_us_fair_for_importance.append(model_us_fair) # 特徴量重要度可視化のためにモデルを収集

    # --- モデル2 (Poorモデル) の学習と推論 ---
    # アンダーサンプリングの戦略を決定（現在のフォールドのトレーニングデータのクラス分布に基づいて）
    pi_poor = {}
    for c in [0, 1, 2]: # 想定されるターゲットクラスが0, 1, 2の場合
        pi_poor[c] = int(fold_train_initial[yml["target_col"]].value_counts().get(c, 0) * yml['run_params']['model2_under_sampling'][f'{"zero" if c == 0 else "one" if c == 1 else "twe"}'])

    # アンダーサンプリングの実行
    train_under_sampling_poor, y_resampled_poor = under_sampling(poor_fold_train, pi_poor[0], pi_poor[1], pi_poor[2])

    # LightGBM モデルの入力データを準備
    X_train_poor = train_under_sampling_poor.drop(columns=[yml["target_col"]])
    y_train_poor = train_under_sampling_poor[yml["target_col"]]
    X_val_poor = poor_fold_val.drop(columns=[yml["target_col"]])
    y_val_poor = poor_fold_val[yml["target_col"]]

    # LightGBM モデルの学習と推論
    test_preds_poor_fold, oof_preds_poor_fold, score_us_poor_fold, logs_poor, model_us_poor = \
        LIGHTGBM(X_train_poor, y_train_poor, X_val_poor, y_val_poor, poor_test_data, yml)

    # 各フォールドの結果を収集
    all_test_preds_us_poor.append(test_preds_poor_fold)
    all_fold_scores_us_poor.append(score_us_poor_fold)
    oof_preds_us_poor[val_index] = oof_preds_poor_fold # 現在のフォールドのバリデーションセットに対するOOF予測値を格納
    all_models_us_poor_for_importance.append(model_us_poor) # 特徴量重要度可視化のためにモデルを収集

print("\n--- K-Fold Cross-Validation Finished ---")

# --- 結果の集計 ---
# 各フォールドのテスト予測値の平均を計算
avg_test_preds_us_fair = np.mean(all_test_preds_us_fair, axis=0)
avg_test_preds_us_poor = np.mean(all_test_preds_us_poor, axis=0)
# 各フォールドのスコアの平均を計算
avg_score_us_fair = np.mean(all_fold_scores_us_fair)
avg_score_us_poor = np.mean(all_fold_scores_us_poor)

# 最終的な予測値（FairモデルとPoorモデルの予測確率の合計をargmaxでクラスに変換）
final_ensemble_pred_probs = avg_test_preds_us_fair + avg_test_preds_us_poor
pred = np.argmax(final_ensemble_pred_probs, axis=1)

# --- 提出ファイルの生成と保存 ---
logs_summary = {
    'fair_fold_scores': all_fold_scores_us_fair,
    'poor_fold_scores': all_fold_scores_us_poor,
    'average_fair_score': avg_score_us_fair,
    'average_poor_score': avg_score_us_poor
}
save_path = submit(pred, f'{round(avg_score_us_fair, 5)}_{round(avg_score_us_poor, 5)}', logs_summary, yml)

# --- 結果の可視化 ---
# Model1 (Fair) の可視化
type = 'model1'
# 全フォールドで学習したモデルのリストを渡し、内部で重要度を平均して可視化することを想定
visualize_importance(all_models_us_fair_for_importance, fair_test_data, save_path, type)
# visualize_oof_pred(train_df_initial[yml["target_col"]], np.argmax(oof_preds_us_fair, axis=1), np.argmax(avg_test_preds_us_fair, axis=1), save_path, type, True)

# Model2 (Poor) の可視化
type = 'model2'
visualize_importance(all_models_us_poor_for_importance, poor_test_data, save_path, type)
# visualize_oof_pred(train_df_initial[yml["target_col"]], np.argmax(oof_preds_us_poor, axis=1), np.argmax(avg_test_preds_us_poor, axis=1), save_path, type, True)

# Ensemble モデルの可視化
type = 'ensemble'
# アンサンブルOOF予測値（FairとPoorのOOF確率の合計をargmax）
ensemble_oof_final = oof_preds_us_fair + oof_preds_us_poor
# visualize_oof_pred(train_df_initial[yml["target_col"]], np.argmax(ensemble_oof_final, axis=1), pred, save_path, type, False)

# --- OOFと予測データの保存 ---
np.save(save_path + '/oof_us_fair.npy', oof_preds_us_fair)
np.save(save_path + '/oof_us_poor.npy', oof_preds_us_poor)
np.save(save_path + '/pred_us_fair.npy', avg_test_preds_us_fair)
np.save(save_path + '/pred_us_poor.npy', avg_test_preds_us_poor)
np.save(save_path + '/score_us_fair.npy', all_fold_scores_us_fair) # 各フォールドのスコアを保存
np.save(save_path + '/score_us_poor.npy', all_fold_scores_us_poor) # 各フォールドのスコアを保存
