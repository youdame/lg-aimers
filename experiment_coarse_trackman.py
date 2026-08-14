"""가설: 기존 113개 트랙맨 피처(rel_speed_mean_mean_fastball 등)도 10-key match_cols로 묶여서
버킷당 표본이 2~3.7개뿐이라 노이즈일 것. coarse 그룹(카운트+손잡이)으로 다시 만들어서 비교.
"""
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import add_engineered_features
from experiment_pitchmix_coarse import COARSE_COLS

DATA_DIR = "./data"
TARGET = "control_success"

BASE_PARAMS = dict(
    max_depth=6, learning_rate=0.014399911854309236, subsample=0.6367420040370018,
    colsample_bytree=0.9920379472108309, reg_lambda=0.06047908057801884, reg_alpha=4.368166043300848,
    min_child_weight=12, gamma=0.008969955448879116, tree_method="hist", enable_categorical=True,
    random_state=42, n_jobs=-1,
)


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def process_trackman_coarse(df_main, df_trm, is_train_split=True):
    """process_trackman_features_safe와 동일하되, match_cols(10-key) 대신
    COARSE_COLS(카운트+손잡이 4개)로 그룹핑."""
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    if is_train_split:
        max_season = df_main_copy['season'].max()
        max_month = df_main_copy[df_main_copy['season'] == max_season]['game_month'].max()
        future_mask = (df_trm_copy['season'] > max_season) | \
                      ((df_trm_copy['season'] == max_season) & (df_trm_copy['game_month'] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)

    df_trm_copy['pitcher_hand'] = df_trm_copy['pitcher_hand'].map({'Left': 1, 'Right': 2}).astype(np.int64)
    df_trm_copy['batter_hand'] = df_trm_copy['batter_hand'].map({'Left': 1, 'Right': 2}).astype(np.int64)

    grouped = df_trm_copy.groupby(COARSE_COLS + ['pitch_type_group', 'auto_pitch_type'])
    grouped_phase1 = grouped[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']].agg(['mean', 'std'])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [col for col in grouped_phase2.columns if 'std' in col]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ['_'.join(col).strip('_') for col in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns='auto_pitch_type')
    grouped_phase3 = grouped_phase3.groupby(COARSE_COLS + ['pitch_type_group']).agg(['mean'])

    pivoted = grouped_phase3.unstack(level='pitch_type_group')
    pivoted.columns = [f"coarse_{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)

    tr_final = pd.merge(df_main_copy, tm_final, on=COARSE_COLS, how='left')
    new_feature_cols = [col for col in tm_final.columns if col not in COARSE_COLS]
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(tr_final[new_feature_cols].mean())

    return tr_final, new_feature_cols


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_coarse, new_cols = process_trackman_coarse(df, df_trm, is_train_split=True)
    print(f"coarse 트랙맨 피처 개수: {len(new_cols)}", flush=True)

    train_df = tr_coarse.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    categorical_features = [c for c in train_df.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna("missing").astype("category")

    X_train, y_train = train_df.loc[train_mask, features], train_df.loc[train_mask, TARGET]
    X_val, y_val = train_df.loc[val_mask, features], train_df.loc[val_mask, TARGET]

    t0 = time.time()
    model = XGBClassifier(**BASE_PARAMS, n_estimators=3000, eval_metric=brier_eval, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    _, bss = compute_bss(preds, y_val)
    score = max(0, 100000 * bss)
    print(f"[전체 트랙맨을 coarse로 재구성] n_features={len(features)} score={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
    print("참고: 기존(10-key 세분화) 트랙맨 피처 기준 baseline XGBoost = 752.70")
    print("참고: coarse 구종비중 4개만 추가했을 때 = 791.57")


if __name__ == "__main__":
    main()
