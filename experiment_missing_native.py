"""feature_engineering.py의 fillna(0)/fillna(mean) 방식 vs
NaN을 그대로 남겨 XGBoost 네이티브 결측 처리에 맡기는 방식 비교.
공유 모듈(feature_engineering.py)은 건드리지 않고 로컬에 변형 버전을 둔다.
"""
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"

BASE_PARAMS = dict(
    max_depth=6,
    learning_rate=0.014399911854309236,
    subsample=0.6367420040370018,
    colsample_bytree=0.9920379472108309,
    reg_lambda=0.06047908057801884,
    reg_alpha=4.368166043300848,
    min_child_weight=12,
    gamma=0.008969955448879116,
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)


def process_trackman_features(df_main, df_trm, is_train_split=True, fill_missing=True):
    """process_trackman_features_safe()와 동일하되, fill_missing=False면
    mean류 피처의 fillna(0)/fillna(mean) 두 곳을 스킵하고 NaN을 그대로 남긴다.
    std → 0 채우기는 두 방식 모두 유지 (n=1일 때 분산 0은 통계적으로 타당한 값이라 결측 처리 대상 아님).
    """
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    if is_train_split:
        max_season = df_main_copy['season'].max()
        max_month = df_main_copy[df_main_copy['season'] == max_season]['game_month'].max()
        future_mask = (df_trm_copy['season'] > max_season) | \
                      ((df_trm_copy['season'] == max_season) & (df_trm_copy['game_month'] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)

    match_cols = [dfc for dfc in df_main_copy.columns if (dfc in df_trm_copy.columns) and dfc != 'row_id']

    grouped = df_trm_copy.groupby(match_cols + ['pitch_type_group', 'auto_pitch_type'])
    grouped_phase1 = grouped[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']].agg(['mean', 'std'])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [col for col in grouped_phase2.columns if 'std' in col]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ['_'.join(col).strip('_') for col in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns='auto_pitch_type')
    grouped_phase3 = grouped_phase3.groupby(match_cols + ['pitch_type_group']).agg(['mean'])

    pivoted = grouped_phase3.unstack(level='pitch_type_group')
    pivoted.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
    tm_final = pivoted.reset_index()
    if fill_missing:
        tm_final = tm_final.fillna(0)
    # fill_missing=False: 특정 (상황,구종군) 조합 자체가 없었던 경우 NaN 그대로 유지

    df_main_copy['top_bottom'] = df_main_copy['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    tm_final['top_bottom'] = tm_final['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)

    for col in ['batter_hand', 'pitcher_hand']:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({'Left': 1, 'Right': 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how='left')
    new_feature_cols = [col for col in tm_final.columns if col not in match_cols]
    if fill_missing:
        tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(tr_final[new_feature_cols].mean())
    # fill_missing=False: merge 자체가 안 된 행(트랙맨 이력 전무)도 NaN 그대로 유지

    return tr_final, match_cols


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def build_data(fill_missing):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features(df, df_trm, is_train_split=True, fill_missing=fill_missing)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna("missing").astype("category")

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET]
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET]

    n_nan_cols = X_train.isna().any().sum()
    print(f"  [fill_missing={fill_missing}] NaN이 하나라도 남아있는 컬럼 수: {n_nan_cols}", flush=True)
    return X_train, y_train, X_val, y_val


def fit_and_eval(X_train, y_train, X_val, y_val, label):
    t0 = time.time()
    model = XGBClassifier(
        **BASE_PARAMS,
        n_estimators=3000,
        eval_metric=brier_eval,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    brier, bss = compute_bss(preds, y_val)
    score = max(0, 100000 * bss)
    print(f"[{label}] best_iter={model.best_iteration:4d} | BSS={bss:.5f} (score={score:.2f}) | {time.time()-t0:.1f}s", flush=True)
    return bss, score


def main():
    print("[A] 현재 방식 (fillna(0) + fillna(mean))", flush=True)
    Xtr_a, ytr_a, Xval_a, yval_a = build_data(fill_missing=True)
    bss_a, score_a = fit_and_eval(Xtr_a, ytr_a, Xval_a, yval_a, "A: fillna(0)/mean (현재)")

    print("\n[B] NaN 유지 (XGBoost 네이티브 결측 분기)", flush=True)
    Xtr_b, ytr_b, Xval_b, yval_b = build_data(fill_missing=False)
    bss_b, score_b = fit_and_eval(Xtr_b, ytr_b, Xval_b, yval_b, "B: NaN native")

    print("\n" + "=" * 60)
    print(f"A (현재, fillna) : {score_a:.2f}")
    print(f"B (NaN native)   : {score_b:.2f}")
    print(f"차이 (B - A)     : {score_b - score_a:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
