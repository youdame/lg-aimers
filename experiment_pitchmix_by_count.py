"""트랙맨 pitch-mix x 상황(카운트) 교차 피처 실험.
match_cols(=season/game_month/.../balls_before/strikes_before 등, 개인 투수ID는 없음)로
묶은 상황별 pitch_type_group 비중을 새 피처로 추가하고, XGBoost로 빠르게 검증한다.
"""
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import process_trackman_features_safe, add_engineered_features

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


def compute_situational_pitchmix(df_main_for_time_filter, df_trm, match_cols, is_train_split=True):
    """match_cols(개인 투수 ID 없는 상황 지문) 기준으로, 해당 상황에서 트랙맨 로그의
    pitch_type_group 비중(fastball/breaking/offspeed/other 비율) 피벗 테이블을 계산해서 반환.
    (main 쪽에 merge하는 건 호출부에서, train_df -- 이미 top_bottom/hand가 int로 변환된 상태 -- 에 직접 함)"""
    df_trm_copy = df_trm.copy()

    if is_train_split:
        max_season = df_main_for_time_filter['season'].max()
        max_month = df_main_for_time_filter[df_main_for_time_filter['season'] == max_season]['game_month'].max()
        future_mask = (df_trm_copy['season'] > max_season) | \
                      ((df_trm_copy['season'] == max_season) & (df_trm_copy['game_month'] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)

    # 상황별 pitch_type_group 비중 계산
    counts = df_trm_copy.groupby(match_cols + ['pitch_type_group']).size().reset_index(name='n')
    totals = counts.groupby(match_cols)['n'].transform('sum')
    counts['pitchmix_ratio'] = counts['n'] / totals
    pivoted = counts.pivot_table(index=match_cols, columns='pitch_type_group', values='pitchmix_ratio', fill_value=0).reset_index()
    pivoted.columns = [f"situational_pitchmix_{c}" if c not in match_cols else c for c in pivoted.columns]

    # 트랙맨 쪽 표기를 main(=train_df, 이미 int로 변환됨) 쪽과 맞춤:
    # top_bottom: 트랙맨은 'Top'/'Bottom' -> 0/1, pitcher_hand/batter_hand: 트랙맨은 'Left'/'Right' -> 1/2
    if 'top_bottom' in pivoted.columns:
        pivoted['top_bottom'] = pivoted['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)
    for col in ['pitcher_hand', 'batter_hand']:
        if col in pivoted.columns:
            pivoted[col] = pivoted[col].map({'Left': 1, 'Right': 2}).astype(np.int64)

    return pivoted


def add_pitchmix_by_situation(train_df, pivoted, match_cols):
    """이미 완성된 train_df(기존 123피처)에 situational pitchmix 피처를 merge."""
    merged = pd.merge(train_df, pivoted, on=match_cols, how='left')
    new_cols = [c for c in pivoted.columns if c not in match_cols]
    merged[new_cols] = merged[new_cols].fillna(merged[new_cols].mean())
    return merged, new_cols


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    print(f"match_cols: {match_cols}", flush=True)

    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # 새 피처 추가: 기존 123피처 완성본(train_df) 위에 그대로 얹음 (raw df에서 다시 시작 X)
    pivoted = compute_situational_pitchmix(df, df_trm, match_cols, is_train_split=True)
    train_df2, new_cols = add_pitchmix_by_situation(train_df, pivoted, match_cols)
    print(f"새로 추가된 피처: {new_cols}", flush=True)

    for base_df, label, extra_cols in [(train_df, "baseline (기존 123피처)", []), (train_df2, "baseline + pitchmix_by_count", new_cols)]:
        drop_cols = ['row_id', TARGET]
        features = [c for c in base_df.columns if c not in drop_cols]
        categorical_features = [c for c in tr_final.select_dtypes(include='object').columns if c != 'row_id' and c in features]
        base_df = base_df.copy()
        for col in categorical_features:
            base_df[col] = base_df[col].astype(str).fillna('missing').astype('category')

        tmask = base_df['season'] < 2024
        vmask = base_df['season'] == 2024
        X_train, y_train = base_df.loc[tmask, features], base_df.loc[tmask, TARGET]
        X_val, y_val = base_df.loc[vmask, features], base_df.loc[vmask, TARGET]

        t0 = time.time()
        model = XGBClassifier(**BASE_PARAMS, n_estimators=3000, eval_metric=brier_eval, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        _, bss = compute_bss(preds, y_val)
        score = max(0, 100000 * bss)
        print(f"[{label}] n_features={len(features)} score={score:.2f} ({time.time()-t0:.1f}s)", flush=True)

        if extra_cols:
            importances = model.get_booster().get_score(importance_type="gain")
            print("  새 피처 importance:")
            for c in extra_cols:
                print(f"    {c}: {importances.get(c, 0):.4f}")


if __name__ == "__main__":
    main()
