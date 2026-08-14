"""experiment_pitchmix_coarse.py의 검증 리크 수정판.

버그: compute_coarse_pitchmix(df, ...)에 season==2024(검증 대상)까지 포함된
전체 df를 넘겨서, coarse pitch-mix 집계에 검증 시즌 자신의 트랙맨 기록이 섞여
들어갔음 (COARSE_COLS에 season이 없어 전 시즌이 하나로 풀링되기 때문에,
process_trackman_features_safe의 시즌별 격리와 달리 이 리크가 그대로 남음).

수정: compute_coarse_pitchmix에는 season < 2024 (학습 구간)만 넘겨서
진짜 홀드아웃 기준 점수를 확인한다.
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

COARSE_COLS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def compute_coarse_pitchmix(df_main_for_time_filter, df_trm, is_train_split=True):
    df_trm_copy = df_trm.copy()
    if is_train_split:
        max_season = df_main_for_time_filter["season"].max()
        max_month = df_main_for_time_filter[df_main_for_time_filter["season"] == max_season]["game_month"].max()
        future_mask = (df_trm_copy["season"] > max_season) | \
                      ((df_trm_copy["season"] == max_season) & (df_trm_copy["game_month"] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)

    df_trm_copy["pitcher_hand"] = df_trm_copy["pitcher_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)
    df_trm_copy["batter_hand"] = df_trm_copy["batter_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)

    counts = df_trm_copy.groupby(COARSE_COLS + ["pitch_type_group"]).size().reset_index(name="n")
    totals = counts.groupby(COARSE_COLS)["n"].transform("sum")
    counts["pitchmix_ratio"] = counts["n"] / totals
    pivoted = counts.pivot_table(index=COARSE_COLS, columns="pitch_type_group", values="pitchmix_ratio", fill_value=0).reset_index()
    pivoted.columns = [f"coarse_pitchmix_{c}" if c not in COARSE_COLS else c for c in pivoted.columns]
    return pivoted


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # 리크 수정: season==2024(검증 대상)를 뺀 학습 구간만 넘겨서 진짜 홀드아웃으로 집계
    pivoted = compute_coarse_pitchmix(df[df["season"] < 2024], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    print(f"새 피처(coarse, 리크 수정판): {new_cols}", flush=True)

    train_df_coarse = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df_coarse[new_cols] = train_df_coarse[new_cols].fillna(train_df_coarse[new_cols].mean())

    for base_df, label in [(train_df, "baseline (123)"), (train_df_coarse, "baseline + coarse pitchmix (127)")]:
        drop_cols = ["row_id", TARGET]
        features = [c for c in base_df.columns if c not in drop_cols]
        categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
        base_df2 = base_df.copy()
        for col in categorical_features:
            base_df2[col] = base_df2[col].astype(str).fillna("missing").astype("category")

        X_train, y_train = base_df2.loc[train_mask, features], base_df2.loc[train_mask, TARGET]
        X_val, y_val = base_df2.loc[val_mask, features], base_df2.loc[val_mask, TARGET]

        t0 = time.time()
        model = XGBClassifier(**BASE_PARAMS, n_estimators=3000, eval_metric=brier_eval, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        _, bss = compute_bss(preds, y_val)
        score = max(0, 100000 * bss)
        print(f"[{label}] score={score:.2f} ({time.time()-t0:.1f}s)", flush=True)

        if new_cols[0] in features:
            importances = model.get_booster().get_score(importance_type="gain")
            for c in new_cols:
                print(f"    {c}: {importances.get(c, 0):.4f}")

    print("\n참고: 10-key(세분화) 버전 결과 = 765.02 (baseline 752.70 대비 +12.32)")


if __name__ == "__main__":
    main()
