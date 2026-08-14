"""XGBoost + 구종비중(coarse, 리크 고친 버전)에 지훈님의 F1 필터
(2019~2022년 game_type=='F' 행 제거)를 적용했을 때 효과를 확인.
CatBoost(+234.5)/MLP(+306.0, 2023+2024 평균 기준)에서 컸던 효과가
XGBoost에도 재현되는지 season==2024 홀드아웃으로 우선 빠르게 확인한다.
"""
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

DATA_DIR = "./data"
TARGET = "control_success"

XGB_PARAMS = dict(
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


def build_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[df["season"] < 2024], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    return train_df, train_mask, val_mask


def run_one(train_df, train_mask, val_mask, apply_f1, label):
    df2 = train_df.copy()
    tmask = train_mask.copy()
    if apply_f1:
        f1_drop = (df2["game_type"] == "F") & (df2["season"] <= 2022)
        tmask = tmask & (~f1_drop)
        print(f"[{label}] F1 필터로 제외된 행: {f1_drop[train_mask].sum()}개 (학습 {train_mask.sum()}행 중)", flush=True)

    drop_cols = ["row_id", TARGET]
    features = [c for c in df2.columns if c not in drop_cols]
    categorical_features = [c for c in df2.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        df2[col] = df2[col].astype(str).fillna("missing").astype("category")

    X_train, y_train = df2.loc[tmask, features], df2.loc[tmask, TARGET]
    X_val, y_val = df2.loc[val_mask, features], df2.loc[val_mask, TARGET]

    t0 = time.time()
    model = XGBClassifier(**XGB_PARAMS, n_estimators=3000, eval_metric=brier_eval, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    _, bss = compute_bss(preds, y_val.values)
    score = max(0, 100000 * bss)
    print(f"[{label}] score={score:.2f} (학습행={len(X_train)}, {time.time()-t0:.1f}s)", flush=True)
    return score


def main():
    print("[데이터 준비] 구종비중(리크 고친 버전) 포함...", flush=True)
    train_df, train_mask, val_mask = build_data()

    score_base = run_one(train_df, train_mask, val_mask, apply_f1=False, label="구종비중만 (F1 미적용)")
    score_f1 = run_one(train_df, train_mask, val_mask, apply_f1=True, label="구종비중 + F1 필터")

    print("\n" + "=" * 60)
    print("요약 (XGBoost, season==2024 단일 홀드아웃 기준)")
    print(f"  구종비중만        : {score_base:.2f}")
    print(f"  구종비중 + F1     : {score_f1:.2f} ({score_f1 - score_base:+.2f})")
    print("=" * 60)
    print("참고: season==2024 하나만 보는 검증이라 지훈님 방식(2023+2024 평균)보다 신뢰도는 낮음")


if __name__ == "__main__":
    main()
