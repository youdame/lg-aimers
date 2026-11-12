"""구종비중(coarse, 리크 고친 버전) + F1 필터(지훈님 발견, 2019~2022 game_type=='F' 제거)를
같이 적용했을 때 효과를 2023+2024 두 시즌 검증(지훈님 방식)으로 확인.
XGBoost로 먼저 확인 (CatBoost보다 빠름).
"""
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.feature_engineering import process_trackman_features_safe, add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

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
    return 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def build_base(val_season):
    """val_season 미만으로 학습, val_season으로 검증할 기본 데이터(구종비중 포함) 준비."""
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < val_season
    val_mask = train_df["season"] == val_season
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # 구종비중: val_season 미만 트랙맨만 사용 (리크 방지)
    pivoted = compute_coarse_pitchmix(df[df["season"] < val_season], df_trm, is_train_split=True)
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
    bss = compute_bss(preds, y_val.values)
    score = max(0, 100000 * bss)
    print(f"[{label}] score={score:.2f} (학습행={len(X_train)}, {time.time()-t0:.1f}s)", flush=True)
    return score


def main():
    results = {}
    for val_season in [2023, 2024]:
        print(f"\n{'='*20} 검증 시즌: {val_season} {'='*20}", flush=True)
        train_df, train_mask, val_mask = build_base(val_season)
        s_base = run_one(train_df, train_mask, val_mask, apply_f1=False, label=f"{val_season} 구종비중만")
        s_f1 = run_one(train_df, train_mask, val_mask, apply_f1=True, label=f"{val_season} 구종비중+F1")
        results[val_season] = (s_base, s_f1)

    print("\n" + "=" * 60)
    print("요약 (XGBoost, 2023+2024 평균)")
    avg_base = np.mean([results[s][0] for s in [2023, 2024]])
    avg_f1 = np.mean([results[s][0 + 1] for s in [2023, 2024]])
    for s in [2023, 2024]:
        print(f"  {s} 검증 | 구종비중만: {results[s][0]:.2f} | 구종비중+F1: {results[s][1]:.2f}")
    print(f"  평균      | 구종비중만: {avg_base:.2f} | 구종비중+F1: {avg_f1:.2f} ({avg_f1-avg_base:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
