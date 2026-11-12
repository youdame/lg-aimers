"""same_hand / same_hand_advantage 피처 효과 검증 (2023+2024 이중검증, XGBoost).

팀원 노트에서 "좋아짐"으로 보고된 피처: 투수-타자 손이 같은지 플래그 +
"리그 평균 대비 잘하는 투수가 같은 손 매치업에서 유독 다른지" 교차 피처.
add_engineered_features()에 반영해뒀으니(feature_engineering.py), 여기선 그 컬럼
2개를 빼고 돌린 버전(적용 전) vs 그대로 둔 버전(적용 후)을 비교.
구종비중+F1은 이미 우리 현재 최선 레시피라 베이스로 깔고 검증.
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

NEW_COLS = ["same_hand", "same_hand_advantage"]


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def build_base(val_season):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < val_season
    val_mask = train_df["season"] == val_season
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[df["season"] < val_season], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_mask = train_mask & (~f1_drop)

    return train_df, train_mask, val_mask


def run_one(train_df, train_mask, val_mask, drop_new, label):
    df2 = train_df.copy()
    if drop_new:
        df2 = df2.drop(columns=NEW_COLS)

    drop_cols = ["row_id", TARGET]
    features = [c for c in df2.columns if c not in drop_cols]
    categorical_features = [c for c in df2.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        df2[col] = df2[col].astype(str).fillna("missing").astype("category")

    X_train, y_train = df2.loc[train_mask, features], df2.loc[train_mask, TARGET]
    X_val, y_val = df2.loc[val_mask, features], df2.loc[val_mask, TARGET]

    t0 = time.time()
    model = XGBClassifier(**XGB_PARAMS, n_estimators=3000, eval_metric=brier_eval, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    bss = compute_bss(preds, y_val.values)
    score = max(0, 100000 * bss)
    print(f"[{label}] score={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return score


def main():
    results = {}
    for val_season in [2023, 2024]:
        print(f"\n{'='*20} 검증 시즌: {val_season} {'='*20}", flush=True)
        train_df, train_mask, val_mask = build_base(val_season)
        s_before = run_one(train_df, train_mask, val_mask, drop_new=True, label=f"{val_season} same_hand 적용 전 (기존 레시피)")
        s_after = run_one(train_df, train_mask, val_mask, drop_new=False, label=f"{val_season} same_hand 적용 후")
        results[val_season] = (s_before, s_after)

    print("\n" + "=" * 60)
    print("요약 (XGBoost, 구종비중+F1 베이스, 2023+2024 평균)")
    avg_before = np.mean([results[s][0] for s in [2023, 2024]])
    avg_after = np.mean([results[s][1] for s in [2023, 2024]])
    for s in [2023, 2024]:
        print(f"  {s} 검증 | 적용 전: {results[s][0]:.2f} | 적용 후: {results[s][1]:.2f}")
    print(f"  평균      | 적용 전: {avg_before:.2f} | 적용 후: {avg_after:.2f} ({avg_after-avg_before:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
