"""same_hand / same_hand_advantage 피처 효과 검증 (2023+2024 이중검증, CatBoost).

experiment_same_hand.py(XGBoost, +14.85)와 동일한 비교를 CatBoost로도 확인.
구종비중+F1은 이미 현재 최선 레시피라 베이스로 깔고, same_hand 두 컬럼만 있고/없고를 비교.
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import process_trackman_features_safe, add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

DATA_DIR = "./data"
TARGET = "control_success"

CATBOOST_PARAMS = dict(
    iterations=1500, learning_rate=0.05040411253232039, depth=7, l2_leaf_reg=3.14659036827521,
    loss_function="Logloss", border_count=106, random_seed=42, verbose=False,
    random_strength=3.715568024268865, eval_metric="BrierScore", bagging_temperature=0.4609270457436248,
    bootstrap_type="Bayesian", early_stopping_rounds=50, min_data_in_leaf=81,
)

NEW_COLS = ["same_hand", "same_hand_advantage"]


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


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
        df2[col] = df2[col].astype(str).fillna("missing")

    train_split = df2.loc[train_mask].reset_index(drop=True)
    val_split = df2.loc[val_mask].reset_index(drop=True)
    train_pool = Pool(train_split[features], train_split[TARGET], cat_features=categorical_features)
    val_pool = Pool(val_split[features], val_split[TARGET], cat_features=categorical_features)

    t0 = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool)
    preds = model.predict_proba(val_pool)[:, 1]
    bss = compute_bss(preds, val_split[TARGET].values)
    score = max(0, 100000 * bss)
    print(f"[{label}] score={score:.2f} (학습행={len(train_split)}, {time.time()-t0:.1f}s)", flush=True)
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
    print("요약 (CatBoost, 구종비중+F1 베이스, 2023+2024 평균)")
    avg_before = np.mean([results[s][0] for s in [2023, 2024]])
    avg_after = np.mean([results[s][1] for s in [2023, 2024]])
    for s in [2023, 2024]:
        print(f"  {s} 검증 | 적용 전: {results[s][0]:.2f} | 적용 후: {results[s][1]:.2f}")
    print(f"  평균      | 적용 전: {avg_before:.2f} | 적용 후: {avg_after:.2f} ({avg_after-avg_before:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
