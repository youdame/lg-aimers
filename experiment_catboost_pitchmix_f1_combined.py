"""CatBoost로 구종비중(coarse, 리크 고친 버전) + F1 필터 조합을 2023+2024
두 시즌 검증(지훈님 방식)으로 확인. XGBoost에서 평균 +283.75로 크게 나온 것이
CatBoost에서도 재현되는지 확인한다.
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

DATA_DIR = "./data"
TARGET = "control_success"

CATBOOST_PARAMS = dict(
    iterations=1500, learning_rate=0.05040411253232039, depth=7, l2_leaf_reg=3.14659036827521,
    loss_function="Logloss", border_count=106, random_seed=42, verbose=False,
    random_strength=3.715568024268865, eval_metric="BrierScore", bagging_temperature=0.4609270457436248,
    bootstrap_type="Bayesian", early_stopping_rounds=50, min_data_in_leaf=81,
)


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
        df2[col] = df2[col].astype(str).fillna("missing")

    train_split = df2.loc[tmask].reset_index(drop=True)
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
        s_base = run_one(train_df, train_mask, val_mask, apply_f1=False, label=f"{val_season} 구종비중만")
        s_f1 = run_one(train_df, train_mask, val_mask, apply_f1=True, label=f"{val_season} 구종비중+F1")
        results[val_season] = (s_base, s_f1)

    print("\n" + "=" * 60)
    print("요약 (CatBoost, 2023+2024 평균)")
    avg_base = np.mean([results[s][0] for s in [2023, 2024]])
    avg_f1 = np.mean([results[s][1] for s in [2023, 2024]])
    for s in [2023, 2024]:
        print(f"  {s} 검증 | 구종비중만: {results[s][0]:.2f} | 구종비중+F1: {results[s][1]:.2f}")
    print(f"  평균      | 구종비중만: {avg_base:.2f} | 구종비중+F1: {avg_f1:.2f} ({avg_f1-avg_base:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
