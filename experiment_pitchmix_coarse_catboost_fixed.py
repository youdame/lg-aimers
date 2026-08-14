"""coarse 구종비중 피처(카운트+손잡이만 그룹, 버킷당 표본 훨씬 많음)를 CatBoost로 테스트."""
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
    return brier, 1.0 - (brier / baseline_brier)


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
            base_df2[col] = base_df2[col].astype(str).fillna("missing")

        train_split = base_df2.loc[train_mask].reset_index(drop=True)
        val_split = base_df2.loc[val_mask].reset_index(drop=True)
        train_pool = Pool(train_split[features], train_split[TARGET], cat_features=categorical_features)
        val_pool = Pool(val_split[features], val_split[TARGET], cat_features=categorical_features)

        t0 = time.time()
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict_proba(val_pool)[:, 1]
        _, bss = compute_bss(preds, val_split[TARGET].values)
        score = max(0, 100000 * bss)
        print(f"[{label}] score={score:.2f} best_iter={model.get_best_iteration()} ({time.time()-t0:.1f}s)", flush=True)

        if new_cols[0] in features:
            importances = model.get_feature_importance(train_pool)
            fmap = dict(zip(features, importances))
            for c in new_cols:
                print(f"    {c}: {fmap.get(c, 0):.4f}")

    print("\n참고: CatBoost baseline=777.71, 10-key(세분화) 버전=805.51 (+27.80)")


if __name__ == "__main__":
    main()
