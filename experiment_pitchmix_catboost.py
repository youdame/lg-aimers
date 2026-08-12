"""구종비중x카운트 피처(situational_pitchmix_*)가 CatBoost에도 도움되는지 로컬에서 확인.
CATBOOST_PARAMS는 lgaimers9/open/reference/best_model.pkl에서 추출한 튜닝된 값(818.54) 그대로.
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_by_count import compute_situational_pitchmix, add_pitchmix_by_situation

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

    pivoted = compute_situational_pitchmix(df, df_trm, match_cols, is_train_split=True)
    train_df_pm, new_cols = add_pitchmix_by_situation(train_df, pivoted, match_cols)
    print(f"새 피처: {new_cols}", flush=True)

    for base_df, label in [(train_df, "baseline (123 features)"), (train_df_pm, "baseline + pitchmix (127 features)")]:
        drop_cols = ["row_id", TARGET]
        features = [c for c in base_df.columns if c not in drop_cols]
        categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
        base_df = base_df.copy()
        for col in categorical_features:
            base_df[col] = base_df[col].astype(str).fillna("missing")

        train_split = base_df.loc[train_mask].reset_index(drop=True)
        val_split = base_df.loc[val_mask].reset_index(drop=True)

        train_pool = Pool(train_split[features], train_split[TARGET], cat_features=categorical_features)
        val_pool = Pool(val_split[features], val_split[TARGET], cat_features=categorical_features)

        t0 = time.time()
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict_proba(val_pool)[:, 1]
        _, bss = compute_bss(preds, val_split[TARGET].values)
        score = max(0, 100000 * bss)
        print(f"[{label}] n_features={len(features)} score={score:.2f} best_iter={model.get_best_iteration()} ({time.time()-t0:.1f}s)", flush=True)

        if base_df is train_df_pm:
            importances = model.get_feature_importance(train_pool)
            fmap = dict(zip(features, importances))
            print("  새 피처 importance:")
            for c in new_cols:
                print(f"    {c}: {fmap.get(c, 0):.4f}")


if __name__ == "__main__":
    main()
