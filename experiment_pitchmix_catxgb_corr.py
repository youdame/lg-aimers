"""구종비중(coarse, 리크 고친 버전) 적용한 CatBoost와 XGBoost를 각각 학습해서
season==2024 홀드아웃 예측값의 상관계수와 블렌드 최적 alpha/점수를 확인한다.
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # 리크 수정: season < 2024만 넘겨서 진짜 홀드아웃으로 구종비중 집계
    pivoted = compute_coarse_pitchmix(df[df["season"] < 2024], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    y_val = train_df.loc[val_mask, TARGET].values

    # --- XGBoost ---
    t0 = time.time()
    xgb_df = train_df.copy()
    cat_features_xgb = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in cat_features_xgb:
        xgb_df[col] = xgb_df[col].astype(str).fillna("missing").astype("category")
    X_train, y_train = xgb_df.loc[train_mask, features], xgb_df.loc[train_mask, TARGET]
    X_val = xgb_df.loc[val_mask, features]
    xgb_model = XGBClassifier(**XGB_PARAMS, n_estimators=3000, eval_metric=brier_eval, early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, xgb_df.loc[val_mask, TARGET])], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    _, xgb_bss = compute_bss(xgb_pred, y_val)
    xgb_score = max(0, 100000 * xgb_bss)
    print(f"XGBoost (구종비중, 리크수정): score={xgb_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- CatBoost ---
    t0 = time.time()
    cat_df = train_df.copy()
    cat_features_cb = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in cat_features_cb:
        cat_df[col] = cat_df[col].astype(str).fillna("missing")
    train_split = cat_df.loc[train_mask].reset_index(drop=True)
    val_split = cat_df.loc[val_mask].reset_index(drop=True)
    train_pool = Pool(train_split[features], train_split[TARGET], cat_features=cat_features_cb)
    val_pool = Pool(val_split[features], val_split[TARGET], cat_features=cat_features_cb)
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool)
    cat_pred = cat_model.predict_proba(val_pool)[:, 1]
    _, cat_bss = compute_bss(cat_pred, val_split[TARGET].values)
    cat_score = max(0, 100000 * cat_bss)
    print(f"CatBoost (구종비중, 리크수정): score={cat_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- 상관계수 + 블렌드 ---
    corr = np.corrcoef(cat_pred, xgb_pred)[0, 1]
    print(f"\nCatBoost vs XGBoost 상관계수 (구종비중 반영): {corr:.4f}")

    best_alpha, best_bss = None, -np.inf
    for alpha in np.arange(0.0, 1.001, 0.01):
        blend = alpha * cat_pred + (1 - alpha) * xgb_pred
        _, bss = compute_bss(blend, y_val)
        if bss > best_bss:
            best_bss, best_alpha = bss, alpha
    best_score = max(0, 100000 * best_bss)

    print("\n" + "=" * 60)
    print("요약 (구종비중 반영, 리크 수정판)")
    print(f"  CatBoost 단독            : {cat_score:.2f}")
    print(f"  XGBoost  단독            : {xgb_score:.2f}")
    print(f"  상관계수                 : {corr:.4f}")
    print(f"  블렌드 (alpha={best_alpha:.2f})        : {best_score:.2f} (CatBoost 단독 대비 {best_score-cat_score:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
