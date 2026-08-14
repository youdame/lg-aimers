"""CatBoost + MLP + XGBoost 3-way 블렌드를 구종비중(리크 고친 버전) + F1 필터
반영해서 season==2024 홀드아웃으로 확인. 예전 3-way 테스트는 MLP가 약해서(38.96)
XGBoost에 밀렸었는데, 지금 MLP(806.69)는 훨씬 강해져서 결과가 다를 수 있음.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality,
)

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    iterations=1500, depth=_tuned["depth"], learning_rate=_tuned["learning_rate"],
    l2_leaf_reg=_tuned["l2_leaf_reg"], random_strength=_tuned["random_strength"],
    bagging_temperature=_tuned["bagging_temperature"], border_count=_tuned["border_count"],
    min_data_in_leaf=_tuned["min_data_in_leaf"], bootstrap_type="Bayesian",
    loss_function="Logloss", eval_metric="BrierScore", random_seed=42,
    early_stopping_rounds=50, verbose=False,
)

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


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[df["season"] < 2024], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_mask = train_mask & (~f1_drop)

    y_val = train_df.loc[val_mask, TARGET].values

    # --- CatBoost ---
    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_df = train_df.copy()
    for col in categorical_features:
        cb_df[col] = cb_df[col].astype(str).fillna("missing")
    train_split_cb = cb_df.loc[train_mask].reset_index(drop=True)
    val_split_cb = cb_df.loc[val_mask].reset_index(drop=True)
    train_pool = Pool(train_split_cb[features], train_split_cb[TARGET], cat_features=categorical_features)
    val_pool = Pool(val_split_cb[features], val_split_cb[TARGET], cat_features=categorical_features)

    t0 = time.time()
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(train_pool, eval_set=val_pool)
    cat_pred = cat_model.predict_proba(val_pool)[:, 1]
    _, cat_bss = compute_bss(cat_pred, y_val)
    print(f"CatBoost: score={max(0,100000*cat_bss):.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- XGBoost ---
    xgb_df = train_df.copy()
    for col in categorical_features:
        xgb_df[col] = xgb_df[col].astype(str).fillna("missing").astype("category")
    X_train, y_train = xgb_df.loc[train_mask, features], xgb_df.loc[train_mask, TARGET]
    X_val = xgb_df.loc[val_mask, features]
    t0 = time.time()
    xgb_model = XGBClassifier(**XGB_PARAMS, n_estimators=3000, eval_metric=brier_eval, early_stopping_rounds=50)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, xgb_df.loc[val_mask, TARGET])], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    _, xgb_bss = compute_bss(xgb_pred, y_val)
    print(f"XGBoost: score={max(0,100000*xgb_bss):.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- MLP ---
    keep_cols = [c for c in train_df.columns if c != "row_id"]
    mlp_df = train_df[keep_cols].copy()
    mlp_features = [c for c in mlp_df.columns if c != TARGET]
    num_cols = [c for c in mlp_features if c not in CAT_COLS]
    train_split_mlp = mlp_df.loc[train_mask].reset_index(drop=True)
    val_split_mlp = mlp_df.loc[val_mask].reset_index(drop=True)

    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split_mlp, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split_mlp, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t, embed_dims=embed_dims,
        seeds=ENSEMBLE_SEEDS, device=DEVICE, verbose=False,
    )
    mlp_pred = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, device=DEVICE)
    _, mlp_bss = compute_bss(mlp_pred, y_val)
    print(f"MLP: score={max(0,100000*mlp_bss):.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- 상관계수 ---
    print("\n[상관계수]")
    print(f"  CatBoost vs XGBoost: {np.corrcoef(cat_pred, xgb_pred)[0,1]:.4f}")
    print(f"  CatBoost vs MLP    : {np.corrcoef(cat_pred, mlp_pred)[0,1]:.4f}")
    print(f"  XGBoost  vs MLP    : {np.corrcoef(xgb_pred, mlp_pred)[0,1]:.4f}")

    # --- 2-way (CatBoost+MLP) 기준 ---
    best_alpha2, best_bss2 = None, -np.inf
    for alpha in np.arange(0.0, 1.001, 0.01):
        blend = alpha * cat_pred + (1 - alpha) * mlp_pred
        _, bss = compute_bss(blend, y_val)
        if bss > best_bss2:
            best_bss2, best_alpha2 = bss, alpha
    best_score2 = max(0, 100000 * best_bss2)

    # --- 3-way 그리드서치 ---
    best_w, best_bss3 = None, -np.inf
    for w1 in np.arange(0, 1.01, 0.02):
        for w2 in np.arange(0, 1.01 - w1, 0.02):
            w3 = 1 - w1 - w2
            blend = w1 * cat_pred + w2 * xgb_pred + w3 * mlp_pred
            _, bss = compute_bss(blend, y_val)
            if bss > best_bss3:
                best_bss3, best_w = bss, (round(w1, 2), round(w2, 2), round(w3, 2))
    best_score3 = max(0, 100000 * best_bss3)

    print("\n" + "=" * 60)
    print("요약 (season==2024, 구종비중+F1)")
    print(f"  CatBoost 단독              : {max(0,100000*cat_bss):.2f}")
    print(f"  XGBoost  단독              : {max(0,100000*xgb_bss):.2f}")
    print(f"  MLP      단독              : {max(0,100000*mlp_bss):.2f}")
    print(f"  Cat+MLP 2-way (alpha={best_alpha2:.2f})   : {best_score2:.2f}")
    print(f"  Cat+XGB+MLP 3-way weights(Cat,XGB,MLP)={best_w} : {best_score3:.2f} ({best_score3-best_score2:+.2f} vs 2-way)")
    print("=" * 60)


if __name__ == "__main__":
    main()
