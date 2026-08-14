"""CatBoost + MLP 블렌드에 F1 필터(2019~2022 game_type=='F' 제거)를 추가 적용해서
season==2024 홀드아웃으로 빠르게 확인 (전체 재학습 전 사전 검증용).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

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
    iterations=1500,
    depth=_tuned["depth"],
    learning_rate=_tuned["learning_rate"],
    l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"],
    bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"],
    min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian",
    loss_function="Logloss",
    eval_metric="BrierScore",
    random_seed=42,
    early_stopping_rounds=50,
    verbose=False,
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

    pivoted = compute_coarse_pitchmix(df, df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    # F1 필터: 학습 구간에서만 제외 (검증셋 val_mask는 그대로)
    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_mask = train_mask & (~f1_drop)
    print(f"F1 필터로 {f1_drop[train_mask.index].sum() if False else f1_drop.sum()}행 학습에서 제외 시도, 실제 학습 mask 적용", flush=True)

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
    cat_score = max(0, 100000 * cat_bss)
    print(f"CatBoost(재튜닝+coarse+F1): score={cat_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

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
    mlp_score = max(0, 100000 * mlp_bss)
    print(f"MLP(+coarse+F1): score={mlp_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nCatBoost vs MLP 상관계수: {np.corrcoef(cat_pred, mlp_pred)[0,1]:.4f}")
    best_alpha, best_bss = None, -np.inf
    for alpha in np.arange(0.0, 1.001, 0.01):
        blend = alpha * cat_pred + (1 - alpha) * mlp_pred
        _, bss = compute_bss(blend, y_val)
        if bss > best_bss:
            best_bss, best_alpha = bss, alpha
    best_score = max(0, 100000 * best_bss)

    print("\n" + "=" * 60)
    print("최종 요약 (season==2024 홀드아웃, F1 적용)")
    print(f"  CatBoost(재튜닝+coarse+F1) 단독 : {cat_score:.2f}")
    print(f"  MLP(+coarse+F1) 단독            : {mlp_score:.2f}")
    print(f"  블렌드 (alpha={best_alpha:.2f})            : {best_score:.2f}")
    print(f"  참고: F1 미적용 블렌드(오늘 시작 전) 로컬 851.19 / 실전 924")
    print(f"  참고: 2024 단일 홀드아웃은 F1 효과가 저평가될 수 있음 (2023+2024 평균이 더 정확)")
    print("=" * 60)


if __name__ == "__main__":
    main()
