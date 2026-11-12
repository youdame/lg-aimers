"""CatBoost+MLP(7-seed) 예측값을 fold별로 캐싱 (EXPERIMENTS.md §6-18 이후 블렌드 방법
실험들이 반복 재학습 없이 재사용하기 위함 — fold당 학습이 25~40분 걸려서 매 실험마다
다시 돌리면 낭비).

968.15 레시피(구종비중+F1) 기준, 새 프로토콜 필수 2-fold(half_2024, expand_2023)에서
cat_pred/mlp_pred/y_val과 컨텍스트 피처(상황 기반 동적 블렌드 메타모델에 필요)를
`model_py311/fold_predictions_<fold_name>.pkl`에 저장.
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import process_trackman_features_safe, add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality,
)
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "./model_py311"

# 메타모델(상황 기반 동적 블렌드)에서 쓸 컨텍스트 피처 후보
CONTEXT_COLS = [
    "balls_before", "strikes_before", "pitcher_hand", "batter_hand",
    "count_diff", "is_full_count", "li", "num_runners_on",
    "pitcher_relative_success",
]

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
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


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


def run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)

    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    # --- CatBoost ---
    drop_cols = ["row_id", TARGET]
    features = [c for c in fold_df.columns if c not in drop_cols]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")
    train_pool = Pool(cb_df.loc[fold_train_mask, features], cb_df.loc[fold_train_mask, TARGET], cat_features=cat_features)
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)

    t0 = time.time()
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    cat_model.fit(train_pool, eval_set=val_pool)
    cat_pred = cat_model.predict_proba(val_pool)[:, 1]
    print(f"[{fold_name}] CatBoost: {score(cat_pred, y_val):.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- MLP (7-seed) ---
    keep_cols = [c for c in fold_df.columns if c != "row_id"]
    mlp_df = fold_df[keep_cols].copy()
    mlp_features = [c for c in mlp_df.columns if c != TARGET]
    num_cols = [c for c in mlp_features if c not in CAT_COLS]

    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        mlp_df.loc[fold_train_mask].reset_index(drop=True), CAT_COLS, num_cols)
    val_proc = apply_preprocessing(
        mlp_df.loc[val_mask].reset_index(drop=True), CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t,
        seeds=ENSEMBLE_SEEDS, device=DEVICE, verbose=False,
    )
    mlp_pred = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, device=DEVICE)
    print(f"[{fold_name}] MLP(7-seed): {score(mlp_pred, y_val):.2f} ({time.time()-t0:.1f}s)", flush=True)

    context_df = fold_df.loc[val_mask, CONTEXT_COLS].reset_index(drop=True)

    return dict(cat_pred=cat_pred, mlp_pred=mlp_pred, y_val=y_val, context_df=context_df)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, df_trm, train_df, tr_final = build_base_df()
    for fold_name, train_mask, val_mask in iter_folds(train_df):
        out_path = os.path.join(OUT_DIR, f"fold_predictions_{fold_name}.pkl")
        if os.path.exists(out_path):
            print(f"[{fold_name}] 이미 캐싱됨, 스킵: {out_path}", flush=True)
            continue
        result = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[{fold_name}] 저장 완료: {out_path}", flush=True)


if __name__ == "__main__":
    main()
