"""현재 프로덕션 레시피(구종비중+F1+시즌진행분+Track A)로 fold별 예측값 캐싱.

로지스틱회귀 메타모델(EXPERIMENTS.md §6-11, 창현님 저장소 +14.22) 검증용 -- 재학습
없이 메타모델만 반복 실험할 수 있도록 cat_pred/mlp_pred/y_val/context를 저장.
cache_fold_predictions.py(구버전, 시즌진행분/Track A 이전 레시피)와 구분하기 위해
파일명에 _v2 사용.
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

from feature_engineering import (
    process_trackman_features_safe, add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS,
)
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality,
)
from local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "./model_py311"

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=50, verbose=False,
)


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


def run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)  # 시즌진행분 포함

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)

    train_split = fold_df.loc[fold_train_mask].reset_index(drop=True)
    val_split = fold_df.loc[val_mask].reset_index(drop=True)

    te_prior = train_split[TARGET].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)

    y_val = val_split[TARGET].to_numpy()

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_split.columns if c not in drop_cols]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_train, cb_val = train_split.copy(), val_split.copy()
    for c in cat_features:
        cb_train[c] = cb_train[c].astype(str).fillna("missing")
        cb_val[c] = cb_val[c].astype(str).fillna("missing")
    train_pool = Pool(cb_train[features], cb_train[TARGET], cat_features=cat_features)
    val_pool = Pool(cb_val[features], cb_val[TARGET], cat_features=cat_features)

    t0 = time.time()
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    cat_model.fit(train_pool, eval_set=val_pool)
    cat_pred = cat_model.predict_proba(val_pool)[:, 1]
    print(f"[{fold_name}] CatBoost: {score(cat_pred, y_val):.2f} ({time.time()-t0:.1f}s)", flush=True)

    mlp_features = [c for c in features if c not in TE_RESIDUAL_COLS]
    num_cols = [c for c in mlp_features if c not in CAT_COLS]
    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
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

    return dict(cat_pred=cat_pred, mlp_pred=mlp_pred, y_val=y_val)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, df_trm, train_df, tr_final = build_base_df()
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        out_path = os.path.join(OUT_DIR, f"fold_predictions_v2_{fold_name}.pkl")
        if os.path.exists(out_path):
            print(f"[{fold_name}] 이미 캐싱됨, 스킵: {out_path}", flush=True)
            continue
        result = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[{fold_name}] 저장 완료: {out_path}", flush=True)


if __name__ == "__main__":
    main()
