"""Track A(target-encoding 잔차, 지훈님 아이디어) 이식 검증 (EXPERIMENTS.md §7/§6-19).

시즌 진행분(채택 완료, 705.10->775.45) 위에 Track A(CatBoost 전용 6개 잔차 피처)를
추가로 얹어서 새 프로토콜(half_2024, expand_2023)로 재검증. 원작 호출 관례:
te_prior/te_source는 F1 필터 적용된 train_split 자신 기준, train_split/val_split
양쪽에 동일 적용. CatBoost에만 라우팅(MLP num_cols에서는 제외).
"""
import json
import os
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
BLEND_ALPHA = 0.61

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
    fold_df = add_engineered_features(train_df, league_success_mean)  # 시즌 진행분 포함

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)

    train_split = fold_df.loc[fold_train_mask].reset_index(drop=True)
    val_split = fold_df.loc[val_mask].reset_index(drop=True)

    # Track A: te_source/te_prior는 F1 필터 적용된 train_split 자신 기준 (원작 관례)
    te_prior = train_split[TARGET].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)

    y_val = val_split[TARGET].to_numpy()

    # --- CatBoost (+Track A) ---
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
    cat_score = score(cat_pred, y_val)
    print(f"[{fold_name}] CatBoost(+TrackA): {cat_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- MLP (7-seed, Track A 제외) ---
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
    mlp_score = score(mlp_pred, y_val)
    print(f"[{fold_name}] MLP(7-seed, Track A 제외): {mlp_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    blend_pred = BLEND_ALPHA * cat_pred + (1 - BLEND_ALPHA) * mlp_pred
    blend_score = score(blend_pred, y_val)
    print(f"[{fold_name}] Blend(alpha={BLEND_ALPHA}): {blend_score:.2f}", flush=True)
    return dict(cat=cat_score, mlp=mlp_score, blend=blend_score)


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 (+시즌 진행분 +Track A) ===")
    for key in ["cat", "mlp", "blend"]:
        vals = [results[f][key] for f in results]
        print(f"{key}: " + ", ".join(f"{f}={results[f][key]:.2f}" for f in results) + f" | 평균={np.mean(vals):.2f}")
    print("\n참고: 시즌 진행분만(Track A 없음) = CatBoost 764.86 / MLP 731.02 / Blend 775.45")


if __name__ == "__main__":
    main()
