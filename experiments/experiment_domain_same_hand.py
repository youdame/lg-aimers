"""same_hand / same_hand_advantage 재검증 (2026-08-26 새벽, EXPERIMENTS.md §1-6 후속).

2026-08-16 최초 테스트 때는 모델별로 정반대였음 -- XGBoost +14.85, CatBoost -16.26,
MLP +47.90(단, 그 MLP는 quantile PLE+다른 하이퍼파라미터 쓰던 옛날 버전). 그날 밤
968.15 회귀 사고에 휘말려서 원인 규명 없이 통째로 코드에서 빠진 채 지금까지 방치됨.
수연님 SHAP 분석(같은 손 매치업 그룹, 피처 2개로 기여도 6.7%)에서 다시 힌트가 나와
재검토. 지금 파이프라인(시즌진행분+Track A+reverse_rate+구종비중+F1, quantile 없음)
기준으로 CatBoost/MLP 둘 다 정식 재검증 -- MLP가 핵심이라 7-seed 풀블렌드로 확인.

기준선(오늘 밤 제출 번들의 CatBoost+MLP 부분) = experiment_final_bundle_baseline.py
결과: half_2024 CatBoost 769.10/MLP 775.33/Blend 801.19,
expand_2023 CatBoost 822.28/MLP 740.21/Blend 819.17.
(주의: 그 기준선은 reverse_rate가 없던 시점 실험도 있었으나, 이 스크립트는
add_engineered_features 최신판을 그대로 쓰므로 reverse_rate 포함됨 -- 즉 이 스크립트
자체 baseline 대비 same_hand 유무 델타만 신뢰하면 됨, 위 숫자와 완전히 동일하진
않을 수 있음.)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import process_trackman_features_safe, add_engineered_features, apply_te_residual_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality,
)
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BLEND_ALPHA = 0.61


def add_same_hand(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"].astype(str) == df["batter_hand"].astype(str)).astype(np.int64)
    df["same_hand_advantage"] = df["pitcher_relative_success"] * df["same_hand"]
    return df


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


def run_variant(fold_df, tr_final, fold_train_mask, val_mask, y_val, fold_name, variant_name):
    features = [c for c in fold_df.columns if c not in ["row_id", TARGET]]
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
    cat_score = score(cat_pred, y_val)
    print(f"[{fold_name}][{variant_name}] CatBoost: {cat_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    from src.feature_engineering import TE_RESIDUAL_COLS
    keep_cols = [c for c in fold_df.columns if c != "row_id" and c not in TE_RESIDUAL_COLS]
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
    mlp_score = score(mlp_pred, y_val)
    print(f"[{fold_name}][{variant_name}] MLP(7-seed): {mlp_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    blend_pred = BLEND_ALPHA * cat_pred + (1 - BLEND_ALPHA) * mlp_pred
    blend_score = score(blend_pred, y_val)
    print(f"[{fold_name}][{variant_name}] Blend: {blend_score:.2f}", flush=True)
    return dict(cat=cat_score, mlp=mlp_score, blend=blend_score)


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

    te_prior = fold_df.loc[fold_train_mask, TARGET].mean()
    te_source = fold_df.loc[fold_train_mask]
    base_df = apply_te_residual_features(te_source, fold_df.copy(), te_prior)

    results = {}
    results["baseline"] = run_variant(base_df, tr_final, fold_train_mask, val_mask, y_val, fold_name, "baseline")

    sh_df = add_same_hand(base_df.copy())
    results["same_hand"] = run_variant(sh_df, tr_final, fold_train_mask, val_mask, y_val, fold_name, "same_hand")
    return results


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    all_results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        all_results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 (same_hand / same_hand_advantage) ===")
    for variant in ["baseline", "same_hand"]:
        for key in ["cat", "mlp", "blend"]:
            vals = [all_results[f][variant][key] for f in all_results]
            avg = np.mean(vals)
            line = ", ".join(f"{f}={all_results[f][variant][key]:.2f}" for f in all_results)
            print(f"{variant}/{key}: {line} | 평균={avg:.2f}")


if __name__ == "__main__":
    main()
