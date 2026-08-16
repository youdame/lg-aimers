"""CatBoost + MLP 블렌드 alpha를 2023+2024 이중검증으로 재확인.

experiment_final_blend_f1.py는 season==2024 단일 홀드아웃만으로 alpha=0.61을 정했는데,
이건 우리가 이미 "단일시즌은 못 믿는다"고 결론 낸(EXPERIMENTS.md §3-1) 바로 그 방식이라
앞뒤가 안 맞는다는 지적이 있었음. 개별 모델/피처 검증에 쓰던 것과 동일한 2023+2024
이중검증 패턴(experiment_pitchmix_f1_combined.py 등)을 블렌드 alpha 선택에도 적용.

구종비중은 val_season 미만 트랙맨만 사용(리크 방지), F1 필터(2019~2022 game_type=='F' 제거)
적용. CatBoost는 기존 재튜닝 파라미터(best_catboost_pitchmix_hparams.json) 그대로 사용.
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
VAL_SEASONS = [2023, 2024]

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


def run_one_season(val_season):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < val_season
    val_mask = train_df["season"] == val_season
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # 구종비중: val_season 미만 트랙맨만 사용 (리크 방지)
    pivoted = compute_coarse_pitchmix(df[df["season"] < val_season], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    # F1 필터: 학습 구간에서만 적용
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
    cat_score = max(0, 100000 * cat_bss)
    print(f"[{val_season}] CatBoost: score={cat_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- MLP (7-seed 앙상블) ---
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
    print(f"[{val_season}] MLP(7-seed): score={mlp_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    corr = np.corrcoef(cat_pred, mlp_pred)[0, 1]
    print(f"[{val_season}] CatBoost vs MLP 상관계수: {corr:.4f}", flush=True)

    return dict(cat_pred=cat_pred, mlp_pred=mlp_pred, y_val=y_val, cat_score=cat_score, mlp_score=mlp_score)


def main():
    results = {s: run_one_season(s) for s in VAL_SEASONS}

    print("\n" + "=" * 60)
    print("alpha 스윕: 시즌별 최적 alpha vs 2023+2024 평균 기준 최적 alpha")
    print("=" * 60)

    per_season_best = {}
    for s in VAL_SEASONS:
        r = results[s]
        best_alpha, best_bss = None, -np.inf
        for alpha in np.arange(0.0, 1.001, 0.01):
            blend = alpha * r["cat_pred"] + (1 - alpha) * r["mlp_pred"]
            _, bss = compute_bss(blend, r["y_val"])
            if bss > best_bss:
                best_bss, best_alpha = bss, alpha
        per_season_best[s] = (best_alpha, max(0, 100000 * best_bss))
        print(f"  {s} 단독 최적: alpha={best_alpha:.2f}, score={per_season_best[s][1]:.2f} "
              f"(CatBoost={r['cat_score']:.2f}, MLP={r['mlp_score']:.2f})")

    # 2023+2024 평균 BSS를 최대화하는 alpha (기존 experiment_final_blend_f1.py의 단일시즌 방식 대체)
    best_alpha_avg, best_avg_score = None, -np.inf
    for alpha in np.arange(0.0, 1.001, 0.01):
        bss_list = []
        for s in VAL_SEASONS:
            r = results[s]
            blend = alpha * r["cat_pred"] + (1 - alpha) * r["mlp_pred"]
            _, bss = compute_bss(blend, r["y_val"])
            bss_list.append(bss)
        avg_bss = np.mean(bss_list)
        avg_score = max(0, 100000 * avg_bss)
        if avg_score > best_avg_score:
            best_avg_score, best_alpha_avg = avg_score, alpha

    print("\n" + "=" * 60)
    print(f"[결론] 2023+2024 평균 기준 최적 alpha = {best_alpha_avg:.2f} (평균 score={best_avg_score:.2f})")
    print(f"참고: 기존 실제 제출값(full_retrain_blend_f1.py) alpha=0.61, "
          f"season==2024 단일 홀드아웃으로 결정됐던 값")
    print("=" * 60)


if __name__ == "__main__":
    main()
