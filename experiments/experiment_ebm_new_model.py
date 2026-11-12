"""EBM(Explainable Boosting Machine) 새 모델 타입 시도 (EXPERIMENTS.md §6-19 후속).

CatBoost(트리 앙상블)/MLP(신경망)/TabNet/ExcelFormer/LightGBM/XGBoost/RandomForest 전부
기각된 상황에서, GAM(Generalized Additive Model) 계열인 EBM은 팀 전체에서 아무도 안
해본 구조적으로 다른 모델 패밀리. 968.15+시즌진행분+Track A 레시피(CatBoost와 동일
피처셋)로 새 프로토콜(half_2024, expand_2023)에서 (1) EBM 단독 성능, (2) 기존
CatBoost+MLP 블렌드에 3번째 모델로 추가했을 때 이득이 있는지 확인.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from interpret.glassbox import ExplainableBoostingClassifier

from src.feature_engineering import (
    process_trackman_features_safe, add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS,
)
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality,
)
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

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
    fold_df = add_engineered_features(train_df, league_success_mean)

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

    # --- CatBoost (기준선, CatBoost와 동일 피처셋) ---
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
    print(f"[{fold_name}] CatBoost: {cat_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # --- EBM (CatBoost와 동일 피처셋, object 컬럼은 category dtype으로) ---
    ebm_train, ebm_val = train_split.copy(), val_split.copy()
    for c in cat_features:
        ebm_train[c] = ebm_train[c].astype(str).fillna("missing").astype("category")
        ebm_val[c] = ebm_val[c].astype(str).fillna("missing").astype("category")
    t0 = time.time()
    ebm = ExplainableBoostingClassifier(random_state=42, n_jobs=4, interactions=10)
    ebm.fit(ebm_train[features], ebm_train[TARGET])
    ebm_pred = ebm.predict_proba(ebm_val[features])[:, 1]
    ebm_score = score(ebm_pred, y_val)
    print(f"[{fold_name}] EBM: {ebm_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

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
    print(f"[{fold_name}] MLP(7-seed): {mlp_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    corr_cat_ebm = np.corrcoef(cat_pred, ebm_pred)[0, 1]
    corr_mlp_ebm = np.corrcoef(mlp_pred, ebm_pred)[0, 1]
    print(f"[{fold_name}] 상관계수: EBM-CatBoost={corr_cat_ebm:.4f} | EBM-MLP={corr_mlp_ebm:.4f}", flush=True)

    # 2-way(기존) vs 3-way(+EBM) 블렌드 alpha 스윕
    best_2way = max(0, 100000 * (1 - ((0.61*cat_pred+0.39*mlp_pred - y_val)**2).mean() / (y_val.mean()*(1-y_val.mean()))))
    best_3way_score, best_w = -1, None
    for w_cat in np.arange(0, 1.01, 0.1):
        for w_ebm in np.arange(0, 1.01 - w_cat, 0.1):
            w_mlp = 1 - w_cat - w_ebm
            blend = w_cat*cat_pred + w_ebm*ebm_pred + w_mlp*mlp_pred
            s = score(blend, y_val)
            if s > best_3way_score:
                best_3way_score, best_w = s, (round(w_cat,2), round(w_ebm,2), round(w_mlp,2))
    print(f"[{fold_name}] 2-way(alpha=0.61) 블렌드: {best_2way:.2f} | "
          f"3-way(+EBM) 최적: {best_3way_score:.2f} (w_cat/w_ebm/w_mlp={best_w}) "
          f"| Δ={best_3way_score-best_2way:+.2f}", flush=True)

    return dict(cat=cat_score, ebm=ebm_score, mlp=mlp_score, blend2=best_2way, blend3=best_3way_score)


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 ===")
    for key in ["cat", "ebm", "mlp", "blend2", "blend3"]:
        vals = [results[f][key] for f in results]
        print(f"{key}: " + ", ".join(f"{f}={results[f][key]:.2f}" for f in results) + f" | 평균={np.mean(vals):.2f}")


if __name__ == "__main__":
    main()
