"""메타모델 계수 재추정 (2026-08-26 새벽, EXPERIMENTS.md 큐 3번).

지금 쓰는 메타모델 계수(w_cat=2.002827, w_mlp=1.988260, intercept=-2.022844)는
reverse_rate/same_hand 반영 전 피처셋의 CatBoost/MLP 예측값으로 학습된 것. 오늘 밤
새로 반영된 피처들로 CatBoost(5-seed, 프로덕션과 동일)+MLP(7-seed, same_hand 포함)를
half_2024 fold에서 학습해서 예측값을 뽑고, 창현님 방법론(같은 레짐 내부, 70/30 재분할,
alpha/메타모델 둘 다 해당 fold에서만 재학습)대로 새 계수를 fitting. 기존 고정 alpha
블렌드와 새 메타모델 둘 다 같은 30% 홀드아웃에서 비교.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

from feature_engineering import (
    process_trackman_features_safe, add_engineered_features, apply_te_residual_features,
    add_same_hand, SAME_HAND_COLS, TE_RESIDUAL_COLS,
)
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality,
)
from local_validation import half_2024_fold, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CATBOOST_SEEDS = ENSEMBLE_SEEDS[:5]

# 기존(고정) 메타모델 계수
OLD_W_CAT, OLD_W_MLP, OLD_INTERCEPT = 2.002827, 1.988260, -2.022844

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    early_stopping_rounds=50, verbose=False,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    t_start = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask, val_mask = half_2024_fold(train_df)
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)

    te_prior = fold_df.loc[fold_train_mask, TARGET].mean()
    te_source = fold_df.loc[fold_train_mask]
    fold_df = apply_te_residual_features(te_source, fold_df, te_prior)
    fold_df = add_same_hand(fold_df)

    drop_cols = ["row_id", TARGET]
    all_cols = [c for c in fold_df.columns if c not in drop_cols]
    cb_features = [c for c in all_cols if c not in SAME_HAND_COLS]
    num_cols = [c for c in all_cols if c not in CAT_COLS and c not in TE_RESIDUAL_COLS]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in cb_features]

    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")
    train_pool = Pool(cb_df.loc[fold_train_mask, cb_features], cb_df.loc[fold_train_mask, TARGET], cat_features=cat_features)
    val_pool = Pool(cb_df.loc[val_mask, cb_features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)
    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    print(f"[메타 재추정] CatBoost {len(CATBOOST_SEEDS)}-seed 학습 시작...", flush=True)
    t0 = time.time()
    cat_preds_list = []
    for seed in CATBOOST_SEEDS:
        m = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500, random_seed=seed)
        m.fit(train_pool, eval_set=val_pool)
        cat_preds_list.append(m.predict_proba(val_pool)[:, 1])
    cat_pred = np.mean(cat_preds_list, axis=0)
    print(f"[메타 재추정] CatBoost 완료 ({time.time()-t0:.1f}s), 단독 점수={score(cat_pred, y_val):.2f}", flush=True)

    t0 = time.time()
    mlp_df = fold_df[[c for c in fold_df.columns if c != "row_id"]].copy()
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
    print(f"[메타 재추정] MLP 완료 ({time.time()-t0:.1f}s), 단독 점수={score(mlp_pred, y_val):.2f}", flush=True)

    # 창현님 방법론: 이 fold의 val_mask(진짜 홀드아웃)를 다시 70/30으로 나눠서
    # 70%로 alpha/메타모델 재학습, 30%에서 평가.
    rng = np.random.RandomState(42)
    n_val = len(y_val)
    perm = rng.permutation(n_val)
    split = int(n_val * 0.7)
    fit_idx, eval_idx = perm[:split], perm[split:]

    # 고정 alpha 블렌드(기존 방식) 비교용: 30% eval set에서 alpha 재최적화
    best_alpha, best_alpha_score = 0.5, -1
    for a in np.arange(0.3, 0.8, 0.01):
        s = score(a * cat_pred[fit_idx] + (1 - a) * mlp_pred[fit_idx], y_val[fit_idx])
        if s > best_alpha_score:
            best_alpha, best_alpha_score = a, s
    alpha_eval_score = score(best_alpha * cat_pred[eval_idx] + (1 - best_alpha) * mlp_pred[eval_idx], y_val[eval_idx])

    # 기존 고정 메타모델 계수를 그대로 eval set에 적용
    old_meta_pred_eval = sigmoid(OLD_W_CAT * cat_pred[eval_idx] + OLD_W_MLP * mlp_pred[eval_idx] + OLD_INTERCEPT)
    old_meta_eval_score = score(old_meta_pred_eval, y_val[eval_idx])

    # 새 메타모델: 70%로 재학습
    X_fit = np.column_stack([cat_pred[fit_idx], mlp_pred[fit_idx]])
    clf = LogisticRegression()
    clf.fit(X_fit, y_val[fit_idx])
    new_w_cat, new_w_mlp = clf.coef_[0]
    new_intercept = clf.intercept_[0]
    new_meta_pred_eval = sigmoid(new_w_cat * cat_pred[eval_idx] + new_w_mlp * mlp_pred[eval_idx] + new_intercept)
    new_meta_eval_score = score(new_meta_pred_eval, y_val[eval_idx])

    print("\n=== 결과 (30% eval set, half_2024 fold) ===")
    print(f"고정 alpha={best_alpha:.2f} 블렌드: {alpha_eval_score:.2f}")
    print(f"기존 메타모델 계수 그대로: {old_meta_eval_score:.2f}")
    print(f"새로 재학습한 메타모델: {new_meta_eval_score:.2f}")
    print(f"새 계수: w_cat={new_w_cat:.6f}, w_mlp={new_w_mlp:.6f}, intercept={new_intercept:.6f}")
    print(f"\n총 소요 {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
