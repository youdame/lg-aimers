"""제출용 최종 모델 (트랙맨 물리량 제외판): 전체 데이터(2019~2024, 홀드아웃 없음)로
CatBoost + MLP(7-seed) 재학습 후 블렌드 번들 저장.

full_retrain_blend_f1.py와 동일하되, `process_trackman_features_safe`(트랙맨 물리량
12개, season이 조인 키라 실제 채점 시즌엔 100% 매칭 실패)를 아예 빼고 학습.
구종비중(coarse, season 불필요)과 F1 필터는 그대로 유지.
창현님이 이 방식(F1 + 트랙맨 전체 제외)으로 971점을 내신 것 참고, 우리는 구종비중까지
유지한 채로 트랙맨 물리량만 빼는 조합을 테스트.

주의: 기존 968.15점 모델(final_retained_model_968_backup.pkl)을 덮어쓰지 않도록
별도 경로에 저장.
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

from mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, to_tensors,
    train_mlp, make_bundle, embed_dim_for_cardinality,
)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

from feature_engineering import add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

DATA_DIR = "./data"
TARGET = "control_success"

CATBOOST_BEST_ITERATION = 1412
CATBOOST_ITERATION_BUFFER = 50
DEFAULT_FULL_RETRAIN_EPOCHS = 20
FULL_RETRAIN_EPOCH_BUFFER = 5
BLEND_ALPHA = 0.61

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
    verbose=100,
)


def main():
    t_start = time.time()
    print("[Full Retrain + F1, 트랙맨 물리량 제외] 데이터 로드 (전체 2019~2024)...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    # 트랙맨 물리량 조인 없음 -- 원본 47개 컬럼(asof_* 포함) 그대로 사용
    train_df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    league_success_mean = train_df[TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df, df_trm, is_train_split=False)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    before_n = len(train_df)
    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_df = train_df[~f1_drop].reset_index(drop=True)
    print(f"[F1 필터] {f1_drop.sum()}행 제외 ({before_n} -> {len(train_df)})", flush=True)

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]
    categorical_features_cb = [c for c in train_df.select_dtypes(include="object").columns if c != "row_id" and c in features]
    print(f"[Full Retrain] {len(train_df)}행, {len(features)}개 피처 (트랙맨 물리량 제외)", flush=True)

    # --- CatBoost ---
    t0 = time.time()
    cb_df = train_df.copy()
    for col in categorical_features_cb:
        cb_df[col] = cb_df[col].astype(str).fillna("missing")
    train_pool = Pool(cb_df[features], cb_df[TARGET], cat_features=categorical_features_cb)

    catboost_iterations = CATBOOST_BEST_ITERATION + CATBOOST_ITERATION_BUFFER
    print(f"[Full Retrain] CatBoost {catboost_iterations} iteration 고정 학습 시작...", flush=True)
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=catboost_iterations)
    cat_model.fit(train_pool)
    print(f"[Full Retrain] CatBoost 완료 ({time.time()-t0:.1f}s)", flush=True)

    # --- MLP (7-seed) ---
    t0 = time.time()
    device = DEVICE
    print(f"device: {device}", flush=True)

    mlp_df = train_df[[c for c in train_df.columns if c != "row_id"]].copy()
    full_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(mlp_df, CAT_COLS, num_cols)
    X_full_cat, X_full_num, y_full = to_tensors(full_proc, CAT_COLS, num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    full_epochs = DEFAULT_FULL_RETRAIN_EPOCHS + FULL_RETRAIN_EPOCH_BUFFER
    full_members = []
    for seed in ENSEMBLE_SEEDS:
        model, _ = train_mlp(
            X_full_cat, X_full_num, y_full,
            cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
            max_epochs=full_epochs, device=device, seed=seed, verbose=False,
        )
        full_members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": full_epochs,
            "seed": seed,
        })
        print(f"[Full Retrain] MLP seed={seed} {full_epochs}epoch 학습 완료 ({time.time()-t0:.1f}s 누적)", flush=True)

    mlp_bundle = make_bundle(full_members, CAT_COLS, num_cols, cat_dims, embed_dims, cat_encoder, num_imputer, num_scaler)
    print(f"[Full Retrain] MLP 전체 완료 ({time.time()-t0:.1f}s)", flush=True)

    final_bundle = {
        "catboost_model": cat_model,
        "mlp_bundle": mlp_bundle,
        "alpha": BLEND_ALPHA,
        "catboost_best_iteration": catboost_iterations,
        "f1_filter_applied": True,
        "trackman_physical_excluded": True,
    }
    OUT_PATH = "./submit/model/final_retained_model_notrackman.pkl"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"[Full Retrain 완료] 저장: {OUT_PATH} (총 {time.time()-t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
