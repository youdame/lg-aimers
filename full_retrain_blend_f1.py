"""제출용 최종 모델 생성 (F1 필터 반영판): 전체 데이터(2019~2024, 홀드아웃 없음)로
CatBoost(재튜닝+coarse 구종비중) + MLP(coarse 구종비중, 7-seed) 재학습 후 블렌드 번들 저장.

full_retrain_blend.py와 동일하되, 지훈님이 발견한 F1 필터(2019~2022년
game_type=='F' 행 제거)를 학습 데이터에 추가로 적용. 2023+2024 이중 검증에서
CatBoost +293.78 / XGBoost +283.75로 크게 확인된 효과를 실제 제출 모델에 반영.

mlp_model.py는 이제 이 저장소에 직접 포함 (기존엔 임시 세션 scratchpad 경로에
의존해서 세션이 끝나면 사라질 위험이 있었음).
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
    train_mlp, make_bundle, embed_dim_for_cardinality, fit_quantile_edges, QUANTILE_D,
)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

DATA_DIR = "./data"
TARGET = "control_success"

CATBOOST_BEST_ITERATION = 1412
CATBOOST_ITERATION_BUFFER = 50
DEFAULT_FULL_RETRAIN_EPOCHS = 20
FULL_RETRAIN_EPOCH_BUFFER = 5
# 2026-08-16 재검증: 2023+2024 이중검증 기준 최적 alpha (기존 0.61은 season==2024 단일
# 홀드아웃으로만 정해졌던 값). MLP가 튜닝+quantile PLE+same_hand로 크게 개선되면서
# (2024: 806.77 CatBoost < 890.13 MLP로 역전) MLP 쪽 비중이 커짐.
BLEND_ALPHA = 0.33

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
    print("[Full Retrain + F1] 데이터 로드 (전체 2019~2024, 홀드아웃 없음)...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    league_success_mean = train_df[TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df, df_trm, is_train_split=False)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    # F1 필터: 2019~2022년 game_type=='F'(퓨처스) 행 제거 (지훈님 발견, 2023+2024 이중검증 +283~294)
    before_n = len(train_df)
    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_df = train_df[~f1_drop].reset_index(drop=True)
    print(f"[F1 필터] {f1_drop.sum()}행 제외 ({before_n} -> {len(train_df)})", flush=True)

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]

    # same_hand/same_hand_advantage: 2023+2024 이중검증에서 XGBoost(+14.85)/MLP(+47.90)엔
    # 도움되지만 CatBoost(-16.26)엔 손해로 나옴(2026-08-16) -> CatBoost 피처셋에서만 제외.
    CATBOOST_EXCLUDE_COLS = ["same_hand", "same_hand_advantage"]
    catboost_features = [c for c in features if c not in CATBOOST_EXCLUDE_COLS]
    categorical_features_cb = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in catboost_features]
    print(f"[Full Retrain] {len(train_df)}행, CatBoost {len(catboost_features)}개/MLP {len(features)}개 피처", flush=True)

    # --- CatBoost 전체 재학습 ---
    t0 = time.time()
    cb_df = train_df.copy()
    for col in categorical_features_cb:
        cb_df[col] = cb_df[col].astype(str).fillna("missing")
    train_pool = Pool(cb_df[catboost_features], cb_df[TARGET], cat_features=categorical_features_cb)

    catboost_iterations = CATBOOST_BEST_ITERATION + CATBOOST_ITERATION_BUFFER
    print(f"[Full Retrain] CatBoost {catboost_iterations} iteration 고정 학습 시작...", flush=True)
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=catboost_iterations)
    cat_model.fit(train_pool)
    print(f"[Full Retrain] CatBoost 완료 ({time.time()-t0:.1f}s)", flush=True)

    # --- MLP 전체 재학습 (7-seed) ---
    t0 = time.time()
    device = DEVICE
    print(f"device: {device}", flush=True)

    mlp_df = train_df[[c for c in train_df.columns if c != "row_id"]].copy()
    full_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(mlp_df, CAT_COLS, num_cols)
    X_full_cat, X_full_num, y_full = to_tensors(full_proc, CAT_COLS, num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_full_num)
    print(f"[Full Retrain] Quantile PLE bin_edges 계산 완료 (수치형 {len(num_cols)}개, "
          f"2023+2024 이중검증 +20.50 확인됨, 2026-08-16)", flush=True)

    full_epochs = DEFAULT_FULL_RETRAIN_EPOCHS + FULL_RETRAIN_EPOCH_BUFFER
    full_members = []
    for seed in ENSEMBLE_SEEDS:
        model, _ = train_mlp(
            X_full_cat, X_full_num, y_full,
            cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
            max_epochs=full_epochs, device=device, seed=seed, verbose=False,
            bin_edges=bin_edges, quantile_d=QUANTILE_D,
        )
        full_members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": full_epochs,
            "seed": seed,
        })
        print(f"[Full Retrain] MLP seed={seed} {full_epochs}epoch 학습 완료 ({time.time()-t0:.1f}s 누적)", flush=True)

    mlp_bundle = make_bundle(
        full_members, CAT_COLS, num_cols, cat_dims, embed_dims, cat_encoder, num_imputer, num_scaler,
        bin_edges=bin_edges, quantile_d=QUANTILE_D,
    )
    print(f"[Full Retrain] MLP 전체 완료 ({time.time()-t0:.1f}s)", flush=True)

    final_bundle = {
        "catboost_model": cat_model,
        "mlp_bundle": mlp_bundle,
        "alpha": BLEND_ALPHA,
        "catboost_best_iteration": catboost_iterations,
        "f1_filter_applied": True,
    }
    OUT_PATH = "./submit/model/final_retained_model.pkl"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"[Full Retrain 완료] 저장: {OUT_PATH} (총 {time.time()-t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
