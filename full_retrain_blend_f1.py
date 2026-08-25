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
    train_mlp, make_bundle, embed_dim_for_cardinality,
)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

from feature_engineering import (
    process_trackman_features_safe, add_engineered_features,
    build_season_end_lookup, apply_te_residual_features, TE_RESIDUAL_COLS,
)
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
    print("[Full Retrain + F1] 데이터 로드 (전체 2019~2024, 홀드아웃 없음)...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    # 시즌 진행분(동섭님 피처) lookup: submit/script.py가 test.csv에 그대로 적용할 수
    # 있도록 정적으로 저장(대회 규칙: 평가 데이터 독립 예측 원칙 — test.csv 자기
    # 자신으로는 계산 불가하므로 train.csv 전체로 1회 계산). add_engineered_features가
    # 내부적으로 동일 함수를 호출해 실제 적용도 같이 수행함.
    season_end_lookup = build_season_end_lookup(train_df)
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

    # Track A(지훈님 아이디어, target-encoding 잔차 6개, CatBoost 전용): te_source/
    # te_prior는 F1 필터 적용된 train_df 자신(홀드아웃 없는 full retrain이라 train_df
    # 전체가 곧 "학습 파티션"). submit/script.py도 이 te_source/te_prior를 그대로
    # 재사용(정적 동봉)해서 test.csv에 적용.
    te_prior = train_df[TARGET].mean()
    te_source = train_df.copy()
    train_df = apply_te_residual_features(te_source, train_df, te_prior)

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS and c not in TE_RESIDUAL_COLS]
    categorical_features_cb = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    print(f"[Full Retrain] {len(train_df)}행, {len(features)}개 피처 (Track A {len(TE_RESIDUAL_COLS)}개 포함, CatBoost 전용)", flush=True)

    # --- CatBoost 전체 재학습 ---
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

    # --- MLP 전체 재학습 (7-seed) ---
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

    te_source_cols = ["pitcher_id", "batter_id", "balls_before", "strikes_before",
                       "batter_hand", "num_runners_on", "inning", "season", TARGET]
    final_bundle = {
        "catboost_model": cat_model,
        "mlp_bundle": mlp_bundle,
        "alpha": BLEND_ALPHA,
        "catboost_best_iteration": catboost_iterations,
        "f1_filter_applied": True,
        "season_end_lookup": season_end_lookup,
        "te_source": te_source[te_source_cols].copy(),
        "te_prior": te_prior,
    }
    OUT_PATH = "./submit/model/final_retained_model.pkl"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"[Full Retrain 완료] 저장: {OUT_PATH} (총 {time.time()-t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
