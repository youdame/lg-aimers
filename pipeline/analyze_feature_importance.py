"""오늘 밤 최종 재학습 결과물(`submit/model/final_retained_model.pkl`)에서 CatBoost
내장 feature importance + MLP 순열 중요도(permutation importance)를 뽑는다
(2026-08-25 밤, 재학습 없이 이미 학습된 모델에서 계산만 하므로 가벼움).

CatBoost: get_feature_importance() (PredictionValuesChange, 기본값) 바로 사용.
MLP: 학습에 쓴 전체 데이터로 permutation importance 계산 (홀드아웃 없는 full retrain
이라 어쩔 수 없이 train 데이터 자체를 평가셋으로 씀 -- 절대 점수가 아니라 "상대적
피처 순위"만 보는 용도라 큰 문제 없음). 시간 절약을 위해 10만 행 샘플로 계산.
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch

from src.feature_engineering import (
    process_trackman_features_safe, add_engineered_features, apply_te_residual_features,
)
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.mlp_model import predict_bundle
from src.local_validation import compute_bss

DATA_DIR = "./data"
TARGET = "control_success"
MODEL_PATH = "./submit/model/final_retained_model.pkl"
SAMPLE_N = 100_000
SEED = 42


def score(preds, y):
    return max(0.0, 100000 * compute_bss(preds, y))


def build_full_features():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    league_success_mean = train_df[TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df, df_trm, is_train_split=False)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_df = train_df[~f1_drop].reset_index(drop=True)

    te_prior = train_df[TARGET].mean()
    te_source = train_df.copy()
    train_df = apply_te_residual_features(te_source, train_df, te_prior)
    return train_df


def catboost_importance(bundle, features):
    cat_model = bundle["catboost_models"][0] if "catboost_models" in bundle else bundle["catboost_model"]
    importances = cat_model.get_feature_importance()
    order = np.argsort(importances)[::-1]
    print("\n=== CatBoost feature importance (상위 30개, PredictionValuesChange) ===")
    for i in order[:30]:
        print(f"{features[i]:40s} {importances[i]:.3f}")
    return list(zip(np.array(features)[order], importances[order]))


def mlp_permutation_importance(bundle, mlp_bundle, df_sample, y_sample, device):
    base_pred = predict_bundle(mlp_bundle, df_sample, device=device)
    base_score = score(base_pred, y_sample)
    print(f"\nMLP baseline BSS score(샘플 {len(df_sample)}행): {base_score:.2f}")

    rng = np.random.RandomState(SEED)
    drops = {}
    num_cols = mlp_bundle["num_cols"]
    for i, col in enumerate(num_cols):
        perturbed = df_sample.copy()
        perturbed[col] = rng.permutation(perturbed[col].values)
        pred = predict_bundle(mlp_bundle, perturbed, device=device)
        s = score(pred, y_sample)
        drops[col] = base_score - s
        if (i + 1) % 20 == 0:
            print(f"  ({i+1}/{len(num_cols)}개 완료)", flush=True)

    ranked = sorted(drops.items(), key=lambda kv: kv[1], reverse=True)
    print("\n=== MLP 순열 중요도 (상위 30개, 점수 하락폭 기준) ===")
    for col, drop in ranked[:30]:
        print(f"{col:40s} {drop:+.3f}")
    return ranked


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 아직 없음: {MODEL_PATH} (재학습 먼저 끝나야 함)")
        return

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    train_df = build_full_features()
    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]

    cb_importance = catboost_importance(bundle, features)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    sample_df = train_df.sample(n=min(SAMPLE_N, len(train_df)), random_state=SEED).reset_index(drop=True)
    y_sample = sample_df[TARGET].to_numpy()
    mlp_importance = mlp_permutation_importance(bundle, bundle["mlp_bundle"], sample_df, y_sample, device)

    with open("feature_importance_results.txt", "w") as f:
        f.write("=== CatBoost feature importance (전체) ===\n")
        for name, imp in cb_importance:
            f.write(f"{name}\t{imp:.4f}\n")
        f.write("\n=== MLP 순열 중요도 (전체) ===\n")
        for name, drop in mlp_importance:
            f.write(f"{name}\t{drop:+.4f}\n")
    print("\n결과 저장: feature_importance_results.txt")


if __name__ == "__main__":
    main()
