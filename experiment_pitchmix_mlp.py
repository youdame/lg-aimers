"""구종비중x카운트 피처(situational_pitchmix_*)가 MLP에도 도움되는지 로컬에서 확인.
mlp_model.py는 클론해둔 lgaimers9-ensemble-v1 저장소 걸 그대로 재사용 (창현님 검증된 구조, 7-seed 그대로).
get_device()가 CUDA만 체크해서 Mac에서는 CPU로 도니, MPS(Apple GPU)를 직접 지정해서 속도를 높인다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SCRATCH_REPO = "/private/tmp/claude-501/-Users-joyudam-GitHub-open/9e6ee3c9-6c84-4f74-9e6f-fb99658907c4/scratchpad/lgaimers9-ensemble-v1"
# 'code'라는 이름이 파이썬 내장 모듈(code.py, 인터랙티브 콘솔용)이랑 겹쳐서 'code.mlp_model'로는 못 불러옴.
# code/ 디렉터리 자체를 경로에 추가해서 mlp_model만 직접 import.
sys.path.insert(0, os.path.join(SCRATCH_REPO, "code"))
from mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality, compute_bss,
)

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_by_count import compute_situational_pitchmix, add_pitchmix_by_situation

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def run_mlp(train_split, val_split, num_cols, label):
    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t, embed_dims=embed_dims,
        seeds=ENSEMBLE_SEEDS, device=DEVICE, verbose=False,
    )
    mlp_pred = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, device=DEVICE)
    _, _, score = compute_bss(mlp_pred, val_split[TARGET].values)
    print(f"[{label}] {len(num_cols)+len(CAT_COLS)}개 피처, {len(ENSEMBLE_SEEDS)}-seed ({DEVICE}): score={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return score


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_situational_pitchmix(df, df_trm, match_cols, is_train_split=True)
    train_df_pm, new_cols = add_pitchmix_by_situation(train_df, pivoted, match_cols)
    print(f"새 피처: {new_cols}", flush=True)

    for base_df, label in [(train_df, "baseline (123 features)"), (train_df_pm, "baseline + pitchmix (127 features)")]:
        keep_cols = [c for c in base_df.columns if c != "row_id"]
        split_df = base_df[keep_cols].copy()
        features = [c for c in split_df.columns if c != TARGET]
        num_cols = [c for c in features if c not in CAT_COLS]

        train_split = split_df.loc[train_mask].reset_index(drop=True)
        val_split = split_df.loc[val_mask].reset_index(drop=True)
        run_mlp(train_split, val_split, num_cols, label)


if __name__ == "__main__":
    main()
