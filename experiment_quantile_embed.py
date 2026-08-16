"""MLP 수치형 피처 전처리: StandardScaler 단순 concat vs quantile 기반 PLE 임베딩 비교.

창현님 저장소(lgaimers9-ensemble-v1)에서 실전 리더보드로 검증된 아이디어(+55.78,
n_bins=24/d_embed=8)를 우리 mlp_model.py에 이식한 뒤, 우리 레시피(구종비중+F1)
기준으로 2023+2024 이중검증(단일시드 프록시, tune_mlp_pitchmix_f1.py와 동일 패턴)으로
재확인한다.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from mlp_model import (
    CAT_COLS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_mlp, embed_dim_for_cardinality, fit_quantile_edges,
    MAX_EPOCHS, PATIENCE, QUANTILE_N_BINS, QUANTILE_D,
)

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VAL_SEASONS = [2023, 2024]
EVAL_SEED = 42


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def build_season_tensors(val_season):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < val_season
    val_mask = train_df["season"] == val_season
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[df["season"] < val_season], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_mask = train_mask & (~f1_drop)

    keep_cols = [c for c in train_df.columns if c != "row_id"]
    mlp_df = train_df[keep_cols].copy()
    mlp_features = [c for c in mlp_df.columns if c != TARGET]
    num_cols = [c for c in mlp_features if c not in CAT_COLS]

    train_split = mlp_df.loc[train_mask].reset_index(drop=True)
    val_split = mlp_df.loc[val_mask].reset_index(drop=True)

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, y_val = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    return dict(
        X_tr_cat=X_tr_cat, X_tr_num=X_tr_num, y_tr=y_tr,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
        bin_edges=bin_edges,
    )


def run_one(d, use_quantile, label):
    t0 = time.time()
    model, best_epoch = train_mlp(
        d["X_tr_cat"], d["X_tr_num"], d["y_tr"], cat_dims=d["cat_dims"],
        num_numeric_feats=d["num_numeric_feats"],
        X_val_cat=d["X_val_cat"], X_val_num=d["X_val_num"], y_val=d["y_val"],
        embed_dims=d["embed_dims"], max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=DEVICE, verbose=False, seed=EVAL_SEED,
        bin_edges=d["bin_edges"] if use_quantile else None, quantile_d=QUANTILE_D,
    )
    with torch.no_grad():
        preds = model(d["X_val_cat"].to(DEVICE), d["X_val_num"].to(DEVICE)).cpu().numpy()
    bss = compute_bss(preds, d["y_val"].numpy())
    score = max(0, 100000 * bss)
    print(f"[{label}] score={score:.2f} (best_epoch={best_epoch}, {time.time()-t0:.1f}s)", flush=True)
    return score


def main():
    results = {}
    for val_season in VAL_SEASONS:
        print(f"\n{'='*20} 검증 시즌: {val_season} {'='*20}", flush=True)
        d = build_season_tensors(val_season)
        s_before = run_one(d, use_quantile=False, label=f"{val_season} StandardScaler(기존)")
        s_after = run_one(d, use_quantile=True, label=f"{val_season} Quantile PLE(n_bins={QUANTILE_N_BINS}, d={QUANTILE_D})")
        results[val_season] = (s_before, s_after)

    print("\n" + "=" * 60)
    print(f"요약 (MLP 단일시드, 구종비중+F1 베이스, 2023+2024 평균)")
    avg_before = np.mean([results[s][0] for s in VAL_SEASONS])
    avg_after = np.mean([results[s][1] for s in VAL_SEASONS])
    for s in VAL_SEASONS:
        print(f"  {s} 검증 | StandardScaler: {results[s][0]:.2f} | Quantile PLE: {results[s][1]:.2f}")
    print(f"  평균      | StandardScaler: {avg_before:.2f} | Quantile PLE: {avg_after:.2f} ({avg_after-avg_before:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
