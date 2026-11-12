"""오늘 세션에서 확정한 MLP 개선사항(Optuna 튜닝된 하이퍼파라미터 + quantile PLE 임베딩 +
same_hand/same_hand_advantage)을 다 합친 최종 레시피를 7-seed 앙상블로 재검증.

지금까지의 single-seed 프록시 결과들(튜닝 676.05, quantile +20.50, same_hand +47.90)은
전부 노이즈가 큰 단일 시드 기준이었음 — 진짜 신뢰할 점수는 7-seed 앙상블 평균이라는 게
mlp_model.py 자체의 기존 경험(단일모델 690.62 vs 7-seed 780.92)이라 이번에도 확인 필요.
구종비중+F1 베이스는 그대로, 2023+2024 이중검증.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from src.feature_engineering import process_trackman_features_safe, add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality, fit_quantile_edges,
)

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VAL_SEASONS = [2023, 2024]


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
    train_df = add_engineered_features(train_df, league_success_mean)  # same_hand 포함됨

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
    bin_edges = fit_quantile_edges(X_tr_num)

    return dict(
        X_tr_cat=X_tr_cat, X_tr_num=X_tr_num, y_tr=y_tr,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
        bin_edges=bin_edges,
    )


def main():
    results = {}
    for val_season in VAL_SEASONS:
        print(f"\n{'='*20} 검증 시즌: {val_season} {'='*20}", flush=True)
        t0 = time.time()
        d = build_season_tensors(val_season)
        print(f"[{val_season}] 데이터 준비 완료 ({time.time()-t0:.1f}s)", flush=True)

        members = train_ensemble(
            d["X_tr_cat"], d["X_tr_num"], d["y_tr"], cat_dims=d["cat_dims"],
            num_numeric_feats=d["num_numeric_feats"],
            X_val_cat=d["X_val_cat"], X_val_num=d["X_val_num"], y_val=d["y_val"],
            embed_dims=d["embed_dims"], seeds=ENSEMBLE_SEEDS, device=DEVICE, verbose=False,
            bin_edges=d["bin_edges"],
        )
        preds = predict_ensemble(
            members, d["cat_dims"], d["num_numeric_feats"], d["embed_dims"],
            d["X_val_cat"], d["X_val_num"], device=DEVICE, bin_edges=d["bin_edges"],
        )
        bss = compute_bss(preds, d["y_val"].numpy())
        score = max(0, 100000 * bss)
        results[val_season] = score
        print(f"[{val_season}] 7-seed 앙상블 score={score:.2f} ({time.time()-t0:.1f}s 누적)", flush=True)

    print("\n" + "=" * 60)
    print("최종 요약 (MLP 7-seed 앙상블, 튜닝된 하이퍼파라미터+quantile PLE+same_hand, 구종비중+F1)")
    for s in VAL_SEASONS:
        print(f"  {s} 검증: {results[s]:.2f}")
    avg = np.mean(list(results.values()))
    print(f"  2023+2024 평균: {avg:.2f}")
    print("  참고: 세션 시작 시점 단일 MLP 기준값 806.69(season==2024만, 구종비중+F1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
