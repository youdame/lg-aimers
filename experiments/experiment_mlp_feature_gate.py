"""MLP 진짜 피처(컬럼) 단위 가중치 실험 (EXPERIMENTS.md §6-17 재해석, 2026-08-24).

CatBoost(트리)는 피처 스케일에 분기가 불변이라 컬럼 가중치가 무의미하지만, MLP는
weight_decay(L2)가 걸려있어서 입력 스케일을 키우면 그 피처를 쓰는 데 필요한 첫
레이어 가중치가 작아도 되니 penalty 부담이 줄어드는 효과가 있음 -> "이 피처를 얼마나
신뢰할지"를 모델이 직접 학습하는 게이트(피처 개수만큼의 학습형 스칼라, 초기값 1.0,
표준화된 수치형 입력에 elementwise 곱) 실험.

게이트 파라미터에는 weight_decay를 안 걸어야 함(안 그러면 그냥 0으로 눌리기만 함) ->
옵티마이저 파라미터 그룹 분리.

단일시드 스크리닝 먼저(팀 컨벤션: 새 아이디어 방향성 확인은 단일시드로, 괜찮으면
7-seed로 확정 검증). CatBoost는 대상 아님(무의미), MLP 단독으로만 비교.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.feature_engineering import process_trackman_features_safe, add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing, to_tensors,
    HIDDEN1, HIDDEN2, DROPOUT, LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
)
from src.local_validation import iter_folds, score, compute_bss

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SCREEN_SEED = 42


class TabularMLPGated(nn.Module):
    def __init__(self, num_numeric_feats, cat_dims, embed_dims=None, use_gate=True):
        super().__init__()
        if embed_dims is None:
            embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Parameter(torch.ones(num_numeric_feats))
        total_input_dim = sum(self.embed_dims) + num_numeric_feats
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, HIDDEN1), nn.BatchNorm1d(HIDDEN1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2), nn.BatchNorm1d(HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_num_in = x_num * self.gate if self.use_gate else x_num
        x_all = torch.cat([x_embed, x_num_in], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def train_gated(X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric_feats, X_val_cat, X_val_num, y_val,
                 embed_dims, use_gate, device, seed=SCREEN_SEED):
    torch.manual_seed(seed)
    model = TabularMLPGated(num_numeric_feats, cat_dims, embed_dims, use_gate=use_gate).to(device)
    criterion = nn.BCELoss()

    if use_gate:
        gate_params = [model.gate]
        other_params = [p for n, p in model.named_parameters() if n != "gate"]
        optimizer = optim.AdamW([
            {"params": other_params, "weight_decay": WEIGHT_DECAY},
            {"params": gate_params, "weight_decay": 0.0},
        ], lr=LR)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loader = DataLoader(TensorDataset(X_tr_cat, X_tr_num, y_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best_state, best_epoch, best_brier, no_improve = None, 0, float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bc, bn, by in loader:
            bc, bn, by = bc.to(device), bn.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bc, bn), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
        brier, _, _ = compute_bss_full(preds, y_val.numpy())
        if brier < best_brier - 1e-9:
            best_brier, best_epoch, no_improve = brier, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def compute_bss_full(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline = r * (1 - r)
    return brier, 1.0 - brier / baseline, max(0, 100000 * (1.0 - brier / baseline))


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


def run_fold(df, df_trm, train_df, train_mask, val_mask, fold_name):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)

    keep_cols = [c for c in fold_df.columns if c != "row_id"]
    mlp_df = fold_df[keep_cols].copy()
    mlp_features = [c for c in mlp_df.columns if c != TARGET]
    num_cols = [c for c in mlp_features if c not in CAT_COLS]

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        mlp_df.loc[fold_train_mask].reset_index(drop=True), CAT_COLS, num_cols)
    val_proc = apply_preprocessing(
        mlp_df.loc[val_mask].reset_index(drop=True), CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    y_val_np = val_proc[TARGET].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    results = {}
    for use_gate, label in [(False, "baseline(게이트 없음)"), (True, "게이트")]:
        t0 = time.time()
        model, best_epoch = train_gated(
            X_tr_cat, X_tr_num, y_tr, cat_dims, len(num_cols), X_val_cat, X_val_num, y_val_t,
            embed_dims, use_gate=use_gate, device=DEVICE,
        )
        with torch.no_grad():
            preds = model(X_val_cat.to(DEVICE), X_val_num.to(DEVICE)).cpu().numpy()
        s = score(preds, y_val_np)
        results[label] = s
        print(f"[{fold_name}] {label}: {s:.2f} (best_epoch={best_epoch}, {time.time()-t0:.1f}s)", flush=True)
        if use_gate:
            gate_vals = model.gate.detach().cpu().numpy()
            order = np.argsort(gate_vals)
            print(f"[{fold_name}] gate 최저 5개: " +
                  ", ".join(f"{num_cols[i]}={gate_vals[i]:.3f}" for i in order[:5]), flush=True)
            print(f"[{fold_name}] gate 최고 5개: " +
                  ", ".join(f"{num_cols[i]}={gate_vals[i]:.3f}" for i in order[-5:]), flush=True)
    return results


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    all_results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        all_results[fold_name] = run_fold(df, df_trm, train_df, train_mask, val_mask, fold_name)

    print("\n=== 요약 ===")
    for label in ["baseline(게이트 없음)", "게이트"]:
        vals = [all_results[f][label] for f in all_results]
        print(f"{label}: " + ", ".join(f"{f}={all_results[f][label]:.2f}" for f in all_results) +
              f" | 평균={np.mean(vals):.2f}")


if __name__ == "__main__":
    main()
