"""Factorization Machine(FM) 새 모델 타입 시도 (EXPERIMENTS.md §7 후속).

CatBoost/MLP/EBM(GAM) 다음으로, CTR 예측(광고 클릭률)에서 쓰이는 FM 계열 시도.
트리도 GAM도 일반 MLP도 아닌, "모든 피처 쌍의 임베딩 내적"으로 상호작용을 모델링하는
완전히 다른 수학 -- 범주형 많고 타겟이 희박한 문제 구조가 CTR 예측과 비슷해서 선택.

범주형 7개: 표준 임베딩. 수치형 ~130개: 피처마다 고유 k차원 임베딩 벡터를 만들고
그 행의 (표준화된) 값을 곱해서 사용(DeepFM 스타일). 1차항(선형) + 2차항(모든 필드
쌍 내적 합, FM 표준 트릭으로 O(n*k)에 계산) + sigmoid.

단일시드 스크리닝. EBM 실험과 동일하게 half_2024/expand_2023, CatBoost와 동일 피처셋
(Track A 포함) 기준 FM 단독 성능 + CatBoost/MLP와의 상관계수만 확인 (블렌드까지는
승산 있을 때만 진행).
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
from catboost import CatBoostClassifier, Pool

from feature_engineering import (
    process_trackman_features_safe, add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS,
)
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from mlp_model import CAT_COLS, fit_preprocessing, apply_preprocessing, to_tensors, embed_dim_for_cardinality
from local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
FM_K = 8  # 필드 임베딩 차원
LR = 0.003
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4096
MAX_EPOCHS = 60
PATIENCE = 7
SEED = 42

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=50, verbose=False,
)


class FactorizationMachine(nn.Module):
    """1차항(선형) + 2차항(모든 필드 쌍 임베딩 내적 합, FM 표준 O(n*k) 트릭)."""

    def __init__(self, num_numeric_feats, cat_dims, k=FM_K):
        super().__init__()
        n_cat = len(cat_dims)
        self.n_numeric = num_numeric_feats
        # 임베딩(2차항용): 범주형은 값별 벡터, 수치형은 피처별 고유 벡터(값을 곱해서 사용)
        self.cat_embed_v = nn.ModuleList([nn.Embedding(dim + 2, k) for dim in cat_dims])
        self.num_embed_v = nn.Parameter(torch.randn(num_numeric_feats, k) * 0.01)
        # 선형항(1차): 범주형은 값별 가중치, 수치형은 피처별 가중치
        self.cat_linear = nn.ModuleList([nn.Embedding(dim + 2, 1) for dim in cat_dims])
        self.num_linear = nn.Parameter(torch.randn(num_numeric_feats) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        batch = x_num.shape[0]
        x_cat_long = x_cat.long()

        # 1차항
        linear = self.bias.expand(batch)
        for i, emb in enumerate(self.cat_linear):
            linear = linear + emb(x_cat_long[:, i]).squeeze(-1)
        linear = linear + (x_num * self.num_linear).sum(dim=1)

        # 2차항: 필드 임베딩 벡터 모으기 (batch, n_fields, k)
        cat_vecs = [emb(x_cat_long[:, i]) for i, emb in enumerate(self.cat_embed_v)]  # 각 (batch, k)
        num_vecs = x_num.unsqueeze(-1) * self.num_embed_v.unsqueeze(0)  # (batch, n_numeric, k)
        all_vecs = torch.cat([torch.stack(cat_vecs, dim=1), num_vecs], dim=1)  # (batch, n_fields, k)

        sum_sq = all_vecs.sum(dim=1) ** 2  # (batch, k)
        sq_sum = (all_vecs ** 2).sum(dim=1)  # (batch, k)
        interaction = 0.5 * (sum_sq - sq_sum).sum(dim=1)  # (batch,)

        return self.sigmoid(linear + interaction)


def train_fm(X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric_feats, X_val_cat, X_val_num, y_val, device, seed=SEED):
    torch.manual_seed(seed)
    model = FactorizationMachine(num_numeric_feats, cat_dims).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
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
        brier = ((preds - y_val.numpy()) ** 2).mean()
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

    # --- CatBoost (기준선) ---
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

    # --- FM (단일시드, CatBoost와 동일 피처셋: CAT_COLS 임베딩 + 나머지 수치형 전부) ---
    fm_features = [c for c in features if c not in CAT_COLS]  # 수치형 대상(Track A 포함)
    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, fm_features)
    val_proc = apply_preprocessing(val_split, CAT_COLS, fm_features, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, fm_features, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, fm_features, TARGET)

    fm_model, best_epoch = train_fm(
        X_tr_cat, X_tr_num, y_tr, cat_dims, len(fm_features), X_val_cat, X_val_num, y_val_t, DEVICE,
    )
    with torch.no_grad():
        fm_pred = fm_model(X_val_cat.to(DEVICE), X_val_num.to(DEVICE)).cpu().numpy()
    fm_score = score(fm_pred, y_val)
    print(f"[{fold_name}] FM(단일시드): {fm_score:.2f} (best_epoch={best_epoch}, {time.time()-t0:.1f}s)", flush=True)

    corr = np.corrcoef(cat_pred, fm_pred)[0, 1]
    print(f"[{fold_name}] 상관계수 FM-CatBoost: {corr:.4f}", flush=True)

    return dict(cat=cat_score, fm=fm_score, corr=corr)


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 ===")
    for key in ["cat", "fm"]:
        vals = [results[f][key] for f in results]
        print(f"{key}: " + ", ".join(f"{f}={results[f][key]:.2f}" for f in results) + f" | 평균={np.mean(vals):.2f}")
    print("상관계수(FM-CatBoost): " + ", ".join(f"{f}={results[f]['corr']:.4f}" for f in results))


if __name__ == "__main__":
    main()
