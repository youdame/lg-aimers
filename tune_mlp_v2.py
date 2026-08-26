"""MLP 하이퍼파라미터 재탐색 (2026-08-26, same_hand/reverse_rate 반영 후).

기존 MLP 하이퍼파라미터(hidden1=128, hidden2=64, dropout=0.3, lr=0.003,
weight_decay=0.01, batch_size=4096)는 2026-08-16에 튜닝된 값 -- 그 뒤로 시즌진행분/
Track A/reverse_rate/same_hand가 다 추가돼서 피처셋이 많이 바뀜.

트라이얼당 2-seed 평균(순수 단일시드보다 노이즈 적음, 유담님 확정). half_2024 fold만
사용. Optuna study는 SQLite에 저장(중단돼도 이어서 가능).
"""
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch

from feature_engineering import (
    process_trackman_features_safe, add_engineered_features, apply_te_residual_features,
    add_same_hand, SAME_HAND_COLS, TE_RESIDUAL_COLS,
)
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from mlp_model import CAT_COLS, fit_preprocessing, apply_preprocessing, to_tensors, train_mlp, embed_dim_for_cardinality
from local_validation import half_2024_fold, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = int(os.environ.get("TUNE_TRIALS", 20))
TUNE_SEEDS = [42, 123]  # 트라이얼당 2-seed 평균
OUT_PATH = "./model_py311/best_mlp_hparams_v2.json"


def build_data():
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

    all_cols = [c for c in fold_df.columns if c not in ["row_id", TARGET]]
    num_cols = [c for c in all_cols if c not in CAT_COLS and c not in TE_RESIDUAL_COLS]

    train_split = fold_df.loc[fold_train_mask].reset_index(drop=True)
    val_split = fold_df.loc[val_mask].reset_index(drop=True)
    y_val = val_split[TARGET].to_numpy()

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    return X_tr_cat, X_tr_num, y_tr, X_val_cat, X_val_num, y_val_t, y_val, cat_dims, embed_dims, len(num_cols)


def make_objective(data):
    X_tr_cat, X_tr_num, y_tr, X_val_cat, X_val_num, y_val_t, y_val, cat_dims, embed_dims, num_numeric_feats = data

    def objective(trial):
        hidden1 = trial.suggest_categorical("hidden1", [64, 128, 256])
        hidden2 = trial.suggest_categorical("hidden2", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 0.0005, 0.01, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.001, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])

        # TabularMLP가 hidden1/hidden2/dropout을 생성자 인자로 안 받고 모듈 전역변수
        # (HIDDEN1/HIDDEN2/DROPOUT)를 __init__ 실행 시점에 읽어오는 구조라서, 인스턴스
        # 생성 직전에 mlp_model 모듈의 전역변수를 바꿔치기(monkeypatch)한다.
        import mlp_model as mlp_module
        from mlp_model import TabularMLP
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader

        mlp_module.HIDDEN1 = hidden1
        mlp_module.HIDDEN2 = hidden2
        mlp_module.DROPOUT = dropout

        scores = []
        for seed in TUNE_SEEDS:
            torch.manual_seed(seed)
            model = TabularMLP(
                num_numeric_feats=num_numeric_feats, cat_dims=cat_dims, embed_dims=embed_dims,
            ).to(DEVICE)
            criterion = nn.BCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            loader = DataLoader(TensorDataset(X_tr_cat, X_tr_num, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)

            best_score, best_epoch, no_improve = -1, 0, 0
            for epoch in range(1, 31):
                model.train()
                for bc, bn, by in loader:
                    bc, bn, by = bc.to(DEVICE), bn.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(bc, bn), by)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_cat.to(DEVICE), X_val_num.to(DEVICE)).cpu().numpy()
                s = score(val_preds, y_val)
                if s > best_score:
                    best_score, best_epoch, no_improve = s, epoch, 0
                else:
                    no_improve += 1
                    if no_improve >= 5:
                        break
            scores.append(best_score)

        avg_score = float(np.mean(scores))
        trial.set_user_attr("seed_scores", scores)
        return avg_score
    return objective


def main():
    t0 = time.time()
    print("[MLP Tune] half_2024 fold 데이터 준비 중...", flush=True)
    data = build_data()
    print(f"[MLP Tune] 준비 완료 ({time.time()-t0:.1f}s), {N_TRIALS}trial x {len(TUNE_SEEDS)}seed 시작", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_path = "sqlite:///model_py311/mlp_tune_v2_study.db"
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        study_name="mlp_v2", storage=storage_path, load_if_exists=True,
    )
    n_done = len(study.trials)
    if n_done > 0:
        print(f"[체크포인트] {storage_path}에서 완료된 trial {n_done}개 발견, 이어서 진행", flush=True)

    def log_cb(study, trial):
        print(f"[Trial {trial.number:03d}] avg_score={trial.value:.2f} seed_scores={trial.user_attrs.get('seed_scores')} "
              f"params={trial.params}", flush=True)

    remaining = max(N_TRIALS - n_done, 0)
    if remaining > 0:
        study.optimize(make_objective(data), n_trials=remaining, callbacks=[log_cb])

    print("\n" + "=" * 60, flush=True)
    print(f"[Tune 완료] Best score: {study.best_value:.2f} ({time.time()-t0:.1f}s 소요)", flush=True)
    print(f"Best params: {study.best_params}", flush=True)
    print("=" * 60, flush=True)

    baseline_score = None  # 기존 하이퍼파라미터 기준값은 별도로 안 재보고 상대 비교만
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"best_score": study.best_value, "best_params": study.best_params}, f, indent=2, ensure_ascii=False)
    print(f"저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
