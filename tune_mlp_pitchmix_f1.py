"""MLP(TabularMLP) 하이퍼파라미터를 Optuna로 재튜닝.

현재 mlp_model.py의 HIDDEN1/HIDDEN2/DROPOUT/LR/WEIGHT_DECAY/BATCH_SIZE는 구종비중(coarse
pitch-mix) + F1 필터가 생기기 전에 대충 잡힌 값이다. CatBoost가 같은 상황(피처 추가 후
재튜닝)에서 Optuna 15trial로 +13.22를 얻었던 것과 같은 효과를 기대해본다
(tune_xgb_pitchmix.py와 동일한 패턴, experiment_pitchmix_f1_combined.py와 동일하게
2023+2024 이중검증 + 구종비중 리크 방지 + F1 필터 적용).

트라이얼당 비용을 줄이기 위해 앙상블(7-seed) 대신 단일 시드(seed=42)로 학습해 평가한다.
최종 채택 전에는 반드시 best params로 7-seed 앙상블을 재검증할 것 (run_ensemble_check 참고).
"""
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from mlp_model import (
    CAT_COLS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_mlp, embed_dim_for_cardinality, MAX_EPOCHS, PATIENCE,
)

N_TRIALS = int(os.environ.get("TUNE_TRIALS", 20))
DATA_DIR = "./data"
TARGET = "control_success"
OUT_PATH = "./model_py311/best_mlp_pitchmix_f1_hparams.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# SQLite storage로 study를 영구 저장 -> 중단 후 재실행해도 이미 끝난 trial은 유지되고
# 이어서 진행됨 (load_if_exists=True). N_TRIALS는 "이번 실행에서 추가로 돌릴 trial 수".
STUDY_NAME = "mlp_pitchmix_f1"
STORAGE_PATH = "./model_py311/mlp_tune_study.db"
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"
VAL_SEASONS = [2023, 2024]
EVAL_SEED = 42


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def build_season_tensors(val_season):
    """val_season 미만 학습(F1 필터 적용) / val_season 검증 텐서를 미리 만들어둔다.

    전처리(인코더/스케일러 fit)는 하이퍼파라미터와 무관하므로 트라이얼마다 반복하지 않고
    시즌당 한 번만 수행해서 재사용한다.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < val_season
    val_mask = train_df["season"] == val_season
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # 구종비중: val_season 미만 트랙맨만 사용 (리크 방지)
    pivoted = compute_coarse_pitchmix(df[df["season"] < val_season], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    # F1 필터: 2019~2022 game_type=='F' 제거 (학습 구간에서만)
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

    return dict(
        X_tr_cat=X_tr_cat, X_tr_num=X_tr_num, y_tr=y_tr,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
    )


def make_objective(season_data):
    def objective(trial):
        t0 = time.time()
        hidden1 = trial.suggest_categorical("hidden1", [64, 128, 192, 256])
        hidden2 = trial.suggest_categorical("hidden2", [32, 64, 96, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192])

        scores = []
        for val_season in VAL_SEASONS:
            d = season_data[val_season]
            model, best_epoch = train_mlp(
                d["X_tr_cat"], d["X_tr_num"], d["y_tr"], cat_dims=d["cat_dims"],
                num_numeric_feats=d["num_numeric_feats"],
                X_val_cat=d["X_val_cat"], X_val_num=d["X_val_num"], y_val=d["y_val"],
                embed_dims=d["embed_dims"], max_epochs=MAX_EPOCHS, patience=PATIENCE,
                batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                device=DEVICE, verbose=False, seed=EVAL_SEED,
                hidden1=hidden1, hidden2=hidden2, dropout=dropout,
            )
            with torch.no_grad():
                preds = model(d["X_val_cat"].to(DEVICE), d["X_val_num"].to(DEVICE)).cpu().numpy()
            bss = compute_bss(preds, d["y_val"].numpy())
            scores.append(bss)
            trial.set_user_attr(f"best_epoch_{val_season}", best_epoch)

        avg_bss = float(np.mean(scores))
        score = max(0, 100000 * avg_bss)
        state = trial.study.user_attrs.get("running_best", -1)
        if score > state:
            trial.study.set_user_attr("running_best", score)
            marker = " <- 지금까지 최고!"
        else:
            marker = ""
        print(f"[Trial {trial.number:03d}] score={score:.2f} "
              f"(h1={hidden1} h2={hidden2} drop={dropout:.2f} lr={lr:.5f} wd={weight_decay:.5f} bs={batch_size}) "
              f"took={time.time()-t0:.1f}s (지금까지 최고: {max(score, state):.2f}){marker}", flush=True)
        return avg_bss
    return objective


def main():
    print("[Tune] 데이터 로드 및 피처 엔지니어링 (구종비중+F1, 시즌당 1회만 수행)...", flush=True)
    t = time.time()
    season_data = {s: build_season_tensors(s) for s in VAL_SEASONS}
    print(f"[Tune] 준비 완료 ({time.time()-t:.1f}s), device={DEVICE}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE_URL, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    n_done = len(study.trials)
    if n_done:
        print(f"[Tune] 기존 study 재개: 이미 완료된 trial {n_done}개 발견, 이번에 {N_TRIALS}개 추가 진행", flush=True)
    study.optimize(make_objective(season_data), n_trials=N_TRIALS)

    print("\n" + "=" * 60, flush=True)
    print(f"[Tune 완료] Best score(단일시드, 2023+2024 평균): {max(0, 100000*study.best_value):.2f}", flush=True)
    print(f"참고: 튜닝 전 기본값(단일시드 기준 비교 안 해봄, 실전 7-seed 앙상블은 837.48~847.44)")
    print(f"Best params: {study.best_params}", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "best_bss": study.best_value,
            "best_score": max(0, 100000 * study.best_value),
            "best_params": study.best_params,
        }, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
