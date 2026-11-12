"""CatBoost loss_function=Logloss vs RMSE(직접 Brier 최적화) 비교 (EXPERIMENTS.md §6-16).

Brier score는 정의상 MSE라서, RMSE(회귀) objective로 학습하면 실제 채점 지표를 직접
최적화하게 됨. 968.15 레시피(구종비중+F1) 기준, 새 검증 프로토콜(local_validation.py)의
필수 2-fold(half_2024, expand_2023)로만 빠르게 비교(사이드 체크, expanding 전체는 생략).

CatBoost는 RMSE loss를 회귀 모델(CatBoostRegressor)로 학습해야 함 -> 예측값이 [0,1]
밖으로 나갈 수 있어 점수 계산 전 clip 필요.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.feature_engineering import process_trackman_features_safe, add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
FOLD_VAL_SEASONS = (2023,)  # 사이드 체크: half_2024 + expand_2023만 (expanding 전체 생략)

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

BASE_PARAMS = dict(
    depth=_tuned["depth"],
    learning_rate=_tuned["learning_rate"],
    l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"],
    bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"],
    min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian",
    random_seed=42,
    verbose=False,
)
ITERATIONS = 1462  # best_iteration(1412) + buffer(50), full_retrain_blend_f1.py와 동일


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    league_success_mean = train_df[TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df, df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    f1_drop = (train_df["game_type"] == "F") & (train_df["season"] <= 2022)
    train_df = train_df[~f1_drop].reset_index(drop=True)
    return train_df, tr_final


def run_fold(train_df, tr_final, train_mask, val_mask, loss):
    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]

    cb_df = train_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")

    X_train, y_train = cb_df.loc[train_mask, features], cb_df.loc[train_mask, TARGET]
    X_val, y_val = cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET]

    if loss == "Logloss":
        model = CatBoostClassifier(**BASE_PARAMS, iterations=ITERATIONS,
                                    loss_function="Logloss", eval_metric="BrierScore")
        model.fit(Pool(X_train, y_train, cat_features=cat_features))
        preds = model.predict_proba(X_val)[:, 1]
    else:  # RMSE
        model = CatBoostRegressor(**BASE_PARAMS, iterations=ITERATIONS,
                                   loss_function="RMSE", eval_metric="RMSE")
        model.fit(Pool(X_train, y_train, cat_features=cat_features))
        preds = np.clip(model.predict(X_val), 0.0, 1.0)

    return score(preds, y_val.to_numpy())


def main():
    train_df, tr_final = build_base_df()
    for loss in ["Logloss", "RMSE"]:
        fold_scores = {}
        for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=FOLD_VAL_SEASONS):
            t0 = time.time()
            s = run_fold(train_df, tr_final, train_mask, val_mask, loss)
            fold_scores[fold_name] = s
            print(f"[{loss}] {fold_name}: {s:.2f} ({time.time()-t0:.1f}s)", flush=True)
        avg = sum(fold_scores.values()) / len(fold_scores)
        print(f"[{loss}] 평균: {avg:.2f}\n", flush=True)


if __name__ == "__main__":
    main()
