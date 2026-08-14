"""tune_xgb.py에 구종비중(coarse pitch-mix, 리크 고친 버전) 피처를 반영해서
Optuna로 재튜닝. 기존 XGBoost 하이퍼파라미터는 구종비중이 생기기 전에 튜닝된
것이라, CatBoost가 재튜닝으로 +13.22 얻었던 것과 같은 효과를 기대해본다.
"""
import os
import time

import optuna
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

N_TRIALS = int(os.environ.get("TUNE_TRIALS", 15))
DATA_DIR = "./data"
TARGET = "control_success"
OUT_PATH = "./model_py311/best_xgb_pitchmix_hparams.json"


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def build_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024

    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # 리크 수정: season < 2024만 넘겨서 진짜 홀드아웃으로 구종비중 집계
    pivoted = compute_coarse_pitchmix(df[df["season"] < 2024], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    drop_cols = ["row_id", TARGET]
    features = [col for col in train_df.columns if col not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna("missing").astype("category")

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET]
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET]
    print(f"피처 개수: {len(features)}개 (구종비중 4개 포함)", flush=True)
    return X_train, y_train, X_val, y_val


def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        t0 = time.time()
        params = dict(
            n_estimators=2500,
            max_depth=trial.suggest_int("max_depth", 4, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.008, 0.05, log=True),
            subsample=trial.suggest_float("subsample", 0.55, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.02, 5.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.3, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 5, 40, log=True),
            gamma=trial.suggest_float("gamma", 5e-4, 0.05, log=True),
            tree_method="hist",
            enable_categorical=True,
            eval_metric=brier_eval,
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        bss = compute_bss(preds, y_val)
        trial.set_user_attr("best_iteration", model.best_iteration)
        score = max(0, 100000 * bss)
        state = trial.study.user_attrs.get("running_best", -1)
        if score > state:
            trial.study.set_user_attr("running_best", score)
            marker = " <- 지금까지 최고!"
        else:
            marker = ""
        print(f"[Trial {trial.number:03d}] score={score:.2f} best_iter={model.best_iteration} "
              f"took={time.time()-t0:.1f}s (지금까지 최고: {max(score, state):.2f}){marker}", flush=True)
        return bss
    return objective


def main():
    print("[Tune] 데이터 로드 및 피처 엔지니어링 (구종비중 포함, 1회만 수행)...", flush=True)
    t = time.time()
    X_train, y_train, X_val, y_val = build_data()
    print(f"[Tune] train: {len(X_train)}행 | val: {len(X_val)}행 ({time.time()-t:.1f}s)", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(X_train, y_train, X_val, y_val), n_trials=N_TRIALS)

    print("\n" + "=" * 60, flush=True)
    print(f"[Tune 완료] Best score: {max(0, 100000*study.best_value):.2f}", flush=True)
    print(f"참고: 구종비중 반영 튜닝 전 = 790.90")
    print(f"Best params: {study.best_params}", flush=True)
    print("=" * 60, flush=True)

    import json
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "best_bss": study.best_value,
            "best_score": max(0, 100000 * study.best_value),
            "best_params": study.best_params,
            "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        }, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
