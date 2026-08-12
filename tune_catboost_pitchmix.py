"""coarse 구종비중 피처(127개 피처)를 포함한 상태로 CatBoost를 다시 Optuna 튜닝.
기존 하이퍼파라미터(818.54를 냈던 값)는 피처 추가 전 기준이라, 새 피처 포함 시 재튜닝하면
더 오를 가능성이 있음. 시간 절약을 위해 trial 수를 줄임(기본 15).
"""
import os
import time

import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS

N_TRIALS = int(os.environ.get("TUNE_TRIALS", 15))
DATA_DIR = "./data"
TARGET = "control_success"
OUT_PATH = "./model_py311/best_catboost_pitchmix_hparams.json"


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def build_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df, df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    train_df = pd.merge(train_df, pivoted, on=COARSE_COLS, how="left")
    train_df[new_cols] = train_df[new_cols].fillna(train_df[new_cols].mean())

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna("missing")

    train_split = train_df.loc[train_mask].reset_index(drop=True)
    val_split = train_df.loc[val_mask].reset_index(drop=True)
    return train_split, val_split, features, categorical_features


def make_objective(train_split, val_split, features, categorical_features):
    train_pool = Pool(train_split[features], train_split[TARGET], cat_features=categorical_features)
    val_pool = Pool(val_split[features], val_split[TARGET], cat_features=categorical_features)

    def objective(trial):
        t0 = time.time()
        params = dict(
            iterations=1500,
            depth=trial.suggest_int("depth", 5, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.5, 8.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
            border_count=trial.suggest_int("border_count", 64, 200),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 150, log=True),
            bootstrap_type="Bayesian",
            loss_function="Logloss",
            eval_metric="BrierScore",
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
        )
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict_proba(val_pool)[:, 1]
        bss = compute_bss(preds, val_split[TARGET].values)
        score = max(0, 100000 * bss)
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        print(f"[Trial {trial.number:03d}] score={score:.2f} best_iter={model.get_best_iteration()} "
              f"took={time.time()-t0:.1f}s params={trial.params}", flush=True)
        return bss
    return objective


def main():
    print("[Tune-CatBoost+pitchmix] 데이터 준비 중...", flush=True)
    t = time.time()
    train_split, val_split, features, categorical_features = build_data()
    print(f"train={len(train_split)} val={len(val_split)} features={len(features)} ({time.time()-t:.1f}s)", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(train_split, val_split, features, categorical_features), n_trials=N_TRIALS)

    print("\n" + "=" * 60, flush=True)
    print(f"[완료] Best score: {max(0, 100000*study.best_value):.2f}", flush=True)
    print(f"Best params: {study.best_params}", flush=True)
    print(f"참고: 기본 하이퍼파라미터(피처 추가 전 튜닝값) + coarse pitchmix = 817.19", flush=True)
    print("=" * 60, flush=True)

    import json
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "best_score": max(0, 100000 * study.best_value),
            "best_params": study.best_params,
            "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        }, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
