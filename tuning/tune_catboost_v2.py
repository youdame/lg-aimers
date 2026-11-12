"""CatBoost 하이퍼파라미터 재탐색 (2026-08-25, 시즌진행분+Track A 반영 후).

기존 하이퍼파라미터(model_py311/best_catboost_pitchmix_hparams.json)는 구종비중만
있던 시절(시즌진행분/Track A 이전)에 튜닝된 값 -- 피처셋이 크게 바뀌었으니 최적값도
바뀔 수 있음(창현님 쪽도 같은 논리로 재탐색 중, EXPERIMENTS.md 참고).

half_2024 fold(실전과 제일 유사)만 사용, N_TRIALS 축소(15~20, 원래 40회보다 줄임),
Optuna TPE + early stopping.
"""
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import process_trackman_features_safe, add_engineered_features, apply_te_residual_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
N_TRIALS = int(os.environ.get("TUNE_TRIALS", 18))
OUT_PATH = "./model_py311/best_catboost_hparams_v2.json"


def build_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask, val_mask = next(iter_folds(train_df, val_seasons=(), include_half_2024=True))[1:]
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

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_split.columns if c not in drop_cols]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for c in cat_features:
        train_split[c] = train_split[c].astype(str).fillna("missing")
        val_split[c] = val_split[c].astype(str).fillna("missing")

    train_pool = Pool(train_split[features], train_split[TARGET], cat_features=cat_features)
    val_pool = Pool(val_split[features], val_split[TARGET], cat_features=cat_features)
    y_val = val_split[TARGET].to_numpy()
    return train_pool, val_pool, y_val


def make_objective(train_pool, val_pool, y_val):
    def objective(trial):
        params = dict(
            iterations=1500,
            loss_function="Logloss",
            eval_metric="BrierScore",
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            border_count=trial.suggest_int("border_count", 32, 254),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 200, log=True),
            bootstrap_type="Bayesian",
        )
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        preds = model.predict_proba(val_pool)[:, 1]
        s = score(preds, y_val)
        trial.set_user_attr("best_iteration", model.best_iteration_)
        trial.set_user_attr("score", s)
        return s
    return objective


def main():
    print(f"[Tune] half_2024 fold 데이터 준비 중...", flush=True)
    train_pool, val_pool, y_val = build_data()
    print(f"[Tune] train={train_pool.num_row()}행, val={val_pool.num_row()}행, {N_TRIALS}trial 시작", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_path = "sqlite:///model_py311/catboost_tune_v2_study.db"
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        study_name="catboost_v2", storage=storage_path, load_if_exists=True,
    )
    n_done = len(study.trials)
    if n_done > 0:
        print(f"[체크포인트] {storage_path}에서 완료된 trial {n_done}개 발견, 이어서 진행", flush=True)

    def log_cb(study, trial):
        print(f"[Trial {trial.number:03d}] score={trial.value:.2f} "
              f"(best_iter={trial.user_attrs.get('best_iteration')}) params={trial.params}", flush=True)

    t0 = time.time()
    study.optimize(make_objective(train_pool, val_pool, y_val), n_trials=N_TRIALS, callbacks=[log_cb])

    print("\n" + "=" * 60, flush=True)
    print(f"[Tune 완료] Best score: {study.best_value:.2f} ({time.time()-t0:.1f}s 소요)", flush=True)
    print(f"Best params: {study.best_params}", flush=True)
    print(f"Best iteration: {study.best_trial.user_attrs.get('best_iteration')}", flush=True)
    print("=" * 60, flush=True)

    baseline_score = 769.10  # half_2024, 기존 하이퍼파라미터(시즌진행분+reverse_rate+TrackA 적용) 기준, 2026-08-26 갱신
    print(f"참고: 기존 하이퍼파라미터 half_2024 점수 = {baseline_score:.2f}", flush=True)
    print(f"Δ = {study.best_value - baseline_score:+.2f}", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "best_score": study.best_value,
            "best_params": study.best_params,
            "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        }, f, indent=2, ensure_ascii=False)
    print(f"저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
