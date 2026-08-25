"""CatBoost 멀티시드 앙상블 (수연님 실측 근거: CatBoost 단독 5-seed 앙상블로 1007.52).
EXPERIMENTS.md §7 후속.

우리는 지금까지 CatBoost를 시드 1개로만 썼음(MLP는 7-seed인데 비대칭). CatBoost 멀티
스레드 히스토그램 계산이 완전히 결정론적이지 않다는 게 이미 확인된 상태라(창현님
저장소, ~7점 차), 시드 여러 개 평균 내면 이 노이즈를 줄여 안정적 이득이 있을 수 있음.

half_2024 fold에서 +4.39 확인(744.52->770.69) 후 이미 프로덕션 코드(`full_retrain_
blend_f1.py`)에 반영했었음 — 재학습 전에 두 번째 fold(expand_2023)에서도 재현되는지
뒤늦게 확인(2026-08-25). CatBoost 전용이라 MLP 없이도 빠름(fold당 5seed ~35~40분).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features, apply_te_residual_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
SEEDS = [42, 123, 7, 2024, 99]

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

BASE_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    early_stopping_rounds=50, verbose=False,
)


def build_data(train_df, df, df_trm, tr_final, train_mask, val_mask):
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


def run_fold(fold_name, train_pool, val_pool, y_val):
    """개별 시드 vs 앙상블 비교는 이 함수 안에서 자기완결적으로 이뤄짐(외부 기준선을
    끌어오지 않음) — half_2024 최초 검증 때도 실제로는 이 방식(개별 최고 시드 대비
    앙상블 delta)으로 채택 판단했고, 스크립트에 있던 외부 하드코딩 기준선(744.52)은
    당시 Track A가 빠진 값이라 비교 대상이 달라 애초에 무의미했음."""
    preds_list = []
    seed_scores = {}
    for seed in SEEDS:
        t0 = time.time()
        model = CatBoostClassifier(**BASE_PARAMS, iterations=1500, random_seed=seed)
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict_proba(val_pool)[:, 1]
        preds_list.append(preds)
        s = score(preds, y_val)
        seed_scores[seed] = s
        cum_avg = np.mean(preds_list, axis=0)
        cum_score = score(cum_avg, y_val)
        print(f"[{fold_name}][seed={seed}] 단독={s:.2f} | 누적앙상블({len(preds_list)}개)={cum_score:.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    final_ens = np.mean(preds_list, axis=0)
    final_score = score(final_ens, y_val)
    best_individual = max(seed_scores.values())
    print(f"[{fold_name}] === {len(SEEDS)}-seed 앙상블: {final_score:.2f} ===", flush=True)
    print(f"[{fold_name}] 개별 시드 중 최고: {best_individual:.2f} | 앙상블 대비 Δ = {final_score - best_individual:+.2f}", flush=True)
    return final_score


def main():
    print("[멀티시드] 데이터 준비 중...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        train_pool, val_pool, y_val = build_data(train_df, df, df_trm, tr_final, train_mask, val_mask)
        results[fold_name] = run_fold(fold_name, train_pool, val_pool, y_val)

    print("\n=== 요약 (시즌진행분+Track A 기준, CatBoost 5-seed 앙상블, reverse_rate 미포함) ===", flush=True)
    for fold_name, s in results.items():
        print(f"{fold_name}: {s:.2f}", flush=True)
    print(f"평균: {np.mean(list(results.values())):.2f}", flush=True)


if __name__ == "__main__":
    main()
