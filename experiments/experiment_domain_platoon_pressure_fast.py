"""플래툰 스플릿 + 득점권 압박 플래그, CatBoost 전용 빠른 재검증 (2026-08-25 밤).

`experiment_domain_platoon_split.py`/`experiment_domain_pressure_flag.py`가 MLP까지
태워서 각각 1시간 넘게 걸렸던 것(중간에 CPU 경합으로 kill됨)을, Track A 선례(CatBoost엔
도움 MLP엔 손해였던 케이스라 CatBoost 전용 라우팅)를 따라 CatBoost만 단일시드로 빠르게
스크리닝. MLP 적합성은 통과하면 별도로 확인.

기준선 = experiment_final_bundle_baseline.py의 CatBoost 부분: half_2024 769.10 /
expand_2023 822.28 / 평균 795.69.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import (
    process_trackman_features_safe, add_engineered_features,
    causal_smoothed_te_encode, TE_MAIN_AXES, apply_te_residual_features,
)
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
BASELINE = {"half_2024": 769.10, "expand_2023": 822.28}

TE_AXES_PLATOON = [
    ("te_p_cnt", ["pitcher_id", "balls_before", "strikes_before"], "p_main"),
    ("te_p_bhand", ["pitcher_id", "batter_hand"], "p_main"),
    ("te_p_run", ["pitcher_id", "num_runners_on"], "p_main"),
    ("te_p_inn", ["pitcher_id", "inning"], "p_main"),
    ("te_b_cnt", ["batter_id", "balls_before", "strikes_before"], "b_main"),
    ("te_b_phand", ["batter_id", "pitcher_hand"], "b_main"),  # 신규
]


def apply_te_residual_features_platoon(source_df, query_df, prior):
    query_df = query_df.copy()
    mains = {}
    covered_any = np.zeros(len(query_df), dtype=np.int64)
    for name, group_cols in TE_MAIN_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        mains[name] = enc
        covered_any = np.maximum(covered_any, covered)
    for name, group_cols, main_key in TE_AXES_PLATOON:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        query_df[f"{name}_res"] = enc - mains[main_key]
        covered_any = np.maximum(covered_any, covered)
    query_df["te_covered"] = covered_any
    return query_df


def add_pressure_flag(df):
    df = df.copy()
    runner_on_3b = df["base_state"].str[2] == "3"
    pressure = (runner_on_3b & (df["outs_before"] < 2)).astype(np.int64)
    df["scoring_pos_pressure_flag"] = pressure
    df["scoring_pos_pressure_li"] = pressure * df["li"]
    return df


with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=50, verbose=False,
)


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


def fit_and_score(fold_df, tr_final, fold_train_mask, val_mask, y_val, features):
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    if "hand_matchup" in features and "hand_matchup" not in cat_features:
        cat_features = cat_features + ["hand_matchup"]
    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")
    train_pool = Pool(cb_df.loc[fold_train_mask, features], cb_df.loc[fold_train_mask, TARGET], cat_features=cat_features)
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)
    model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    model.fit(train_pool, eval_set=val_pool)
    pred = model.predict_proba(val_pool)[:, 1]
    return score(pred, y_val)


def run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)
    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    te_prior = fold_df.loc[fold_train_mask, TARGET].mean()
    te_source = fold_df.loc[fold_train_mask]

    results = {}

    # baseline: 기존 Track A(5축)
    base_df = apply_te_residual_features(te_source, fold_df.copy(), te_prior)
    features_base = [c for c in base_df.columns if c not in ["row_id", TARGET]]
    t0 = time.time()
    results["baseline"] = fit_and_score(base_df, tr_final, fold_train_mask, val_mask, y_val, features_base)
    print(f"[{fold_name}][baseline] {results['baseline']:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # platoon: Track A 6축(te_b_phand 추가)
    platoon_df = apply_te_residual_features_platoon(te_source, fold_df.copy(), te_prior)
    features_platoon = [c for c in platoon_df.columns if c not in ["row_id", TARGET]]
    t0 = time.time()
    results["platoon"] = fit_and_score(platoon_df, tr_final, fold_train_mask, val_mask, y_val, features_platoon)
    print(f"[{fold_name}][platoon] {results['platoon']:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # pressure: baseline + 압박 플래그 2개
    pressure_df = add_pressure_flag(base_df.copy())
    features_pressure = [c for c in pressure_df.columns if c not in ["row_id", TARGET]]
    t0 = time.time()
    results["pressure"] = fit_and_score(pressure_df, tr_final, fold_train_mask, val_mask, y_val, features_pressure)
    print(f"[{fold_name}][pressure] {results['pressure']:.2f} ({time.time()-t0:.1f}s)", flush=True)

    return results


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    all_results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        all_results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 (CatBoost 전용, 플래툰 스플릿 / 압박 플래그) ===")
    for variant in ["baseline", "platoon", "pressure"]:
        vals = [all_results[f][variant] for f in all_results]
        avg = np.mean(vals)
        line = ", ".join(f"{f}={all_results[f][variant]:.2f}" for f in all_results)
        print(f"{variant}: {line} | 평균={avg:.2f}")


if __name__ == "__main__":
    main()
