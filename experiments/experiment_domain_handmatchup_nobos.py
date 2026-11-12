"""수연님 레포(github.com/lg-first-penguin/sooyun, CatBoost 단독 실전 1007.52)에서
확인한 2개 저비용 후보를 CatBoost 전용으로 빠르게 검증 (2026-08-25 밤).

- hand_matchup: pitcher_hand+batter_hand 조합 범주형 1개 (same_hand 이진 플래그보다
  정교한 버전)
- no_both: pitcher_id/batter_id를 시즌진행분/Track A lookup 키로는 쓰되, 최종
  CatBoost 피처에서는 제외 ("CV 전 구간 일관 개선" 확인됐다고 README에 명시)

Track A가 "CatBoost엔 도움, MLP엔 손해"로 CatBoost 전용 라우팅된 선례를 따라, 이번에도
MLP는 아예 안 태우고 CatBoost만 2-fold로 빠르게 확인 (병목인 MLP 7-seed를 생략하면
fold당 5~10분, 훨씬 빠름). MLP 적합성은 내일 별도 확인.

기준선(오늘 밤 제출 번들의 CatBoost 부분, 단일시드) = half_2024 769.10 / expand_2023
822.28 / 평균 795.69 (experiment_final_bundle_baseline.py 결과, EXPERIMENTS.md 참고).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import process_trackman_features_safe, add_engineered_features, apply_te_residual_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"

BASELINE = {"half_2024": 769.10, "expand_2023": 822.28}

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=50, verbose=False,
)


def add_hand_matchup(df):
    df = df.copy()
    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    return df


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


def run_variant(fold_df, tr_final, fold_train_mask, val_mask, y_val, fold_name, variant_name, drop_ids):
    features = [c for c in fold_df.columns if c not in ["row_id", TARGET]]
    if drop_ids:
        features = [c for c in features if c not in ("pitcher_id", "batter_id")]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    if "hand_matchup" in fold_df.columns and "hand_matchup" not in cat_features:
        cat_features = cat_features + ["hand_matchup"]

    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")
    train_pool = Pool(cb_df.loc[fold_train_mask, features], cb_df.loc[fold_train_mask, TARGET], cat_features=cat_features)
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)

    t0 = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    model.fit(train_pool, eval_set=val_pool)
    pred = model.predict_proba(val_pool)[:, 1]
    s = score(pred, y_val)
    print(f"[{fold_name}][{variant_name}] CatBoost: {s:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return s


def run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)  # 시즌진행분 + reverse_rate

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)
    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    te_prior = fold_df.loc[fold_train_mask, TARGET].mean()
    te_source = fold_df.loc[fold_train_mask]
    fold_df = apply_te_residual_features(te_source, fold_df, te_prior)  # Track A

    fold_df_hm = add_hand_matchup(fold_df)

    results = {}
    results["baseline"] = run_variant(fold_df, tr_final, fold_train_mask, val_mask, y_val, fold_name, "baseline", drop_ids=False)
    results["hand_matchup"] = run_variant(fold_df_hm, tr_final, fold_train_mask, val_mask, y_val, fold_name, "hand_matchup", drop_ids=False)
    results["no_both"] = run_variant(fold_df, tr_final, fold_train_mask, val_mask, y_val, fold_name, "no_both", drop_ids=True)
    results["both"] = run_variant(fold_df_hm, tr_final, fold_train_mask, val_mask, y_val, fold_name, "hand_matchup+no_both", drop_ids=True)
    return results


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    all_results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        all_results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 (CatBoost 전용, hand_matchup / no_both) ===")
    for variant in ["baseline", "hand_matchup", "no_both", "both"]:
        vals = [all_results[f][variant] for f in all_results]
        avg = np.mean(vals)
        line = ", ".join(f"{f}={all_results[f][variant]:.2f}" for f in all_results)
        print(f"{variant}: {line} | 평균={avg:.2f}")


if __name__ == "__main__":
    main()
