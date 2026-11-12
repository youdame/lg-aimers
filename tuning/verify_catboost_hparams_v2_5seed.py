"""CatBoost 신규 하이퍼파라미터(tune_catboost_v2.py 1등, +15.15) 5-seed 정식 검증
(2026-08-26 오후).

tune_catboost_v2.py는 half_2024 fold **1개, 단일시드**로만 25trial 탐색해서 뽑은
값이라, 그 fold/시드 조합의 노이즈에 과적합됐을 위험이 있음(MLP 튜닝 때 2-seed 탐색
순위가 7-seed 정식비교에서 뒤집힌 전례가 있어서 동일한 의심 적용). 실전 제출
(submit_0826b.zip, 1092.55)이 로컬 기대치보다 훨씬 작게 나온 뒤 재검증 필요성이
제기됨(본인).

experiment_catboost_multiseed.py와 동일한 패턴으로, 기존 하이퍼파라미터 vs 신규
하이퍼파라미터를 **5-seed 앙상블 기준으로, half_2024/expand_2023 양쪽 fold** 다
비교한다.
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
SEEDS = [42, 123, 7, 2024, 99]

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _old = json.load(f)["best_params"]
with open("./model_py311/best_catboost_hparams_v2.json") as f:
    _new = json.load(f)["best_params"]


def make_params(p):
    return dict(
        depth=p["depth"], learning_rate=p["learning_rate"], l2_leaf_reg=p["l2_leaf_reg"],
        random_strength=p["random_strength"], bagging_temperature=p["bagging_temperature"],
        border_count=p["border_count"], min_data_in_leaf=p["min_data_in_leaf"],
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
        early_stopping_rounds=50, verbose=False,
    )


OLD_PARAMS = make_params(_old)
NEW_PARAMS = make_params(_new)


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


def run_ensemble(name, params, iterations, train_pool, val_pool, y_val):
    preds_list = []
    for seed in SEEDS:
        t0 = time.time()
        model = CatBoostClassifier(**params, iterations=iterations, random_seed=seed)
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict_proba(val_pool)[:, 1]
        preds_list.append(preds)
        s = score(preds, y_val)
        print(f"  [{name}][seed={seed}] 단독={s:.2f} ({time.time()-t0:.1f}s)", flush=True)
    final_score = score(np.mean(preds_list, axis=0), y_val)
    print(f"  [{name}] 5-seed 앙상블 = {final_score:.2f}", flush=True)
    return final_score


def main():
    print("[검증] 데이터 준비 중...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        print(f"\n=== {fold_name} ===", flush=True)
        train_pool, val_pool, y_val = build_data(train_df, df, df_trm, tr_final, train_mask, val_mask)

        old_score = run_ensemble("기존", OLD_PARAMS, 1500, train_pool, val_pool, y_val)
        new_score = run_ensemble("신규", NEW_PARAMS, 734 + 50, train_pool, val_pool, y_val)
        results[fold_name] = (old_score, new_score)
        print(f"[{fold_name}] Δ (신규-기존) = {new_score - old_score:+.2f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("=== 요약 (5-seed 앙상블, 기존 vs 신규 하이퍼파라미터) ===", flush=True)
    deltas = []
    for fold_name, (old_s, new_s) in results.items():
        d = new_s - old_s
        deltas.append(d)
        print(f"{fold_name}: 기존={old_s:.2f} 신규={new_s:.2f} Δ={d:+.2f}", flush=True)
    print(f"평균 Δ = {np.mean(deltas):+.2f}", flush=True)
    if all(d > 0 for d in deltas):
        print("=> 양쪽 fold 다 신규가 이김, 채택 근거 유지", flush=True)
    elif all(d < 0 for d in deltas):
        print("=> 양쪽 fold 다 신규가 짐, 기존으로 되돌리는 걸 고려", flush=True)
    else:
        print("=> fold마다 방향이 다름 (fold 노이즈 가능성) — 신중히 판단 필요", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
