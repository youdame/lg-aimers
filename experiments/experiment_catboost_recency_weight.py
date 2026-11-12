"""기존 피처 가중치 부여 실험, 2번째 시도: 시즌 recency 가중치 (EXPERIMENTS.md §6-17).

F1 소프트가중치는 실패(측정 체계가 바뀐 데이터라 부분포함 자체가 나쁨) -> F1 필터는
그대로 하드드롭 유지. 대신 그 위에서, 남은 학습 데이터에 "얼마나 최신 시즌인가"로
지수적 recency 가중치를 추가로 적용. 이건 분포가 바뀐 데이터를 섞는 게 아니라 정상
데이터 내에서 최신 트렌드 비중을 높이는 것이라 F1 문제와 성격이 다름.

weight = RECENCY_DECAY ** (fold의 마지막 학습 시즌 - season), F1 드롭된 행 제외.
CatBoost 단독, 새 프로토콜(half_2024, expand_2023) 기준선(673.60)과 비교.
"""
import json
import os
import time

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import process_trackman_features_safe, add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
RECENCY_DECAY = 0.85  # 시즌 1개 멀어질 때마다 가중치 0.85배

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"],
    learning_rate=_tuned["learning_rate"],
    l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"],
    bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"],
    min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian",
    loss_function="Logloss",
    eval_metric="BrierScore",
    random_seed=42,
    early_stopping_rounds=50,
    verbose=False,
)


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


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

    drop_cols = ["row_id", TARGET]
    features = [c for c in fold_df.columns if c not in drop_cols]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")

    # recency 가중치: 이 fold의 학습 구간에서 가장 최신 시즌 기준 지수 감쇠
    max_train_season = fold_df.loc[fold_train_mask, "season"].max()
    seasons_ago = max_train_season - fold_df["season"]
    recency_weight = RECENCY_DECAY ** seasons_ago.clip(lower=0)

    train_pool = Pool(
        cb_df.loc[fold_train_mask, features], cb_df.loc[fold_train_mask, TARGET],
        cat_features=cat_features, weight=recency_weight.loc[fold_train_mask].to_numpy(),
    )
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)

    t0 = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    model.fit(train_pool, eval_set=val_pool)
    preds = model.predict_proba(val_pool)[:, 1]
    s = score(preds, y_val)
    print(f"[{fold_name}] recency_decay={RECENCY_DECAY}: {s:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return s


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    fold_scores = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        fold_scores[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    avg = sum(fold_scores.values()) / len(fold_scores)
    print("\n=== 요약 ===")
    for name, s in fold_scores.items():
        print(f"  {name}: {s:.2f}")
    print(f"  평균: {avg:.2f}")
    print(f"\n참고: 기존(가중치 없음) CatBoost 기준선 = half_2024 727.51 / expand_2023 619.70 / 평균 673.60")


if __name__ == "__main__":
    main()
