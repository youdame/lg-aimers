"""기존 피처(F1 필터) 가중치 부여 실험 (EXPERIMENTS.md §6-17).

지금까지 F1 필터는 2019~2022년 game_type=='F'(퓨처스) 행을 통째로 드롭하는 이진 방식.
이걸 드롭 대신 낮은 sample_weight(F1_WEIGHT)로 남기는 방식으로 바꿔서, CatBoost 단독으로
새 프로토콜(half_2024, expand_2023) 기준선(673.60, `experiment_blend_baseline_new_protocol.py`)과
비교. 완전히 새 피처를 추가하는 게 아니라 기존 데이터 활용 방식만 바꾸는 실험이라
"새 피처 추가 벽" 우회 아이디어(2026-08-23 회의)에 해당.

CatBoost만 테스트(MLP는 가중 손실로 바꾸려면 train_mlp 수정이 필요해서 방향성 확인 후 진행).
"""
import json
import os
import time

import pandas as pd
from catboost import CatBoostClassifier, Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
F1_WEIGHT = 0.1  # 기존: 드롭(weight=0과 동일 효과). 완전 배제 대신 약하게 반영.

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

    is_f1 = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    drop_cols = ["row_id", TARGET]
    features = [c for c in fold_df.columns if c not in drop_cols]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")

    sample_weight = pd.Series(1.0, index=cb_df.index)
    sample_weight[is_f1] = F1_WEIGHT

    train_pool = Pool(
        cb_df.loc[train_mask, features], cb_df.loc[train_mask, TARGET],
        cat_features=cat_features, weight=sample_weight.loc[train_mask].to_numpy(),
    )
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)

    t0 = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    model.fit(train_pool, eval_set=val_pool)
    preds = model.predict_proba(val_pool)[:, 1]
    s = score(preds, y_val)
    print(f"[{fold_name}] soft_f1_weight={F1_WEIGHT}: {s:.2f} ({time.time()-t0:.1f}s)", flush=True)
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
    print(f"\n참고: 기존 F1 하드드롭 CatBoost 기준선 = half_2024 727.51 / expand_2023 619.70 / 평균 673.60")


if __name__ == "__main__":
    main()
