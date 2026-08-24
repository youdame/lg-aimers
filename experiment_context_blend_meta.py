"""상황 기반 동적 블렌드 메타모델 (EXPERIMENTS.md §6-18).

지금까지 스태킹은 [cat_pred, mlp_pred] 두 숫자만 메타모델 입력으로 썼음(로지스틱회귀든
뭐든 동일, 창현님 검증 +14.22). 여기서는 볼카운트/손 등 컨텍스트 피처까지 같이 넣어서
"행마다 다른 블렌드 규칙"을 학습하는 CatBoost 메타모델을 시도 — 창현님이 안 해본 방향.

cache_fold_predictions.py가 저장해둔 fold별 cat_pred/mlp_pred/context를 그대로 사용
(재학습 없음). 각 fold의 val set을 meta_train/meta_test로 랜덤 분할(70/30, seed 고정)
해서, base 모델이 전혀 보지 않은 meta_test에서만 최종 비교 — base 모델 학습에 전혀
관여 안 하므로 §6-11에서 실패했던 "여러 시즌 OOF 스태킹 x F1 필터 충돌"과는 무관한
구조(여긴 시즌 경계를 넘나드는 재학습이 아니라, 이미 고정된 fold의 val set 내부 분할).

기존 alpha=0.61 고정 블렌드도 같은 meta_test 부분집합으로 재채점해서 공정 비교.
"""
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from local_validation import score

FOLD_NAMES = ["half_2024", "expand_2023"]
BLEND_ALPHA = 0.61
META_TEST_FRAC = 0.3
SPLIT_SEED = 42

CATEGORICAL_CONTEXT = ["pitcher_hand", "batter_hand"]

META_PARAMS = dict(
    depth=4, learning_rate=0.05, iterations=500, l2_leaf_reg=5.0,
    loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=30, verbose=False,
)


def load_fold(fold_name):
    with open(f"./model_py311/fold_predictions_{fold_name}.pkl", "rb") as f:
        return pickle.load(f)


def run_fold(fold_name):
    d = load_fold(fold_name)
    cat_pred, mlp_pred, y_val, context_df = d["cat_pred"], d["mlp_pred"], d["y_val"], d["context_df"]
    n = len(y_val)
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(idx, test_size=META_TEST_FRAC, random_state=SPLIT_SEED)

    # 고정 alpha 블렌드를 meta_test 부분집합에서 재채점 (공정 비교 기준)
    fixed_blend = BLEND_ALPHA * cat_pred + (1 - BLEND_ALPHA) * mlp_pred
    fixed_score = score(fixed_blend[idx_test], y_val[idx_test])

    meta_df = context_df.copy()
    meta_df["cat_pred"] = cat_pred
    meta_df["mlp_pred"] = mlp_pred
    meta_df["pred_diff"] = cat_pred - mlp_pred
    for c in CATEGORICAL_CONTEXT:
        meta_df[c] = meta_df[c].astype(str)

    meta_features = [c for c in meta_df.columns]
    train_pool = Pool(meta_df.iloc[idx_train][meta_features], y_val[idx_train], cat_features=CATEGORICAL_CONTEXT)
    test_pool = Pool(meta_df.iloc[idx_test][meta_features], y_val[idx_test], cat_features=CATEGORICAL_CONTEXT)

    meta_model = CatBoostClassifier(**META_PARAMS)
    meta_model.fit(train_pool, eval_set=test_pool)
    meta_pred = meta_model.predict_proba(test_pool)[:, 1]
    meta_score = score(meta_pred, y_val[idx_test])

    print(f"[{fold_name}] 고정 alpha=0.61 (meta_test에서 재채점): {fixed_score:.2f}", flush=True)
    print(f"[{fold_name}] 컨텍스트 메타모델(CatBoost): {meta_score:.2f}", flush=True)

    importances = dict(zip(meta_features, meta_model.get_feature_importance(train_pool)))
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:6]
    print(f"[{fold_name}] 메타모델 피처 중요도 top6: {top}", flush=True)

    return fixed_score, meta_score


def main():
    fixed_scores, meta_scores = [], []
    for fold_name in FOLD_NAMES:
        fixed_s, meta_s = run_fold(fold_name)
        fixed_scores.append(fixed_s)
        meta_scores.append(meta_s)

    print("\n=== 요약 (meta_test 부분집합 기준) ===")
    print(f"고정 alpha=0.61: " + ", ".join(f"{n}={s:.2f}" for n, s in zip(FOLD_NAMES, fixed_scores)) +
          f" | 평균={np.mean(fixed_scores):.2f}")
    print(f"컨텍스트 메타모델: " + ", ".join(f"{n}={s:.2f}" for n, s in zip(FOLD_NAMES, meta_scores)) +
          f" | 평균={np.mean(meta_scores):.2f}")


if __name__ == "__main__":
    main()
