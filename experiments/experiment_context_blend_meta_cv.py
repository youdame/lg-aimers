"""컨텍스트 메타모델 견고성 재확인: val set 내부 5-fold CV (EXPERIMENTS.md §6-18 후속).

이전 실험(랜덤 70/30 1회 분할)은 표본이 작은 half_2024에서 우연에 의존했을 수 있음.
여기서는 각 fold의 val set을 5-fold CV로 나눠서, 메타모델은 매번 4/5로 학습하고 나머지
1/5에 예측 -> 이걸 다 모으면 val set 전체를 커버하는 OOF 예측이 나옴(정보 유출 없음,
매 행이 자기 자신을 학습에 쓴 적 없는 메타모델로만 예측됨). 고정 alpha 블렌드는 학습이
필요 없으므로 val set 전체에 바로 채점해서 공정 비교.

cache_fold_predictions.py의 캐시를 그대로 사용, 재학습 없음.
"""
import pickle

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold

from src.local_validation import score

FOLD_NAMES = ["half_2024", "expand_2023", "expand_2021", "expand_2022"]
BLEND_ALPHA = 0.61
N_SPLITS = 5
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

    meta_df = context_df.copy()
    meta_df["cat_pred"] = cat_pred
    meta_df["mlp_pred"] = mlp_pred
    meta_df["pred_diff"] = cat_pred - mlp_pred
    for c in CATEGORICAL_CONTEXT:
        meta_df[c] = meta_df[c].astype(str)
    meta_features = list(meta_df.columns)

    oof_meta_pred = np.zeros(n)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    for fold_i, (idx_train, idx_test) in enumerate(kf.split(np.arange(n))):
        train_pool = Pool(meta_df.iloc[idx_train][meta_features], y_val[idx_train], cat_features=CATEGORICAL_CONTEXT)
        test_pool = Pool(meta_df.iloc[idx_test][meta_features], y_val[idx_test], cat_features=CATEGORICAL_CONTEXT)
        model = CatBoostClassifier(**META_PARAMS)
        model.fit(train_pool, eval_set=test_pool)
        oof_meta_pred[idx_test] = model.predict_proba(test_pool)[:, 1]

    meta_score = score(oof_meta_pred, y_val)
    fixed_blend = BLEND_ALPHA * cat_pred + (1 - BLEND_ALPHA) * mlp_pred
    fixed_score = score(fixed_blend, y_val)

    print(f"[{fold_name}] 고정 alpha=0.61 (val 전체): {fixed_score:.2f}", flush=True)
    print(f"[{fold_name}] 컨텍스트 메타모델 ({N_SPLITS}-fold OOF, val 전체): {meta_score:.2f}", flush=True)
    return fixed_score, meta_score


def main():
    fixed_scores, meta_scores = [], []
    for fold_name in FOLD_NAMES:
        fixed_s, meta_s = run_fold(fold_name)
        fixed_scores.append(fixed_s)
        meta_scores.append(meta_s)

    print("\n=== 요약 (val set 전체, 5-fold OOF) ===")
    print(f"고정 alpha=0.61: " + ", ".join(f"{n}={s:.2f}" for n, s in zip(FOLD_NAMES, fixed_scores)) +
          f" | 평균={np.mean(fixed_scores):.2f}")
    print(f"컨텍스트 메타모델: " + ", ".join(f"{n}={s:.2f}" for n, s in zip(FOLD_NAMES, meta_scores)) +
          f" | 평균={np.mean(meta_scores):.2f}")


if __name__ == "__main__":
    main()
