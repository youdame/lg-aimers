"""컨텍스트 메타모델 재검증 -- leave-one-fold-out (EXPERIMENTS.md §6-18 정정).

이전 버전(experiment_context_blend_meta_cv.py)은 val set 내부 5-fold CV로 메타모델을
학습했는데, 이건 "타깃 시즌 자기 자신의 레이블"로 메타모델을 학습하는 구조라 실전
(test.csv, 2025)에서는 재현 불가능함(2025 행은 레이블이 아예 없음) -- 순수한 미래
일반화 성능이 아니라 정보 유출에 가까운 결과였음(expand_2021/2022에서 나온
+396/+439 같은 비현실적 개선폭이 그 증거).

올바른 방식: 4개 fold 중 3개(레이블 있음, 서로 다른 시기)의 (cat_pred, mlp_pred,
context, y)를 합쳐서 메타모델을 학습하고, 나머지 1개 fold에는 그 메타모델을
그대로(재학습 없이) 적용 -- 이게 "레이블 없는 미래 시즌에 과거에 학습된 메타모델을
적용"하는 실전 상황과 구조적으로 동일함. 4개 fold 각각을 한 번씩 홀드아웃.
"""
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from local_validation import score

FOLD_NAMES = ["half_2024", "expand_2023", "expand_2021", "expand_2022"]
BLEND_ALPHA = 0.61
CATEGORICAL_CONTEXT = ["pitcher_hand", "batter_hand"]

META_PARAMS = dict(
    depth=4, learning_rate=0.05, iterations=500, l2_leaf_reg=5.0,
    loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=30, verbose=False,
)


def load_fold(fold_name):
    with open(f"./model_py311/fold_predictions_{fold_name}.pkl", "rb") as f:
        return pickle.load(f)


def build_meta_df(d):
    meta_df = d["context_df"].copy()
    meta_df["cat_pred"] = d["cat_pred"]
    meta_df["mlp_pred"] = d["mlp_pred"]
    meta_df["pred_diff"] = d["cat_pred"] - d["mlp_pred"]
    for c in CATEGORICAL_CONTEXT:
        meta_df[c] = meta_df[c].astype(str)
    return meta_df


def main():
    folds = {name: load_fold(name) for name in FOLD_NAMES}
    meta_dfs = {name: build_meta_df(d) for name, d in folds.items()}
    meta_features = list(next(iter(meta_dfs.values())).columns)

    fixed_scores, meta_scores = {}, {}
    for held_out in FOLD_NAMES:
        train_names = [n for n in FOLD_NAMES if n != held_out]
        train_X = pd.concat([meta_dfs[n] for n in train_names], ignore_index=True)
        train_y = np.concatenate([folds[n]["y_val"] for n in train_names])

        test_X = meta_dfs[held_out]
        test_y = folds[held_out]["y_val"]

        train_pool = Pool(train_X[meta_features], train_y, cat_features=CATEGORICAL_CONTEXT)
        test_pool = Pool(test_X[meta_features], test_y, cat_features=CATEGORICAL_CONTEXT)

        model = CatBoostClassifier(**META_PARAMS)
        model.fit(train_pool, eval_set=test_pool)
        meta_pred = model.predict_proba(test_pool)[:, 1]
        meta_scores[held_out] = score(meta_pred, test_y)

        cat_pred, mlp_pred = folds[held_out]["cat_pred"], folds[held_out]["mlp_pred"]
        fixed_blend = BLEND_ALPHA * cat_pred + (1 - BLEND_ALPHA) * mlp_pred
        fixed_scores[held_out] = score(fixed_blend, test_y)

        print(f"[held-out={held_out}] (학습: {train_names}) "
              f"고정 alpha=0.61={fixed_scores[held_out]:.2f} | "
              f"컨텍스트 메타모델(LOFO)={meta_scores[held_out]:.2f}", flush=True)

    print("\n=== 요약 (leave-one-fold-out, 진짜 미래 일반화 근사) ===")
    print("고정 alpha=0.61: " + ", ".join(f"{n}={fixed_scores[n]:.2f}" for n in FOLD_NAMES) +
          f" | 평균={np.mean(list(fixed_scores.values())):.2f}")
    print("컨텍스트 메타모델(LOFO): " + ", ".join(f"{n}={meta_scores[n]:.2f}" for n in FOLD_NAMES) +
          f" | 평균={np.mean(list(meta_scores.values())):.2f}")


if __name__ == "__main__":
    main()
