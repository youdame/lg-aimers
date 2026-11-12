"""로지스틱 메타모델 재검증 -- 창현님 방식(같은 레짐 내부 분할)으로 정정.

experiment_logistic_meta.py(LOFO, 레짐 통째로 다른 fold로 건너뜀)는 창현님 검증
방식과 다른 걸 측정한 것으로 판단됨(EXPERIMENTS.md 참고). 창현님은 season==2024
안에서 시간순으로 쪼개 "가까운 미래"에 적용하는 방식으로 검증했음(rolling-origin,
3/3 승 평균 +14.22). 여기서는 우리 fold(half_2024/expand_2023) 각각 내부를 70/30
랜덤 분할해서 재현 -- alpha/메타모델 둘 다 70%에서 재학습, 30%로 평가.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.local_validation import score

FOLD_NAMES = ["half_2024", "expand_2023"]
SPLIT_SEED = 42


def load_fold(fold_name):
    with open(f"./model_py311/fold_predictions_v2_{fold_name}.pkl", "rb") as f:
        return pickle.load(f)


def best_alpha(cat_pred, mlp_pred, y):
    best_a, best_s = 0.5, -1
    for a in np.arange(0.0, 1.001, 0.01):
        s = score(a * cat_pred + (1 - a) * mlp_pred, y)
        if s > best_s:
            best_a, best_s = a, s
    return best_a, best_s


def main():
    alpha_scores, meta_scores = {}, {}
    for fold_name in FOLD_NAMES:
        d = load_fold(fold_name)
        n = len(d["y_val"])
        idx_train, idx_test = train_test_split(np.arange(n), test_size=0.3, random_state=SPLIT_SEED)

        cat_tr, mlp_tr, y_tr = d["cat_pred"][idx_train], d["mlp_pred"][idx_train], d["y_val"][idx_train]
        cat_te, mlp_te, y_te = d["cat_pred"][idx_test], d["mlp_pred"][idx_test], d["y_val"][idx_test]

        alpha, _ = best_alpha(cat_tr, mlp_tr, y_tr)
        alpha_test_score = score(alpha * cat_te + (1 - alpha) * mlp_te, y_te)

        meta = LogisticRegression()
        meta.fit(np.column_stack([cat_tr, mlp_tr]), y_tr)
        meta_pred = meta.predict_proba(np.column_stack([cat_te, mlp_te]))[:, 1]
        meta_test_score = score(meta_pred, y_te)

        alpha_scores[fold_name] = alpha_test_score
        meta_scores[fold_name] = meta_test_score
        print(f"[{fold_name}] alpha(재탐색={alpha:.2f}, 30%로 평가)={alpha_test_score:.2f} | "
              f"로지스틱메타(70%로 학습, 30%로 평가)={meta_test_score:.2f} | "
              f"Δ={meta_test_score-alpha_test_score:+.2f} | "
              f"w_cat={meta.coef_[0][0]:.3f}, w_mlp={meta.coef_[0][1]:.3f}, intercept={meta.intercept_[0]:.3f}",
              flush=True)

    print("\n=== 요약 (같은 레짐 내부 70/30 분할) ===")
    print(f"alpha 재탐색: " + ", ".join(f"{n}={alpha_scores[n]:.2f}" for n in FOLD_NAMES) +
          f" | 평균={np.mean(list(alpha_scores.values())):.2f}")
    print(f"로지스틱 메타: " + ", ".join(f"{n}={meta_scores[n]:.2f}" for n in FOLD_NAMES) +
          f" | 평균={np.mean(list(meta_scores.values())):.2f}")


if __name__ == "__main__":
    main()
