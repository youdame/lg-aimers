"""2-입력 로지스틱회귀 메타모델 검증 (EXPERIMENTS.md §6-11, 창현님 저장소 +14.22).

final_pred = sigmoid(w_cat * cat_pred + w_mlp * mlp_pred + intercept)

leave-one-fold-out: half_2024/expand_2023 중 하나로 메타모델(3개 파라미터)을 학습하고
다른 fold에 재학습 없이 적용 -- 컨텍스트 블렌드 사고(§6-18) 때 배운 교훈 그대로,
타깃 fold 자신의 레이블로 메타모델을 학습하지 않음. 고정 alpha=0.61과 비교.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.local_validation import score

FOLD_NAMES = ["half_2024", "expand_2023"]
BLEND_ALPHA = 0.61


def load_fold(fold_name):
    with open(f"./model_py311/fold_predictions_v2_{fold_name}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    folds = {name: load_fold(name) for name in FOLD_NAMES}

    fixed_scores, meta_scores, coefs = {}, {}, {}
    for held_out in FOLD_NAMES:
        train_name = [n for n in FOLD_NAMES if n != held_out][0]
        train_d = folds[train_name]
        test_d = folds[held_out]

        X_train = np.column_stack([train_d["cat_pred"], train_d["mlp_pred"]])
        y_train = train_d["y_val"]
        X_test = np.column_stack([test_d["cat_pred"], test_d["mlp_pred"]])
        y_test = test_d["y_val"]

        meta = LogisticRegression()
        meta.fit(X_train, y_train)
        meta_pred = meta.predict_proba(X_test)[:, 1]
        meta_scores[held_out] = score(meta_pred, y_test)
        coefs[held_out] = (meta.coef_[0][0], meta.coef_[0][1], meta.intercept_[0])

        fixed_blend = BLEND_ALPHA * test_d["cat_pred"] + (1 - BLEND_ALPHA) * test_d["mlp_pred"]
        fixed_scores[held_out] = score(fixed_blend, y_test)

        print(f"[held-out={held_out}] (메타모델 학습: {train_name}) "
              f"고정 alpha=0.61={fixed_scores[held_out]:.2f} | "
              f"로지스틱 메타(LOFO)={meta_scores[held_out]:.2f} | "
              f"w_cat={coefs[held_out][0]:.3f}, w_mlp={coefs[held_out][1]:.3f}, intercept={coefs[held_out][2]:.3f}",
              flush=True)

    print("\n=== 요약 (leave-one-fold-out) ===")
    print(f"고정 alpha=0.61: " + ", ".join(f"{n}={fixed_scores[n]:.2f}" for n in FOLD_NAMES) +
          f" | 평균={np.mean(list(fixed_scores.values())):.2f}")
    print(f"로지스틱 메타(LOFO): " + ", ".join(f"{n}={meta_scores[n]:.2f}" for n in FOLD_NAMES) +
          f" | 평균={np.mean(list(meta_scores.values())):.2f}")

    # 참고: 전체 두 fold를 합쳐서 학습한 메타모델 계수(프로덕션에 실제로 쓸 값 후보)
    X_all = np.column_stack([
        np.concatenate([folds[n]["cat_pred"] for n in FOLD_NAMES]),
        np.concatenate([folds[n]["mlp_pred"] for n in FOLD_NAMES]),
    ])
    y_all = np.concatenate([folds[n]["y_val"] for n in FOLD_NAMES])
    meta_all = LogisticRegression()
    meta_all.fit(X_all, y_all)
    print(f"\n참고: 두 fold 전체로 학습한 메타모델 계수 (프로덕션 후보) -- "
          f"w_cat={meta_all.coef_[0][0]:.4f}, w_mlp={meta_all.coef_[0][1]:.4f}, "
          f"intercept={meta_all.intercept_[0]:.4f}")


if __name__ == "__main__":
    main()
