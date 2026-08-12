import os
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import process_trackman_features_safe, add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"

# retrain_xgb.py와 동일한 튜닝 결과 (Trial 012, BSS 0.00751 / score 751.05)
BASE_PARAMS = dict(
    max_depth=6,
    learning_rate=0.014399911854309236,
    subsample=0.6367420040370018,
    colsample_bytree=0.9920379472108309,
    reg_lambda=0.06047908057801884,
    reg_alpha=4.368166043300848,
    min_child_weight=12,
    gamma=0.008969955448879116,
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def build_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna("missing").astype("category")

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET]
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET]
    return X_train, y_train, X_val, y_val, features


def fit_and_eval(X_train, y_train, X_val, y_val, label):
    t0 = time.time()
    model = XGBClassifier(
        **BASE_PARAMS,
        n_estimators=3000,
        eval_metric=brier_eval,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    brier, bss = compute_bss(preds, y_val)
    score = max(0, 100000 * bss)
    print(f"[{label}] n_features={X_train.shape[1]:3d} | best_iter={model.best_iteration:4d} | "
          f"BSS={bss:.5f} (score={score:.2f}) | {time.time()-t0:.1f}s", flush=True)
    return model, bss, score


def main():
    print("[Prune] 데이터 로드 및 피처 엔지니어링...", flush=True)
    X_train, y_train, X_val, y_val, features = build_data()
    print(f"[Prune] train={len(X_train)} val={len(X_val)} features={len(features)}", flush=True)

    # 1) 베이스라인 (현재 retrain_xgb.py와 동일한 123피처 구성)
    base_model, base_bss, base_score = fit_and_eval(X_train, y_train, X_val, y_val, "baseline (all 123)")

    # 2) gain importance 집계. get_score()는 한 번도 split에 안 쓰인 피처는 아예 안 나옴 -> 그게 1차 제거 후보.
    importances = base_model.get_booster().get_score(importance_type="gain")
    used_features = set(importances.keys())
    zero_gain_features = [f for f in features if f not in used_features]
    print(f"\n[Zero-gain] 학습 중 단 한 번도 분기에 쓰이지 않은 피처: {len(zero_gain_features)}개")
    for f in zero_gain_features:
        print(f"  - {f}")

    ranked = sorted(importances.items(), key=lambda kv: kv[1])
    print(f"\n[Bottom 20 gain] (사용은 됐지만 기여도 최하위)")
    for name, gain in ranked[:20]:
        print(f"  {name}: {gain:.4f}")

    # 3) zero-gain 피처만 제거하고 재학습 -> BSS 변화 확인
    if zero_gain_features:
        keep_features = [f for f in features if f not in zero_gain_features]
        Xtr2 = X_train[keep_features]
        Xval2 = X_val[keep_features]
        _, bss2, score2 = fit_and_eval(Xtr2, y_train, Xval2, y_val, "drop zero-gain")
        print(f"  -> baseline 대비: {score2 - base_score:+.2f}")

    # 4) zero-gain + bottom-20 used까지 합쳐서 제거 -> 조금 더 공격적인 pruning
    bottom20_names = [name for name, _ in ranked[:20]]
    drop_set = set(zero_gain_features) | set(bottom20_names)
    keep_features_aggr = [f for f in features if f not in drop_set]
    Xtr3 = X_train[keep_features_aggr]
    Xval3 = X_val[keep_features_aggr]
    _, bss3, score3 = fit_and_eval(Xtr3, y_train, Xval3, y_val, f"drop zero-gain+bottom20 ({len(drop_set)}개 제거)")
    print(f"  -> baseline 대비: {score3 - base_score:+.2f}")

    print("\n" + "=" * 60)
    print(f"요약: baseline={base_score:.2f} | features_before={len(features)}")
    if zero_gain_features:
        print(f"      zero-gain 제거({len(zero_gain_features)}개)={score2:.2f} (features={len(keep_features)})")
    print(f"      공격적 제거({len(drop_set)}개)={score3:.2f} (features={len(keep_features_aggr)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
