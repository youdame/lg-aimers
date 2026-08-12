"""LightGBM 단독 성능을 확인 (튜닝 없이 기본값 근처, XGBoost와 동일한 조건).
CatBoost(818.54, Optuna 40trial 튜닝됨) / XGBoost(752.70, Optuna 튜닝됨)와 공정 비교하려면
나중에 LightGBM도 튜닝이 필요하지만, 우선 '단독으로 붙어볼 만한지' 방향성부터 확인.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"
CATBOOST_MODEL_PATH = "../lgaimers9/open/reference/best_model.pkl"
XGB_MODEL_PATH = "./model_py311/xgb_val.pkl"


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def lgb_brier_feval(y_true, y_pred):
    return 'brier', float(((y_pred - y_true) ** 2).mean()), False  # False = 낮을수록 좋음


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id', TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include='object').columns if c != 'row_id' and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna('missing').astype('category')

    X_train, y_train = train_df.loc[train_mask, features], train_df.loc[train_mask, TARGET]
    X_val, y_val = train_df.loc[val_mask, features], train_df.loc[val_mask, TARGET]
    print(f"train={len(X_train)} val={len(X_val)} features={len(features)}", flush=True)

    t0 = time.time()
    model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lgb_brier_feval,
        callbacks=[__import__('lightgbm').early_stopping(50, verbose=False)],
        categorical_feature=categorical_features,
    )
    print(f"[LightGBM] 학습 완료: {time.time()-t0:.1f}s, best_iter={model.best_iteration_}", flush=True)

    lgb_pred = model.predict_proba(X_val)[:, 1]
    lgb_brier, lgb_bss = compute_bss(lgb_pred, y_val)
    lgb_score = max(0, 100000 * lgb_bss)
    print(f"LightGBM 단독: BSS={lgb_bss:.5f} (score={lgb_score:.2f})", flush=True)

    # --- 기존 CatBoost / XGBoost ---
    val_cat_df = train_df.loc[val_mask].reset_index(drop=True).copy()
    for col in categorical_features:
        val_cat_df[col] = val_cat_df[col].astype(str)
    catboost_model = joblib.load(CATBOOST_MODEL_PATH)
    cat_pool = Pool(data=val_cat_df[features], cat_features=categorical_features)
    cat_pred = catboost_model.predict_proba(cat_pool)[:, 1]
    _, cat_bss = compute_bss(cat_pred, y_val)

    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    _, xgb_bss = compute_bss(xgb_pred, y_val)

    cat_score = max(0, 100000 * cat_bss)
    xgb_score = max(0, 100000 * xgb_bss)

    print("\n[상관계수]")
    print(f"  CatBoost vs LightGBM : {np.corrcoef(cat_pred, lgb_pred)[0,1]:.4f}")
    print(f"  XGBoost  vs LightGBM : {np.corrcoef(xgb_pred, lgb_pred)[0,1]:.4f}")

    print("\n[2-way 블렌드: alpha*CatBoost + (1-alpha)*LightGBM]")
    best_alpha, best_bss = None, -np.inf
    for alpha in np.arange(0.0, 1.01, 0.01):
        blend = alpha * cat_pred + (1 - alpha) * lgb_pred
        _, bss = compute_bss(blend, y_val)
        if bss > best_bss:
            best_bss, best_alpha = bss, alpha
    best_score = max(0, 100000 * best_bss)

    print("\n" + "=" * 60)
    print("요약")
    print(f"  CatBoost 단독 (튜닝됨)      : {cat_score:.2f}")
    print(f"  XGBoost  단독 (튜닝됨)      : {xgb_score:.2f}")
    print(f"  LightGBM 단독 (튜닝 안 됨)  : {lgb_score:.2f}")
    print(f"  Cat+LightGBM 최적블렌드    : {best_score:.2f} (alpha={best_alpha:.2f}, CatBoost 대비 {best_score-cat_score:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
