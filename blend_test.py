import os

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"
CATBOOST_MODEL_PATH = "../lgaimers9/open/reference/best_model.pkl"
XGB_MODEL_PATH = "./model_py311/xgb_val.pkl"  # season<2024만 학습한 검증 전용 모델 (리크 방지)


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def brier_eval(y_true, y_pred):
    """xgb_val.pkl 안에 pickle된 커스텀 eval_metric 참조를 풀기 위해 동일 이름으로 재정의."""
    return float(((y_pred - y_true) ** 2).mean())


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
    features = [col for col in train_df.columns if col not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include='object').columns if c != 'row_id' and c in features]

    val_df = train_df.loc[val_mask].reset_index(drop=True)
    y_val = val_df[TARGET].values

    # --- CatBoost 예측 (범주형은 문자열 그대로 + Pool) ---
    val_cat = val_df.copy()
    for col in categorical_features:
        val_cat[col] = val_cat[col].astype(str).fillna('missing')
    catboost_model = joblib.load(CATBOOST_MODEL_PATH)
    cat_pool = Pool(data=val_cat[features], cat_features=categorical_features)
    cat_pred = catboost_model.predict_proba(cat_pool)[:, 1]

    # --- XGBoost 예측 (범주형은 category dtype) ---
    val_xgb = val_df.copy()
    for col in categorical_features:
        val_xgb[col] = val_xgb[col].astype(str).fillna('missing').astype('category')
    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_pred = xgb_model.predict_proba(val_xgb[features])[:, 1]

    cat_brier, cat_bss = compute_bss(cat_pred, y_val)
    xgb_brier, xgb_bss = compute_bss(xgb_pred, y_val)
    print(f"CatBoost 단독: BSS={cat_bss:.5f} (score={max(0,100000*cat_bss):.2f})")
    print(f"XGBoost  단독: BSS={xgb_bss:.5f} (score={max(0,100000*xgb_bss):.2f})")

    corr = np.corrcoef(cat_pred, xgb_pred)[0, 1]
    print(f"두 예측값 상관계수: {corr:.4f}")

    print("\n[Alpha 스윕: alpha*CatBoost + (1-alpha)*XGBoost]")
    best_alpha, best_bss = None, -np.inf
    for alpha in np.arange(0.0, 1.01, 0.01):
        blend_pred = alpha * cat_pred + (1 - alpha) * xgb_pred
        _, bss = compute_bss(blend_pred, y_val)
        if bss > best_bss:
            best_bss, best_alpha = bss, alpha

    print(f"\n최적 alpha={best_alpha:.2f} -> BSS={best_bss:.5f} (score={max(0,100000*best_bss):.2f})")
    print(f"CatBoost 단독 대비: {max(0,100000*best_bss) - max(0,100000*cat_bss):+.2f}")
    print(f"XGBoost  단독 대비: {max(0,100000*best_bss) - max(0,100000*xgb_bss):+.2f}")


if __name__ == "__main__":
    main()
