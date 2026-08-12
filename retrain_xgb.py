import os
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import process_trackman_features_safe, add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"

# tune_xgb.py Optuna 탐색 결과 (Trial 020, BSS 0.00739 / score 739.48)
BASE_PARAMS = dict(
    max_depth=8,
    learning_rate=0.012583472359491006,
    subsample=0.6723396692361412,
    colsample_bytree=0.8017774544431577,
    reg_lambda=0.11135751392032872,
    reg_alpha=7.461678308002679,
    min_child_weight=11,
    gamma=0.001771199734883011,
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


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    # CatBoost 파이프라인(lgaimers9/code/train.py)과 동일한 트랙맨 조인 + 파생 피처
    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)

    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024

    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id', TARGET]
    features = [col for col in train_df.columns if col not in drop_cols]

    # CatBoost와 동일하게 문자열(object) 컬럼을 범주형으로 취급.
    # XGBoost는 cat_features 리스트 대신 'category' dtype으로 직접 표시해야 함.
    categorical_features = [c for c in tr_final.select_dtypes(include='object').columns if c != 'row_id' and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna('missing').astype('category')

    X_train, y_train = train_df.loc[train_mask, features], train_df.loc[train_mask, TARGET]
    X_val, y_val = train_df.loc[val_mask, features], train_df.loc[val_mask, TARGET]
    print(f"train: {len(X_train)} | val: {len(X_val)} | 피처 수: {len(features)} (범주형 {len(categorical_features)}개)")

    t = time.time()
    val_model = XGBClassifier(
        **BASE_PARAMS,
        n_estimators=3000,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    val_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    print(f"검증용 학습 완료 :: {time.time() - t:.1f}s (best_iteration={val_model.best_iteration})")

    val_pred = val_model.predict_proba(X_val)[:, 1]
    brier, bss = compute_bss(val_pred, y_val)
    score = max(0, 100000 * bss)
    print(f"Brier: {brier:.6f} | BSS: {bss:.5f}")
    print(f"Validation Score: {score:.2f}")

    # 최종 제출용: 전체 데이터(2019~2024)로 재학습 (CatBoost dopip.py의 Full Retrain과 동일 관례)
    final_model = XGBClassifier(
        **BASE_PARAMS,
        n_estimators=val_model.best_iteration + 10,
    )

    t = time.time()
    final_model.fit(train_df[features], train_df[TARGET])
    print(f"최종 재학습 완료 :: {time.time() - t:.1f}s (n={len(train_df)})")

    os.makedirs("./model_py311", exist_ok=True)
    joblib.dump(final_model, "./model_py311/xgb.pkl", compress=3)
    print("저장 완료: ./model_py311/xgb.pkl")


if __name__ == "__main__":
    main()
