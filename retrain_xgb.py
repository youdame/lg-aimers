import os
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering import process_trackman_features_safe, add_engineered_features, add_experimental_features

DATA_DIR = "./data"
TARGET = "control_success"

# tune_xgb.py 2차(좁힌 구간) Optuna 탐색 결과 (Trial 012, BSS 0.00751 / score 751.05)
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
    """조기종료 기준을 logloss 대신 대회 채점 지표(Brier)에 직접 맞춤."""
    return float(((y_pred - y_true) ** 2).mean())


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
    # add_experimental_features(train_df) 시도했으나 실측 결과 역효과 확인 (737.03 < 751.05).
    # 신규 피처 7개가 feature importance Top 25에 하나도 안 들어가 노이즈로만 작용한 것으로 판단, 제외.

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
        eval_metric=brier_eval,
        early_stopping_rounds=50,
    )
    val_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    print(f"검증용 학습 완료 :: {time.time() - t:.1f}s (best_iteration={val_model.best_iteration})")

    # season==2024를 학습에 안 쓴 이 모델을 따로 저장 (블렌딩/평가용).
    # xgb.pkl(최종 전체데이터 재학습본)은 2024를 이미 학습에 포함하므로
    # season==2024 검증에 쓰면 리크가 됨 -- 반드시 이 파일을 써야 함.
    os.makedirs("./model_py311", exist_ok=True)
    joblib.dump(val_model, "./model_py311/xgb_val.pkl", compress=3)
    print("저장 완료: ./model_py311/xgb_val.pkl (검증 전용, season<2024만 학습)")

    val_pred = val_model.predict_proba(X_val)[:, 1]
    brier, bss = compute_bss(val_pred, y_val)
    score = max(0, 100000 * bss)
    print(f"Brier: {brier:.6f} | BSS: {bss:.5f}")
    print(f"Validation Score: {score:.2f}")

    # feature importance (gain 기준: 그 피처가 분기에 쓰였을 때 평균적으로 얼마나 손실을 줄였는지)
    importances = val_model.get_booster().get_score(importance_type="gain")
    top_n = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:25]
    print("\n[Feature Importance Top 25 (gain 기준)]")
    for name, gain in top_n:
        marker = "  <- 신규 실험 피처" if name in (
            "runner_pressure_reverse", "runner_pressure_middle", "scoring_position_reverse",
            "win_expectancy_gap", "fastball_count_interaction",
            "breaking_fullcount_interaction", "offspeed_pressure",
        ) else ""
        print(f"  {name}: {gain:.2f}{marker}")

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
