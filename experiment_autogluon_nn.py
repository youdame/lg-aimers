"""AutoGluon의 NN_TORCH(임베딩 포함, 제대로 된 신경망)만 학습시켜서
아까 손으로 짠 naive MLP(score 38.96)보다 얼마나 나은지, 그리고 그걸로 다시
CatBoost/XGBoost와 블렌드했을 때 실제 이득이 있는지 확인한다.
시간 제한 있는 대회 규칙과 무관하게 '탐색용'으로만 씀 (최종 제출 후보 아님).
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer

from feature_engineering import process_trackman_features_safe, add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"
CATBOOST_MODEL_PATH = "../lgaimers9/open/reference/best_model.pkl"
XGB_MODEL_PATH = "./model_py311/xgb_val.pkl"
TIME_LIMIT = int(os.environ.get("AG_TIME_LIMIT", 600))


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def brier_score_func(y_true, y_pred):
    return float(((np.asarray(y_pred) - np.asarray(y_true)) ** 2).mean())


brier_scorer = make_scorer(
    name="brier",
    score_func=brier_score_func,
    optimum=0,
    greater_is_better=False,
    needs_proba=True,
)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id']
    keep_cols = [c for c in train_df.columns if c not in drop_cols]

    train_part = train_df.loc[train_mask, keep_cols].reset_index(drop=True)
    val_part = train_df.loc[val_mask, keep_cols].reset_index(drop=True)
    y_val = val_part[TARGET].values

    sample_n = int(os.environ.get("AG_TRAIN_SAMPLE", 300_000))
    if sample_n and sample_n < len(train_part):
        train_part = train_part.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[메모리 제약으로 학습 데이터 샘플링] {sample_n}행만 사용 (val은 전체 {len(val_part)}행 그대로)", flush=True)

    print(f"train={len(train_part)} val={len(val_part)} cols={len(keep_cols)}", flush=True)

    t0 = time.time()
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric=brier_scorer,
        problem_type='binary',
        verbosity=2,
    ).fit(
        train_data=train_part,
        tuning_data=val_part,
        use_bag_holdout=True,
        hyperparameters={'NN_TORCH': {}, 'FASTAI': {}, 'REALMLP': {}},
        time_limit=TIME_LIMIT,
    )
    print(f"[AutoGluon NN_TORCH] 학습 완료: {time.time()-t0:.1f}s", flush=True)

    print("\n[Leaderboard on val]")
    lb = predictor.leaderboard(val_part, silent=True)
    print(lb.to_string(), flush=True)

    # 개별 모델(NN_TORCH/FASTAI/REALMLP)별 예측 + BSS 따로 확인
    model_names = [m for m in predictor.model_names() if 'WeightedEnsemble' not in m]
    model_preds = {}
    print("\n[모델별 개별 성능]")
    for m in model_names:
        p = predictor.predict_proba(val_part, model=m)[1].values
        _, bss_m = compute_bss(p, y_val)
        score_m = max(0, 100000 * bss_m)
        model_preds[m] = p
        print(f"  {m}: score={score_m:.2f}")

    best_model_name = max(model_preds, key=lambda m: compute_bss(model_preds[m], y_val)[1])
    nn_pred = model_preds[best_model_name]
    _, nn_bss = compute_bss(nn_pred, y_val)
    nn_score = max(0, 100000 * nn_bss)
    print(f"\n최고 성능 NN 계열: {best_model_name} (score={nn_score:.2f}) -- 이걸로 아래 블렌드 진행", flush=True)

    # --- 기존 CatBoost / XGBoost 예측 (blend_test.py와 동일 로직) ---
    features = [c for c in train_df.columns if c not in ('row_id', TARGET)]
    categorical_features = [c for c in tr_final.select_dtypes(include='object').columns if c != 'row_id' and c in features]

    val_cat_df = train_df.loc[val_mask].reset_index(drop=True).copy()
    for col in categorical_features:
        val_cat_df[col] = val_cat_df[col].astype(str).fillna('missing')
    catboost_model = joblib.load(CATBOOST_MODEL_PATH)
    cat_pool = Pool(data=val_cat_df[features], cat_features=categorical_features)
    cat_pred = catboost_model.predict_proba(cat_pool)[:, 1]
    _, cat_bss = compute_bss(cat_pred, y_val)

    val_xgb_df = train_df.loc[val_mask].reset_index(drop=True).copy()
    for col in categorical_features:
        val_xgb_df[col] = val_xgb_df[col].astype(str).fillna('missing').astype('category')
    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_pred = xgb_model.predict_proba(val_xgb_df[features])[:, 1]
    _, xgb_bss = compute_bss(xgb_pred, y_val)

    print(f"CatBoost 단독: {max(0,100000*cat_bss):.2f} | XGBoost 단독: {max(0,100000*xgb_bss):.2f}")

    print("\n[상관계수]")
    print(f"  CatBoost vs AG-NN : {np.corrcoef(cat_pred, nn_pred)[0,1]:.4f}")
    print(f"  XGBoost  vs AG-NN : {np.corrcoef(xgb_pred, nn_pred)[0,1]:.4f}")
    print(f"  (참고) CatBoost vs 이전 naive MLP 상관계수 : 0.6537")

    print("\n[3-way 블렌드: w1*CatBoost + w2*XGBoost + w3*AG-NN] (그리드서치, step=0.05)")
    best_w, best3_bss = None, -np.inf
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = 1 - w1 - w2
            blend = w1 * cat_pred + w2 * xgb_pred + w3 * nn_pred
            _, bss = compute_bss(blend, y_val)
            if bss > best3_bss:
                best3_bss, best_w = bss, (round(w1, 2), round(w2, 2), round(w3, 2))
    best3_score = max(0, 100000 * best3_bss)

    print("\n" + "=" * 60)
    print("요약")
    print(f"  CatBoost 단독            : {max(0,100000*cat_bss):.2f}")
    print(f"  XGBoost  단독            : {max(0,100000*xgb_bss):.2f}")
    print(f"  naive MLP (이전 실험)    : 38.96")
    print(f"  AutoGluon NN_TORCH 단독  : {nn_score:.2f}")
    print(f"  Cat+XGB 2-way (기존)     : 821.42 (+2.88)")
    print(f"  Cat+XGB+AG-NN 3-way      : {best3_score:.2f} ({best3_score - max(0,100000*cat_bss):+.2f}) weights(Cat,XGB,NN)={best_w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
