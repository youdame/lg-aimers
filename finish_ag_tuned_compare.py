import os

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
AG_MODEL_PATH = "/Users/joyudam/GitHub/open/AutogluonModels/ag-20260812_055333"


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def brier_score_func(y_true, y_pred):
    return float(((np.asarray(y_pred) - np.asarray(y_true)) ** 2).mean())


brier_scorer = make_scorer(name="brier", score_func=brier_score_func, optimum=0, greater_is_better=False, needs_proba=True)


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    val_part = train_df.loc[val_mask, [c for c in train_df.columns if c != 'row_id']].reset_index(drop=True)
    y_val = val_part[TARGET].values

    predictor = TabularPredictor.load(AG_MODEL_PATH)
    nn_pred = predictor.predict_proba(val_part)[1].values  # best = WeightedEnsemble_L2

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

    cat_score = max(0, 100000 * cat_bss)
    xgb_score = max(0, 100000 * xgb_bss)

    print(f"CatBoost 단독      : {cat_score:.2f}")
    print(f"XGBoost 단독       : {xgb_score:.2f}")
    print(f"튜닝된 NN 단독     : 604.29")

    print("\n[상관계수]")
    print(f"  CatBoost vs 튜닝된 NN : {np.corrcoef(cat_pred, nn_pred)[0,1]:.4f}  (참고: 튜닝전 NN은 0.7315)")
    print(f"  XGBoost  vs 튜닝된 NN : {np.corrcoef(xgb_pred, nn_pred)[0,1]:.4f}")

    print("\n[3-way 블렌드: w1*CatBoost + w2*XGBoost + w3*튜닝된NN] (그리드서치, step=0.05)")
    best_w, best3_bss = None, -np.inf
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = 1 - w1 - w2
            blend = w1 * cat_pred + w2 * xgb_pred + w3 * nn_pred
            _, bss = compute_bss(blend, y_val)
            if bss > best3_bss:
                best3_bss, best_w = bss, (round(w1, 2), round(w2, 2), round(w3, 2))
    best3_score = max(0, 100000 * best3_bss)
    print(f"  최적 weights (Cat,XGB,NN)={best_w} -> score={best3_score:.2f} (CatBoost 대비 {best3_score - cat_score:+.2f})")


if __name__ == "__main__":
    main()
