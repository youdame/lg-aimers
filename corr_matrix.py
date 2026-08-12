import os

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from autogluon.tabular import TabularPredictor

from feature_engineering import process_trackman_features_safe, add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"
CATBOOST_MODEL_PATH = "../lgaimers9/open/reference/best_model.pkl"
XGB_MODEL_PATH = "./model_py311/xgb_val.pkl"
AG_MODEL_PATH = "/Users/joyudam/GitHub/open/AutogluonModels/ag-20260812_054207"


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def brier_score_func(y_true, y_pred):
    return float(((np.asarray(y_pred) - np.asarray(y_true)) ** 2).mean())


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
    nn_torch_pred = predictor.predict_proba(val_part, model='NeuralNetTorch')[1].values
    nn_fastai_pred = predictor.predict_proba(val_part, model='NeuralNetFastAI')[1].values

    features = [c for c in train_df.columns if c not in ('row_id', TARGET)]
    categorical_features = [c for c in tr_final.select_dtypes(include='object').columns if c != 'row_id' and c in features]

    val_cat_df = train_df.loc[val_mask].reset_index(drop=True).copy()
    for col in categorical_features:
        val_cat_df[col] = val_cat_df[col].astype(str).fillna('missing')
    catboost_model = joblib.load(CATBOOST_MODEL_PATH)
    cat_pool = Pool(data=val_cat_df[features], cat_features=categorical_features)
    cat_pred = catboost_model.predict_proba(cat_pool)[:, 1]

    val_xgb_df = train_df.loc[val_mask].reset_index(drop=True).copy()
    for col in categorical_features:
        val_xgb_df[col] = val_xgb_df[col].astype(str).fillna('missing').astype('category')
    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_pred = xgb_model.predict_proba(val_xgb_df[features])[:, 1]

    preds = {
        'CatBoost': cat_pred,
        'XGBoost': xgb_pred,
        'NeuralNetTorch': nn_torch_pred,
        'NeuralNetFastAI': nn_fastai_pred,
    }
    scores = {}
    for name, p in preds.items():
        _, bss = compute_bss(p, y_val)
        scores[name] = max(0, 100000 * bss)

    names = list(preds.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            mat.loc[a, b] = np.corrcoef(preds[a], preds[b])[0, 1]

    print("[개별 점수]")
    for n in names:
        print(f"  {n}: {scores[n]:.2f}")

    print("\n[전체 상관계수 행렬]")
    print(mat.round(4).to_string())


if __name__ == "__main__":
    main()
