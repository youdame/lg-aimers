"""트리 모델(CatBoost/XGBoost)과 완전히 다른 방식인 MLP(신경망)를 하나 학습시켜서
1) MLP 단독 성능이 트리 대비 얼마나 낮은지
2) CatBoost/XGBoost 예측값과 상관계수가 실제로 더 낮은지 (다양성 가설 검증)
3) 3개 모델(Cat+XGB+MLP) 블렌드가 2개 블렌드(+2.88)보다 더 이득인지
를 직접 확인한다. sklearn MLPClassifier 사용 (torch/AutoGluon 미설치 환경).
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
    numeric_features = [c for c in features if c not in categorical_features]

    train_split = train_df.loc[train_mask].reset_index(drop=True)
    val_split = train_df.loc[val_mask].reset_index(drop=True)
    y_train = train_split[TARGET].values
    y_val = val_split[TARGET].values

    # --- MLP용 전처리: 범주형은 원핫, 결측은 평균대체, 스케일링 (MLP는 이 세 개 다 필수) ---
    print("[MLP] 전처리 (원핫 인코딩 + 결측 평균대체 + 표준화)...", flush=True)
    train_num = train_split[numeric_features].copy()
    val_num = val_split[numeric_features].copy()

    imputer = SimpleImputer(strategy='mean')
    train_num_imp = imputer.fit_transform(train_num)
    val_num_imp = imputer.transform(val_num)

    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_num_imp)
    val_num_scaled = scaler.transform(val_num_imp)

    train_cat = pd.get_dummies(train_split[categorical_features].astype(str), dummy_na=True)
    val_cat = pd.get_dummies(val_split[categorical_features].astype(str), dummy_na=True)
    val_cat = val_cat.reindex(columns=train_cat.columns, fill_value=0)

    X_train_mlp = np.hstack([train_num_scaled, train_cat.values.astype(np.float32)])
    X_val_mlp = np.hstack([val_num_scaled, val_cat.values.astype(np.float32)])
    print(f"[MLP] 입력 shape: train={X_train_mlp.shape}, val={X_val_mlp.shape}", flush=True)

    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        alpha=1e-3,
        batch_size=2048,
        learning_rate_init=1e-3,
        max_iter=40,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train_mlp, y_train)
    print(f"[MLP] 학습 완료: {time.time()-t0:.1f}s, n_iter={mlp.n_iter_}", flush=True)

    mlp_pred = mlp.predict_proba(X_val_mlp)[:, 1]
    mlp_brier, mlp_bss = compute_bss(mlp_pred, y_val)
    mlp_score = max(0, 100000 * mlp_bss)
    print(f"MLP 단독: BSS={mlp_bss:.5f} (score={mlp_score:.2f})", flush=True)

    # --- 기존 CatBoost / XGBoost 예측 불러오기 ---
    val_cat_df = val_split.copy()
    for col in categorical_features:
        val_cat_df[col] = val_cat_df[col].astype(str).fillna('missing')
    catboost_model = joblib.load(CATBOOST_MODEL_PATH)
    cat_pool = Pool(data=val_cat_df[features], cat_features=categorical_features)
    cat_pred = catboost_model.predict_proba(cat_pool)[:, 1]
    cat_brier, cat_bss = compute_bss(cat_pred, y_val)
    print(f"CatBoost 단독: BSS={cat_bss:.5f} (score={max(0,100000*cat_bss):.2f})", flush=True)

    val_xgb_df = val_split.copy()
    for col in categorical_features:
        val_xgb_df[col] = val_xgb_df[col].astype(str).fillna('missing').astype('category')
    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_pred = xgb_model.predict_proba(val_xgb_df[features])[:, 1]
    xgb_brier, xgb_bss = compute_bss(xgb_pred, y_val)
    print(f"XGBoost  단독: BSS={xgb_bss:.5f} (score={max(0,100000*xgb_bss):.2f})", flush=True)

    print("\n[상관계수]")
    print(f"  CatBoost vs XGBoost : {np.corrcoef(cat_pred, xgb_pred)[0,1]:.4f}  (참고: 기존 실측치)")
    print(f"  CatBoost vs MLP     : {np.corrcoef(cat_pred, mlp_pred)[0,1]:.4f}")
    print(f"  XGBoost  vs MLP     : {np.corrcoef(xgb_pred, mlp_pred)[0,1]:.4f}")

    print("\n[2-way 블렌드: alpha*CatBoost + (1-alpha)*MLP]")
    best_alpha, best_bss = None, -np.inf
    for alpha in np.arange(0.0, 1.01, 0.01):
        blend = alpha * cat_pred + (1 - alpha) * mlp_pred
        _, bss = compute_bss(blend, y_val)
        if bss > best_bss:
            best_bss, best_alpha = bss, alpha
    best_score = max(0, 100000 * best_bss)
    print(f"  최적 alpha={best_alpha:.2f} -> score={best_score:.2f} (CatBoost 단독 대비 {best_score - max(0,100000*cat_bss):+.2f})")

    print("\n[3-way 블렌드: w1*CatBoost + w2*XGBoost + w3*MLP] (그리드서치, step=0.05)")
    best_w, best3_bss = None, -np.inf
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = 1 - w1 - w2
            blend = w1 * cat_pred + w2 * xgb_pred + w3 * mlp_pred
            _, bss = compute_bss(blend, y_val)
            if bss > best3_bss:
                best3_bss, best_w = bss, (w1, w2, w3)
    best3_score = max(0, 100000 * best3_bss)
    print(f"  최적 weights (Cat,XGB,MLP)={best_w} -> score={best3_score:.2f}")

    print("\n" + "=" * 60)
    print("요약")
    print(f"  CatBoost 단독          : {max(0,100000*cat_bss):.2f}")
    print(f"  XGBoost  단독          : {max(0,100000*xgb_bss):.2f}")
    print(f"  MLP      단독          : {mlp_score:.2f}")
    print(f"  Cat+XGB 2-way (기존)   : 821.42 (+2.88)")
    print(f"  Cat+MLP 2-way          : {best_score:.2f} ({best_score - max(0,100000*cat_bss):+.2f})")
    print(f"  Cat+XGB+MLP 3-way      : {best3_score:.2f} ({best3_score - max(0,100000*cat_bss):+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
