"""시즌 진행분 방식을 ball_rate/strike_rate로 확장 (EXPERIMENTS.md §6-19 후속).

success_rate 버전은 시즌 마지막 행 자신의 실제 결과(control_success, 행 단위 라벨)로
정밀 보정 가능했지만, ball_rate/strike_rate는 그런 행 단위 라벨이 데이터에 없음
(data_description.md 확인 완료) -> 근사 방식(마지막 투구 보정 생략, 시즌당 최대 1구
오차) 사용. p_middle_season(지훈님, 같은 근사 방식)은 애매하게 기각됐었음 — ball_rate/
strike_rate가 다른 결과를 보이는지 확인.

기존 시즌진행분(success_rate만) 기준선 위에 추가: 775.45 (CatBoost 764.86/MLP 731.02).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, fit_preprocessing, apply_preprocessing, to_tensors,
    train_ensemble, predict_ensemble, embed_dim_for_cardinality,
)
from local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BLEND_ALPHA = 0.61

# (피처 프리픽스, id_col, n_col, rate_col) -- pitcher 전용(투수 쪽 비율만, 배터 대칭은 생략)
RATE_SPECS = [
    ("pitcher_ball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate"),
    ("pitcher_strike", "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate"),
]

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=50, verbose=False,
)


def build_rate_end_lookup(df):
    """success_rate 버전(build_season_end_lookup)과 동일 구조지만, 행 단위 라벨이
    없어 마지막 투구 보정 없이 근사(end_n/end_rate를 그대로 "시즌 시작 시점 값"으로
    사용, 시즌당 최대 1구 오차)."""
    tables = []
    for name, id_col, n_col, rate_col in RATE_SPECS:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate"]
        season_end["season"] = season_end["season"] + 1
        season_end.insert(0, "name", name)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_rate_progression_features(df, lookup):
    df = df.copy()
    for name, id_col, n_col, rate_col in RATE_SPECS:
        lut = lookup.loc[lookup["name"] == name, ["id", "season", "end_n", "end_rate"]]
        merged = df[[id_col, "season"]].merge(lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left")
        pre_n = merged["end_n"].fillna(0).values
        pre_count = (merged["end_n"] * merged["end_rate"]).round().fillna(0).values
        cum_n = df[n_col].values
        cum_count = np.round(df[n_col].values * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_count = np.maximum(cum_count - pre_count, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_count / season_n, np.nan)
        df[f"{name}_season_rate"] = season_rate
        df[f"{name}_season_rate_gap"] = season_rate - df[rate_col].values
    return df


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


def run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)  # 시즌진행분(success_rate) 포함
    fold_df = apply_rate_progression_features(fold_df, build_rate_end_lookup(train_df))  # ball/strike 추가

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)
    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    drop_cols = ["row_id", TARGET]
    features = [c for c in fold_df.columns if c not in drop_cols]
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")
    train_pool = Pool(cb_df.loc[fold_train_mask, features], cb_df.loc[fold_train_mask, TARGET], cat_features=cat_features)
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)

    t0 = time.time()
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    cat_model.fit(train_pool, eval_set=val_pool)
    cat_pred = cat_model.predict_proba(val_pool)[:, 1]
    cat_score = score(cat_pred, y_val)
    print(f"[{fold_name}] CatBoost: {cat_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    keep_cols = [c for c in fold_df.columns if c != "row_id"]
    mlp_df = fold_df[keep_cols].copy()
    mlp_features = [c for c in mlp_df.columns if c != TARGET]
    num_cols = [c for c in mlp_features if c not in CAT_COLS]
    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        mlp_df.loc[fold_train_mask].reset_index(drop=True), CAT_COLS, num_cols)
    val_proc = apply_preprocessing(
        mlp_df.loc[val_mask].reset_index(drop=True), CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t,
        seeds=ENSEMBLE_SEEDS, device=DEVICE, verbose=False,
    )
    mlp_pred = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, device=DEVICE)
    mlp_score = score(mlp_pred, y_val)
    print(f"[{fold_name}] MLP(7-seed): {mlp_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    blend_pred = BLEND_ALPHA * cat_pred + (1 - BLEND_ALPHA) * mlp_pred
    blend_score = score(blend_pred, y_val)
    print(f"[{fold_name}] Blend(alpha={BLEND_ALPHA}): {blend_score:.2f}", flush=True)
    return dict(cat=cat_score, mlp=mlp_score, blend=blend_score)


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    results = {}
    # half_2024/expand_2023는 이미 완료(rolling-origin 확장 검증, Track A 때와 동일 패턴)
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2021, 2022), include_half_2024=False):
        results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 (+ball/strike rate 시즌분해, expand_2021/2022) ===")
    for key in ["cat", "mlp", "blend"]:
        vals = [results[f][key] for f in results]
        print(f"{key}: " + ", ".join(f"{f}={results[f][key]:.2f}" for f in results) + f" | 평균={np.mean(vals):.2f}")
    print("\n참고: half_2024/expand_2023 결과(이미 완료) = "
          "CatBoost 754.99/767.07, MLP 775.91/738.21, Blend 791.43/783.13")


if __name__ == "__main__":
    main()
