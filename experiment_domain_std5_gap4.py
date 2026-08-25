"""Season-1 트랙맨 조인: std5(구속/회전수/IVB/HB/extension 표준편차) + gap4(패스트볼 vs
변화구 물리량 평균 차이), 수연님 레포(github.com/lg-first-penguin/sooyun) 확인 후 포팅
(2026-08-25 밤). 우리가 예전에 "season이 조인 키라 2025는 100% 매칭 실패"로 포기했던
트랙맨 물리량 문제의 구조적 해법 — season이 아니라 **season-1**로 조인해서 2025도
2024 트랙맨으로 항상 매칭 가능하게 만드는 방식.

CatBoost 전용 빠른 검증(단일시드, MLP 스킵) — 이유는 hand_matchup/no_both 때와 동일.
기준선 = experiment_final_bundle_baseline.py의 CatBoost 부분: half_2024 769.10 /
expand_2023 822.28 / 평균 795.69.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from feature_engineering import process_trackman_features_safe, add_engineered_features, apply_te_residual_features
from experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"
BASELINE = {"half_2024": 769.10, "expand_2023": 822.28}

MATCH_COLS_6 = ["inning", "top_bottom", "balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
MIN_N = 10  # 수연님 레포 기준, 그룹 표본 너무 적으면 std/gap 불안정


def build_std5_gap4_tables(df_trm):
    """trm(전체 트랙맨 이력)에서 (season, 6키) 단위 std5/gap4 테이블을 만듦.
    train.csv 인코딩(top_bottom: T/B, hand: 1/2)에 맞춰 변환."""
    trm = df_trm.copy()
    # process_trackman_features_safe가 작업용 df의 top_bottom을 T/B 문자열에서
    # {T:0, B:1} 정수로 이미 바꿔놓기 때문에, 여기서도 같은 정수 인코딩으로 맞춘다
    # (버그 수정: 원래 "T"/"B" 문자열로 매핑했다가 int64 vs object merge 에러 발생).
    trm["top_bottom"] = trm["top_bottom"].map({"Top": 0, "Bottom": 1})
    trm["pitcher_hand"] = trm["pitcher_hand"].map({"Right": 2, "Left": 1})
    trm["batter_hand"] = trm["batter_hand"].map({"Right": 2, "Left": 1})

    grp = trm.groupby(["season"] + MATCH_COLS_6)
    counts = grp.size().rename("n")

    std5 = grp[["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension"]].std()
    std5.columns = [f"tm_{c}_std" for c in std5.columns]
    std5 = std5.join(counts)
    std5 = std5[std5["n"] >= MIN_N].drop(columns="n").reset_index()

    fb = trm[trm["pitch_type_group"] == "fastball"].groupby(["season"] + MATCH_COLS_6)[
        ["rel_speed", "induced_vert_break", "horz_break", "spin_rate"]].mean()
    brk = trm[trm["pitch_type_group"] == "breaking"].groupby(["season"] + MATCH_COLS_6)[
        ["rel_speed", "induced_vert_break", "horz_break", "spin_rate"]].mean()
    gap4 = (fb - brk).dropna()
    gap4.columns = ["speed_gap_fast_break", "ivb_gap_fast_break", "hbreak_gap_fast_break", "spin_gap_fast_break"]
    gap4 = gap4.reset_index()

    return std5, gap4


def add_season1_join(df, std5, gap4):
    df = df.copy()
    df["match_season"] = df["season"] - 1
    n_before = len(df)
    df = df.merge(std5, left_on=["match_season"] + MATCH_COLS_6, right_on=["season"] + MATCH_COLS_6,
                   how="left", suffixes=("", "_std5drop"))
    df = df.merge(gap4, left_on=["match_season"] + MATCH_COLS_6, right_on=["season"] + MATCH_COLS_6,
                  how="left", suffixes=("", "_gap4drop"))
    assert len(df) == n_before, "merge로 행 개수가 변했습니다"
    drop_cols = [c for c in df.columns if c.endswith("_std5drop") or c.endswith("_gap4drop")]
    df = df.drop(columns=["match_season"] + drop_cols)
    return df


with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=50, verbose=False,
)


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df, tr_final


def fit_and_score(fold_df, tr_final, fold_train_mask, val_mask, y_val, features):
    cat_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")
    train_pool = Pool(cb_df.loc[fold_train_mask, features], cb_df.loc[fold_train_mask, TARGET], cat_features=cat_features)
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)
    model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    model.fit(train_pool, eval_set=val_pool)
    pred = model.predict_proba(val_pool)[:, 1]
    return score(pred, y_val)


def run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name, std5, gap4):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)
    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    te_prior = fold_df.loc[fold_train_mask, TARGET].mean()
    te_source = fold_df.loc[fold_train_mask]
    base_df = apply_te_residual_features(te_source, fold_df.copy(), te_prior)

    results = {}
    features_base = [c for c in base_df.columns if c not in ["row_id", TARGET]]
    t0 = time.time()
    results["baseline"] = fit_and_score(base_df, tr_final, fold_train_mask, val_mask, y_val, features_base)
    print(f"[{fold_name}][baseline] {results['baseline']:.2f} ({time.time()-t0:.1f}s)", flush=True)

    std5gap4_df = add_season1_join(base_df, std5, gap4)
    new_trm_cols = [c for c in std5gap4_df.columns if c.startswith("tm_") or c.endswith("_fast_break")]
    std5gap4_df[new_trm_cols] = std5gap4_df[new_trm_cols].fillna(std5gap4_df.loc[fold_train_mask, new_trm_cols].mean())
    features_std5gap4 = [c for c in std5gap4_df.columns if c not in ["row_id", TARGET]]
    t0 = time.time()
    results["std5_gap4"] = fit_and_score(std5gap4_df, tr_final, fold_train_mask, val_mask, y_val, features_std5gap4)
    print(f"[{fold_name}][std5_gap4] {results['std5_gap4']:.2f} ({time.time()-t0:.1f}s, 신규 {len(new_trm_cols)}개 피처)", flush=True)

    return results


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    print("std5/gap4 매칭 테이블 생성 중...", flush=True)
    std5, gap4 = build_std5_gap4_tables(df_trm)
    print(f"std5: {std5.shape}, gap4: {gap4.shape}", flush=True)

    all_results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        all_results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name, std5, gap4)

    print("\n=== 요약 (CatBoost 전용, std5+gap4) ===")
    for variant in ["baseline", "std5_gap4"]:
        vals = [all_results[f][variant] for f in all_results]
        avg = np.mean(vals)
        line = ", ".join(f"{f}={all_results[f][variant]:.2f}" for f in all_results)
        print(f"{variant}: {line} | 평균={avg:.2f}")


if __name__ == "__main__":
    main()
