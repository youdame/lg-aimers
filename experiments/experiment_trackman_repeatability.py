"""트랙맨 물리량을 "전년도 이전 causal 매칭 + 반복성(표준편차)"으로 재구성 (동섭님 방식
재현, EXPERIMENTS.md §7 참고). 원본 코드는 없고 노션 설명만 있어서 직접 새로 구현 --
검증 안 된 시도.

기존 트랙맨 물리량 64개(season==매칭이라 2025 실전에서 100% fallback)를 대체:
- 상황 6개(season 제외): inning, top_bottom, balls_before, strikes_before,
  pitcher_hand, batter_hand
- 각 행마다 "자기 시즌보다 엄격히 이전 시즌"의 트랙맨 데이터만 사용(causal, Track A와
  동일한 merge_asof 패턴) -> test.csv(2025)도 항상 매칭됨
- 평균 대신 표준편차(그 상황에서 투수 매커니즘이 얼마나 들쭉날쭉한지, "반복성")

CatBoost 단독으로 먼저 확인 (기존 트랙맨 물리량 64개 vs 이 방식 8개).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.feature_engineering import add_engineered_features
from experiments.experiment_pitchmix_coarse import compute_coarse_pitchmix, COARSE_COLS
from src.local_validation import iter_folds, score

DATA_DIR = "./data"
TARGET = "control_success"

PHYS_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
REPEAT_MATCH_COLS = ["inning", "top_bottom", "balls_before", "strikes_before", "pitcher_hand", "batter_hand"]

with open("./model_py311/best_catboost_pitchmix_hparams.json") as f:
    _tuned = json.load(f)["best_params"]

CATBOOST_PARAMS = dict(
    depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
    random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
    border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
    bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    random_seed=42, early_stopping_rounds=50, verbose=False,
)


def build_causal_agg(df_trm):
    """(REPEAT_MATCH_COLS, season)별 phys_cols의 누적 sum/sumsq/n (시즌순 정렬, cumsum)."""
    df_trm = df_trm.copy()
    df_trm["top_bottom"] = df_trm["top_bottom"].map({"Top": 0, "Bottom": 1}).astype(np.int64)
    for col in ["batter_hand", "pitcher_hand"]:
        df_trm[col] = df_trm[col].map({"Left": 1, "Right": 2}).astype(np.int64)

    group_keys = REPEAT_MATCH_COLS + ["season"]
    sum_df = df_trm.groupby(group_keys)[PHYS_COLS].sum().add_suffix("_sum")
    sq = df_trm[PHYS_COLS] ** 2
    sq[group_keys] = df_trm[group_keys]
    sqsum_df = sq.groupby(group_keys)[PHYS_COLS].sum().add_suffix("_sqsum")
    n_df = df_trm.groupby(group_keys).size().rename("n")

    agg = pd.concat([sum_df, sqsum_df, n_df], axis=1).reset_index().sort_values("season")
    sum_cols = [c + "_sum" for c in PHYS_COLS]
    sqsum_cols = [c + "_sqsum" for c in PHYS_COLS]
    agg[sum_cols] = agg.groupby(REPEAT_MATCH_COLS)[sum_cols].cumsum()
    agg[sqsum_cols] = agg.groupby(REPEAT_MATCH_COLS)[sqsum_cols].cumsum()
    agg["n"] = agg.groupby(REPEAT_MATCH_COLS)["n"].cumsum()
    return agg


def apply_causal_repeatability(df_main, agg):
    """merge_asof(backward, allow_exact_matches=False)로 자기 시즌보다 이전 누적치를 붙이고
    표준편차(반복성) 8개를 계산."""
    query = df_main[REPEAT_MATCH_COLS + ["season"]].reset_index()
    query_sorted = query.sort_values("season")
    merged = pd.merge_asof(
        query_sorted, agg, on="season", by=REPEAT_MATCH_COLS,
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("index")

    out = df_main.copy()
    n = merged["n"].fillna(0).values
    for col in PHYS_COLS:
        s = merged[f"{col}_sum"].fillna(0).values
        sq = merged[f"{col}_sqsum"].fillna(0).values
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s / n, np.nan)
            var = np.where(n > 1, sq / n - mean ** 2, np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        out[f"{col}_repeat_std"] = std
    # 결측(참조할 과거 시즌 없음, 주로 2019년 행) -> 학습셋 자체 평균으로 고정 fallback
    repeat_cols = [f"{c}_repeat_std" for c in PHYS_COLS]
    fill_values = out[repeat_cols].mean()
    out[repeat_cols] = out[repeat_cols].fillna(fill_values)
    return out


def build_base_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))
    train_df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df, df_trm, train_df


def run_fold(df, df_trm, train_df, train_mask, val_mask, fold_name):
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    fold_df = add_engineered_features(train_df, league_success_mean)

    pivoted = compute_coarse_pitchmix(df[train_mask.values], df_trm, is_train_split=True)
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    fold_df = pd.merge(fold_df, pivoted, on=COARSE_COLS, how="left")
    fold_df[new_cols] = fold_df[new_cols].fillna(fold_df[new_cols].mean())

    f1_drop = (fold_df["game_type"] == "F") & (fold_df["season"] <= 2022)
    fold_train_mask = train_mask & (~f1_drop)
    y_val = fold_df.loc[val_mask, TARGET].to_numpy()

    # causal repeatability std: trackman 집계는 train_mask 구간만 사용(검증 시즌 이후 트랙맨 제외)
    agg = build_causal_agg(df_trm[df_trm["season"].isin(fold_df.loc[fold_train_mask, "season"].unique())])
    fold_df["top_bottom"] = fold_df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    fold_df = apply_causal_repeatability(fold_df, agg)

    drop_cols = ["row_id", TARGET]
    features = [c for c in fold_df.columns if c not in drop_cols]
    cat_features = [c for c in fold_df.select_dtypes(include="object").columns if c != "row_id" and c in features]
    cb_df = fold_df.copy()
    for c in cat_features:
        cb_df[c] = cb_df[c].astype(str).fillna("missing")
    train_pool = Pool(cb_df.loc[fold_train_mask, features], cb_df.loc[fold_train_mask, TARGET], cat_features=cat_features)
    val_pool = Pool(cb_df.loc[val_mask, features], cb_df.loc[val_mask, TARGET], cat_features=cat_features)

    t0 = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS, iterations=1500)
    model.fit(train_pool, eval_set=val_pool)
    preds = model.predict_proba(val_pool)[:, 1]
    s = score(preds, y_val)
    print(f"[{fold_name}] CatBoost(+repeatability std): {s:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return s


def main():
    df, df_trm, train_df = build_base_df()
    fold_scores = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2023,)):
        fold_scores[fold_name] = run_fold(df, df_trm, train_df, train_mask, val_mask, fold_name)

    avg = sum(fold_scores.values()) / len(fold_scores)
    print("\n=== 요약 ===")
    for name, s in fold_scores.items():
        print(f"  {name}: {s:.2f}")
    print(f"  평균: {avg:.2f}")
    print("\n참고: 시즌진행분+구종비중+F1 기준선(트랙맨 물리량 season==매칭 포함) = "
          "half_2024 744.52 / expand_2023 785.19 / 평균 764.86")


if __name__ == "__main__":
    main()
