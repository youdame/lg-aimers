# script.py
"""제출용 독립 실행 추론 스크립트. code/ 패키지에 의존하지 않고 필요한 로직을 전부 복제합니다
(대회 규칙 관례: submit.zip에는 code/ 패키지가 포함되지 않음).

모델: CatBoost(재튜닝+coarse 구종비중) + Tabular MLP(coarse 구종비중, 7-seed) 블렌드.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]
COARSE_COLS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]


class TabularMLP(nn.Module):
    """mlp_model.py::TabularMLP 와 동일 구조 (독립 실행을 위해 복제)."""

    def __init__(self, num_numeric_feats, cat_dims, embed_dims, hidden1=128, hidden2=64, dropout=0.3):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        total_input_dim = sum(self.embed_dims) + num_numeric_feats
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_all = torch.cat([x_embed, x_num], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def process_trackman_features(df_main, df_trm):
    """feature_engineering.py::process_trackman_features_safe 와 동일 (추론 시점이라 시간필터 없음, 복제).

    결측치는 test.csv 유래 통계가 아니라 trackman_history.csv(공식 데이터) 자체 평균인
    고정값으로 채운다 (평가 데이터 독립 예측 원칙 준수 — CLAUDE.md 참고).
    """
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    match_cols = [c for c in df_main_copy.columns if (c in df_trm_copy.columns) and c != "row_id"]

    grouped = df_trm_copy.groupby(match_cols + ["pitch_type_group", "auto_pitch_type"])
    grouped_phase1 = grouped[["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]].agg(["mean", "std"])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [c for c in grouped_phase2.columns if "std" in c]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ["_".join(c).strip("_") for c in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns="auto_pitch_type")
    grouped_phase3 = grouped_phase3.groupby(match_cols + ["pitch_type_group"]).agg(["mean"])

    pivoted = grouped_phase3.unstack(level="pitch_type_group")
    pivoted.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)

    df_main_copy["top_bottom"] = df_main_copy["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    tm_final["top_bottom"] = tm_final["top_bottom"].map({"Top": 0, "Bottom": 1}).astype(np.int64)
    for col in ["batter_hand", "pitcher_hand"]:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({"Left": 1, "Right": 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how="left")
    new_feature_cols = [c for c in tm_final.columns if c not in match_cols]
    # tm_final(트랙맨 집계 결과, 공식 데이터 유래)의 평균 — test.csv와 무관한 고정값
    fill_values = tm_final[new_feature_cols].mean()
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(fill_values)
    return tr_final


def add_engineered_features(df, league_success_mean):
    """feature_engineering.py::add_engineered_features 와 동일 (복제)."""
    df = df.copy()
    df["pitcher_recent1_gap"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["pitcher_recent3_gap"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["pitcher_recent5_gap"] = df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["pitcher_relative_success"] = df["asof_pitcher_success_rate"] - league_success_mean
    df["count_diff"] = df["strikes_before"] - df["balls_before"]
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(np.int64)
    df["pitcher_count_advantage_raw"] = df["asof_pitcher_success_rate"] * df["count_diff"]
    df["pitcher_count_advantage_rel"] = df["pitcher_relative_success"] * df["count_diff"]
    df["pitcher_trend"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    df["pitcher_consistency"] = df[[
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    ]].std(axis=1)
    df["matchup"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    pressure_signal = df["li"] * ((df["strikes_before"] >= 2) | (df["balls_before"] >= 3)).astype(np.int64)
    df["count_pressure"] = df["pitcher_relative_success"] * pressure_signal
    return df


def apply_season_progression_features(df, lookup):
    """feature_engineering.py::apply_season_progression_features 와 동일 (복제).
    lookup은 model/final_retained_model.pkl에 정적 동봉된 것을 그대로 씀 — test.csv
    자기 자신으로는 lookup을 만들 수 없음(과거 시즌 행/control_success 없음, 평가
    데이터 독립 예측 원칙)."""
    df = df.copy()
    specs = [
        ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
    ]
    for role, id_col, n_col, rate_col in specs:
        lut = lookup.loc[lookup["role"] == role, ["id", "season", "end_n", "end_rate", "end_success"]]
        merged = df[[id_col, "season"]].merge(
            lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left",
        )
        pre_n = (merged["end_n"] + 1).fillna(0).values
        pre_success = ((merged["end_n"] * merged["end_rate"]).round().fillna(0) + merged["end_success"].fillna(0)).values
        cum_n = df[n_col].values
        cum_success = np.round(df[n_col].values * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_success = np.maximum(cum_success - pre_success, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_success / season_n, np.nan)
        df[f"{role}_season_n"] = season_n
        df[f"{role}_season_success_count"] = season_success
        df[f"{role}_season_success_rate"] = season_rate
        df[f"{role}_season_rate_gap"] = season_rate - df[rate_col].values
    return df


RATE_PROGRESSION_SPECS = [
    ("pitcher_reverse", "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate"),
]


def apply_rate_progression_features(df, lookup):
    """feature_engineering.py::apply_rate_progression_features 와 동일 (복제). lookup은
    model/final_retained_model.pkl에 정적 동봉된 것을 그대로 씀 — test.csv 자기
    자신으로는 lookup을 만들 수 없음(평가 데이터 독립 예측 원칙)."""
    df = df.copy()
    for name, id_col, n_col, rate_col in RATE_PROGRESSION_SPECS:
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


TE_K_SMOOTH = 200
TE_AXES = [
    ("te_p_cnt", ["pitcher_id", "balls_before", "strikes_before"], "p_main"),
    ("te_p_bhand", ["pitcher_id", "batter_hand"], "p_main"),
    ("te_p_run", ["pitcher_id", "num_runners_on"], "p_main"),
    ("te_p_inn", ["pitcher_id", "inning"], "p_main"),
    ("te_b_cnt", ["batter_id", "balls_before", "strikes_before"], "b_main"),
]
TE_MAIN_AXES = [("p_main", ["pitcher_id"]), ("b_main", ["batter_id"])]
TE_RESIDUAL_COLS = [f"{name}_res" for name, *_ in TE_AXES] + ["te_covered"]


def causal_smoothed_te_encode(source_df, query_df, group_cols, prior, k=TE_K_SMOOTH):
    """feature_engineering.py::causal_smoothed_te_encode 와 동일 (복제). source_df는
    model 번들에 정적 동봉된 te_source(F1 필터 적용된 train.csv 유래, 공식 데이터) —
    test.csv와 무관한 고정 데이터."""
    agg = source_df.groupby(group_cols + ["season"])["control_success"].agg(["sum", "count"]).reset_index()
    agg = agg.sort_values("season")
    agg["cum_sum"] = agg.groupby(group_cols)["sum"].cumsum()
    agg["cum_n"] = agg.groupby(group_cols)["count"].cumsum()
    agg = agg[group_cols + ["season", "cum_sum", "cum_n"]].sort_values("season")

    query = query_df[group_cols + ["season"]].reset_index()
    query_sorted = query.sort_values("season")
    merged = pd.merge_asof(
        query_sorted, agg, on="season", by=group_cols,
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("index")
    cum_sum = merged["cum_sum"].fillna(0.0).values
    cum_n = merged["cum_n"].fillna(0.0).values
    enc = (cum_sum + prior * k) / (cum_n + k)
    covered = (cum_n > 0).astype(np.int64)
    return enc, covered


def apply_te_residual_features(source_df, query_df, prior):
    """feature_engineering.py::apply_te_residual_features 와 동일 (복제). CatBoost
    전용 — MLP num_cols에는 애초에 포함 안 됨(학습 시점에 고정된 mlp_bundle["num_cols"])."""
    query_df = query_df.copy()
    mains = {}
    covered_any = np.zeros(len(query_df), dtype=np.int64)
    for name, group_cols in TE_MAIN_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        mains[name] = enc
        covered_any = np.maximum(covered_any, covered)
    for name, group_cols, main_key in TE_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        query_df[f"{name}_res"] = enc - mains[main_key]
        covered_any = np.maximum(covered_any, covered)
    query_df["te_covered"] = covered_any
    return query_df


def add_pressure_flag(df):
    """feature_engineering.py::add_pressure_flag 와 동일 (복제). CatBoost 전용 —
    MLP num_cols에는 애초에 포함 안 됨(학습 시점에 고정된 mlp_bundle["num_cols"])."""
    df = df.copy()
    runner_on_3b = df["base_state"].str[2] == "3"
    pressure = (runner_on_3b & (df["outs_before"] < 2)).astype(np.int64)
    df["scoring_pos_pressure_flag"] = pressure
    df["scoring_pos_pressure_li"] = pressure * df["li"]
    return df


def add_coarse_pitchmix(df_main, df_trm):
    """experiment_pitchmix_coarse.py::compute_coarse_pitchmix + merge 로직 복제 (추론용, 시간필터 없음).

    결측치는 trackman_history.csv(공식 데이터) 유래 고정값으로 채운다 (평가 데이터 독립
    예측 원칙 준수 — CLAUDE.md 참고).
    """
    df_trm_copy = df_trm.copy()
    df_trm_copy["pitcher_hand"] = df_trm_copy["pitcher_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)
    df_trm_copy["batter_hand"] = df_trm_copy["batter_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)

    counts = df_trm_copy.groupby(COARSE_COLS + ["pitch_type_group"]).size().reset_index(name="n")
    totals = counts.groupby(COARSE_COLS)["n"].transform("sum")
    counts["pitchmix_ratio"] = counts["n"] / totals
    pivoted = counts.pivot_table(index=COARSE_COLS, columns="pitch_type_group", values="pitchmix_ratio", fill_value=0).reset_index()
    pivoted.columns = [f"coarse_pitchmix_{c}" if c not in COARSE_COLS else c for c in pivoted.columns]

    merged = pd.merge(df_main, pivoted, on=COARSE_COLS, how="left")
    new_cols = [c for c in pivoted.columns if c not in COARSE_COLS]
    # pivoted(트랙맨 집계 결과, 공식 데이터 유래)의 평균 — test.csv와 무관한 고정값
    fill_values = pivoted[new_cols].mean()
    merged[new_cols] = merged[new_cols].fillna(fill_values)
    return merged


def apply_mlp_preprocessing(df, cat_cols, num_cols, cat_encoder, num_imputer, num_scaler):
    df = df.copy()
    df[cat_cols] = df[cat_cols].astype(str)
    df[cat_cols] = cat_encoder.transform(df[cat_cols]) + 1
    df[num_cols] = num_imputer.transform(df[num_cols])
    df[num_cols] = num_scaler.transform(df[num_cols])
    return df


def main():
    DATA_DIR = "./data"
    MODEL_PATH = "./model/final_retained_model.pkl"
    OUTPUT_DIR = "./output"

    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
    trackman_path = os.path.join(DATA_DIR, "trackman_history.csv")
    train_path = os.path.join(DATA_DIR, "train.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {test_path}")

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    df_sub = pd.read_csv(sample_sub_path, encoding="utf-8-sig")
    df_trm = pd.read_csv(trackman_path, encoding="utf-8-sig")
    df_train_raw = pd.read_csv(train_path, encoding="utf-8-sig")
    league_success_mean = df_train_raw[TARGET_COL].mean()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    mlp_bundle = bundle["mlp_bundle"]

    # 피처 엔지니어링: 트랙맨 merge -> 파생피처 -> coarse 구종비중 -> 시즌 진행분 -> Track A -> 압박 플래그
    tr_final = process_trackman_features(df_test, df_trm)
    tr_final = add_engineered_features(tr_final, league_success_mean)
    tr_final = add_coarse_pitchmix(tr_final, df_trm)
    tr_final = apply_season_progression_features(tr_final, bundle["season_end_lookup"])
    tr_final = apply_rate_progression_features(tr_final, bundle["rate_end_lookup"])
    tr_final = apply_te_residual_features(bundle["te_source"], tr_final, bundle["te_prior"])
    # 득점권 압박 플래그는 근거가 약해서(노이즈 수준) 최종적으로 제외 -- 함수는 남겨둠

    drop_cols = [ID_COL, TARGET_COL]
    features = [c for c in tr_final.columns if c not in drop_cols]
    X_test = tr_final[features]

    # CatBoost 예측 (5-seed 앙상블 평균, 2026-08-25 채택)
    cat_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in bundle["catboost_models"]], axis=0)

    # MLP 예측 (7-seed 평균)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_proc = apply_mlp_preprocessing(
        X_test, mlp_bundle["cat_cols"], mlp_bundle["num_cols"],
        mlp_bundle["cat_encoder"], mlp_bundle["num_imputer"], mlp_bundle["num_scaler"],
    )
    X_cat = torch.tensor(X_proc[mlp_bundle["cat_cols"]].values.astype(np.float32)).to(device)
    X_num = torch.tensor(X_proc[mlp_bundle["num_cols"]].values.astype(np.float32)).to(device)

    preds_list = []
    with torch.no_grad():
        for member in mlp_bundle["members"]:
            model = TabularMLP(
                num_numeric_feats=len(mlp_bundle["num_cols"]),
                cat_dims=mlp_bundle["cat_dims"],
                embed_dims=mlp_bundle["embed_dims"],
            ).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            preds_list.append(model(X_cat, X_num).cpu().numpy())
    mlp_preds = np.mean(preds_list, axis=0)

    # 2-입력 로지스틱회귀 스태킹 (고정 alpha 블렌드 대체, 2026-08-25 채택)
    m = bundle["meta_model"]
    logit = m["w_cat"] * cat_preds + m["w_mlp"] * mlp_preds + m["intercept"]
    preds = 1.0 / (1.0 + np.exp(-logit))

    df_sub[TARGET_COL] = preds
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"추론 완료: {out_path}")


if __name__ == "__main__":
    main()
