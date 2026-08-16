# script.py
"""제출용 독립 실행 추론 스크립트. code/ 패키지에 의존하지 않고 필요한 로직을 전부 복제합니다
(대회 규칙 관례: submit.zip에는 code/ 패키지가 포함되지 않음).

모델: CatBoost(재튜닝+coarse 구종비중) + Tabular MLP(coarse 구종비중, 7-seed) 블렌드.
"""
import math
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


class QuantileEmbedding(nn.Module):
    """mlp_model.py::QuantileEmbedding 와 동일 구조 (독립 실행을 위해 복제).
    수치형 피처별 quantile 기반 piecewise-linear 인코딩(PLE) + 피처별 독립 Linear + ReLU."""

    def __init__(self, bin_edges, d_embed=8, use_relu=True):
        super().__init__()
        self.register_buffer("edges", bin_edges)
        num_numeric, n_bins_plus1 = bin_edges.shape
        n_bins = n_bins_plus1 - 1
        self.num_numeric = num_numeric
        self.n_bins = n_bins
        self.use_relu = use_relu
        self.weight = nn.Parameter(torch.empty(num_numeric, n_bins, d_embed))
        self.bias = nn.Parameter(torch.zeros(num_numeric, d_embed))
        bound = 1.0 / math.sqrt(n_bins)
        nn.init.uniform_(self.weight, -bound, bound)

    def encode(self, x_num):
        left = self.edges[:, :-1].unsqueeze(0)
        right = self.edges[:, 1:].unsqueeze(0)
        x = x_num.unsqueeze(-1)
        width = (right - left).clamp_min(1e-6)
        frac = (x - left) / width
        return frac.clamp(0.0, 1.0)

    def forward(self, x_num):
        p = self.encode(x_num)
        e = torch.einsum("bnf,nfd->bnd", p, self.weight) + self.bias
        if self.use_relu:
            e = torch.relu(e)
        return e.reshape(e.shape[0], -1)


class TabularMLP(nn.Module):
    """mlp_model.py::TabularMLP 와 동일 구조 (독립 실행을 위해 복제).

    `bin_edges`가 주어지면 수치형 입력을 QuantileEmbedding(PLE)으로 인코딩하고,
    `bin_edges=None`이면 예전처럼 표준화된 수치형을 그대로 concat한다(하위 호환)."""

    def __init__(self, num_numeric_feats, cat_dims, embed_dims, hidden1=128, hidden2=64, dropout=0.3,
                 bin_edges=None, quantile_d=8):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        if bin_edges is not None:
            self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d)
            numeric_out_dim = bin_edges.shape[0] * quantile_d
        else:
            self.quantile = None
            numeric_out_dim = num_numeric_feats
        total_input_dim = sum(self.embed_dims) + numeric_out_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_num_enc = self.quantile(x_num) if self.quantile is not None else x_num
        x_all = torch.cat([x_embed, x_num_enc], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def _aggregate_trackman(df_trm_copy, group_cols):
    """group_cols 기준으로 트랙맨 물리량 8개를 구종군별 mean/std로 집계+피벗한 표를 반환.
    (feature_engineering.py::process_trackman_features_safe 의 집계 로직과 동일)"""
    grouped = df_trm_copy.groupby(group_cols + ["pitch_type_group", "auto_pitch_type"])
    grouped_phase1 = grouped[["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]].agg(["mean", "std"])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [c for c in grouped_phase2.columns if "std" in c]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ["_".join(c).strip("_") for c in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns="auto_pitch_type")
    grouped_phase3 = grouped_phase3.groupby(group_cols + ["pitch_type_group"]).agg(["mean"])

    pivoted = grouped_phase3.unstack(level="pitch_type_group")
    pivoted.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in pivoted.columns]
    result = pivoted.reset_index()
    return result.fillna(0)


def process_trackman_features(df_main, df_trm):
    """feature_engineering.py::process_trackman_features_safe 와 동일하되(추론 시점이라
    시간필터 없음, 복제), 결측치 처리는 3단계로 강화:

    1차: season 포함 10개 컬럼으로 매칭(학습 때와 동일 정밀도).
    2차(fallback): 2025 test처럼 season이 trackman_history.csv 범위(2019~2024) 밖이라
        1차 매칭이 100% 실패하는 상황을 위해, season을 뺀 9개 컬럼으로 재매칭한
        "상황별" 값으로 채움 (원작자 노트: "2025년 실제 제출시엔 9개로 매칭해야함").
        학습 파이프라인(`feature_engineering.py`)은 그대로 10개를 쓰고, 이 fallback은
        추론에서만 적용됨.
    3차(최종 보정): 그래도 결측이면(9개 조합조차 트랙맨에 없는 극희귀 케이스) 전체 평균으로
        채움. 전부 trackman_history.csv(공식 데이터) 유래 고정값이라 "평가 데이터 독립
        예측 원칙"을 그대로 준수함.
    """
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    match_cols = [c for c in df_main_copy.columns if (c in df_trm_copy.columns) and c != "row_id"]
    fallback_cols = [c for c in match_cols if c != "season"]

    tm_final = _aggregate_trackman(df_trm_copy, match_cols)
    tm_fallback = _aggregate_trackman(df_trm_copy, fallback_cols)

    df_main_copy["top_bottom"] = df_main_copy["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    for tbl in (tm_final, tm_fallback):
        tbl["top_bottom"] = tbl["top_bottom"].map({"Top": 0, "Bottom": 1}).astype(np.int64)
        for col in ["batter_hand", "pitcher_hand"]:
            if col in tbl.columns:
                tbl[col] = tbl[col].map({"Left": 1, "Right": 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how="left")
    new_feature_cols = [c for c in tm_final.columns if c not in match_cols]

    # 2차: 1차 매칭 실패 행만 season 제외(9개) 상황별 fallback으로 채움
    fb_merged = pd.merge(df_main_copy[fallback_cols], tm_fallback, on=fallback_cols, how="left")
    for col in new_feature_cols:
        missing = tr_final[col].isna()
        if missing.any():
            tr_final.loc[missing, col] = fb_merged.loc[missing, col].values

    # 3차: 그래도 남은 결측은 전체 평균(트랙맨 데이터 자체 유래 고정값)으로 최종 보정
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
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.int64)
    df["same_hand_advantage"] = df["pitcher_relative_success"] * df["same_hand"]
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

    # 피처 엔지니어링: 트랙맨 merge -> 파생피처 -> coarse 구종비중
    tr_final = process_trackman_features(df_test, df_trm)
    tr_final = add_engineered_features(tr_final, league_success_mean)
    tr_final = add_coarse_pitchmix(tr_final, df_trm)

    drop_cols = [ID_COL, TARGET_COL]
    features = [c for c in tr_final.columns if c not in drop_cols]
    X_test = tr_final[features]

    # CatBoost 예측: same_hand/same_hand_advantage는 CatBoost 학습 때 제외한 피처라
    # (2023+2024 이중검증에서 손해로 나옴, 2026-08-16) 예측 시에도 동일하게 제외해야 함.
    catboost_exclude_cols = ["same_hand", "same_hand_advantage"]
    catboost_features = [c for c in features if c not in catboost_exclude_cols]
    cat_preds = bundle["catboost_model"].predict_proba(X_test[catboost_features])[:, 1]

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
                hidden1=mlp_bundle.get("hidden1", 128),
                hidden2=mlp_bundle.get("hidden2", 64),
                dropout=mlp_bundle.get("dropout", 0.3),
                bin_edges=mlp_bundle.get("bin_edges"),
                quantile_d=mlp_bundle.get("quantile_d", 8),
            ).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            preds_list.append(model(X_cat, X_num).cpu().numpy())
    mlp_preds = np.mean(preds_list, axis=0)

    alpha = bundle["alpha"]
    preds = alpha * cat_preds + (1 - alpha) * mlp_preds

    df_sub[TARGET_COL] = preds
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"추론 완료: {out_path}")


if __name__ == "__main__":
    main()
