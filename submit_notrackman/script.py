# script.py
"""제출용 독립 실행 추론 스크립트 (트랙맨 물리량 제외판).

`submit/script.py`와 동일하되, `process_trackman_features()`(트랙맨 물리량 12개,
season이 조인 키라 실제 채점 시즌엔 100% 매칭 실패 -> 학습/추론 분포 불일치)를
아예 빼고 원본 47개 컬럼(asof_* 포함) + 파생피처 + coarse 구종비중만 사용.

모델: CatBoost(F1 필터+coarse 구종비중, 트랙맨 물리량 미포함) + Tabular MLP(동일) 블렌드.
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


def add_coarse_pitchmix(df_main, df_trm):
    """experiment_pitchmix_coarse.py::compute_coarse_pitchmix + merge 로직 복제 (추론용).

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
    MODEL_PATH = "./model/final_retained_model_notrackman.pkl"
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

    # 피처 엔지니어링: 트랙맨 물리량 merge 없음 -> 파생피처 -> coarse 구종비중
    tr_final = add_engineered_features(df_test, league_success_mean)
    tr_final = add_coarse_pitchmix(tr_final, df_trm)

    drop_cols = [ID_COL, TARGET_COL]
    features = [c for c in tr_final.columns if c not in drop_cols]
    X_test = tr_final[features]

    # CatBoost 예측
    cat_preds = bundle["catboost_model"].predict_proba(X_test)[:, 1]

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

    alpha = bundle["alpha"]
    preds = alpha * cat_preds + (1 - alpha) * mlp_preds

    df_sub[TARGET_COL] = preds
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"추론 완료: {out_path}")


if __name__ == "__main__":
    main()
