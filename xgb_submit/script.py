# script.py
import os
import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL_FEATURES = ["game_type", "base_state"]


def brier_eval(y_true, y_pred):
    """xgb.pkl 안에 pickle된 커스텀 eval_metric 참조를 풀기 위해 학습 때와 동일 이름으로 재정의."""
    return float(((y_pred - y_true) ** 2).mean())


def process_trackman_features_safe(df_main, df_trm):
    """트랙맨 정밀조인. (lgaimers9/code/train.py의 로직과 동일, code/ 미의존을 위해 복제)"""
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    match_cols = [c for c in df_main_copy.columns if (c in df_trm_copy.columns) and c != "row_id"]

    grouped = df_trm_copy.groupby(match_cols + ["pitch_type_group", "auto_pitch_type"])
    grouped_phase1 = grouped[["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]].agg(["mean", "std"])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [col for col in grouped_phase2.columns if "std" in col]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ["_".join(col).strip("_") for col in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns="auto_pitch_type")
    grouped_phase3 = grouped_phase3.groupby(match_cols + ["pitch_type_group"]).agg(["mean"])

    pivoted = grouped_phase3.unstack(level="pitch_type_group")
    pivoted.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)

    df_main_copy["top_bottom"] = df_main_copy["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    tm_final["top_bottom"] = tm_final["top_bottom"].map({"Top": 0, "Bottom": 1}).astype(np.int64)

    for col in ["batter_hand", "pitcher_hand"]:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({"Left": 1, "Right": 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how="left")
    new_feature_cols = [col for col in tm_final.columns if col not in match_cols]
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(0)

    return tr_final


def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처. (lgaimers9/code/train.py와 동일 정의)"""
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
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].std(axis=1)

    df["matchup"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]

    pressure_signal = df["li"] * ((df["strikes_before"] >= 2) | (df["balls_before"] >= 3)).astype(np.int64)
    df["count_pressure"] = df["pitcher_relative_success"] * pressure_signal

    return df


def main():
    DATA_DIR = "./data"
    MODEL_PATH = "./model/xgb.pkl"
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

    # xgb.pkl 학습 시 사용한 것과 동일하게: season<2024 데이터 기준 리그 평균 (retrain_xgb.py와 동일 기준)
    league_success_mean = df_train_raw.loc[df_train_raw["season"] < 2024, TARGET_COL].mean()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    tr_final = process_trackman_features_safe(df_test, df_trm)
    tr_final = add_engineered_features(tr_final, league_success_mean)

    drop_cols = [ID_COL, TARGET_COL]
    features = [col for col in tr_final.columns if col not in drop_cols]

    for col in CATEGORICAL_FEATURES:
        if col in tr_final.columns:
            tr_final[col] = tr_final[col].astype(str).fillna("missing").astype("category")

    X_test = tr_final[features]

    preds = model.predict_proba(X_test)[:, 1]

    df_sub[TARGET_COL] = preds

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"추론 및 제출용 파일 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
