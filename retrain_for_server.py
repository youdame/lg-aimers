import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "./data"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]
ID_COLS_DROP = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]

test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
FEATURES = [c for c in test_cols if c != ID and c not in ID_COLS_DROP]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=FEATURES + [TARGET])
print("train:", train.shape)

train["count_pressure"] = train["strikes_before"] * train["balls_before"]
train["pitcher_batter_success"] = train["asof_pitcher_success_rate"] * train["asof_batter_success_rate"]
train["is_full_count"] = ((train["balls_before"] == 3) & (train["strikes_before"] == 2)).astype(int)
train["pitcher_is_new"] = train["asof_pitcher_success_rate"].isna().astype(int)
train["pitcher_recent5_gap"] = train["asof_pitcher_prev5_game_success_rate"] - train["asof_pitcher_success_rate"]
train["pitcher_success_squared"] = train["asof_pitcher_success_rate"] ** 2
train["fastball_success"] = train["asof_pitcher_fastball_rate"] * train["asof_pitcher_success_rate"]
train["batter_reliability"] = train["asof_batter_success_rate"] * (train["asof_batter_n"] > 10).astype(int)
train["pitcher_reverse_squared"] = train["asof_pitcher_reverse_rate"] ** 2
train["batter_success_squared"] = train["asof_batter_success_rate"] ** 2
train["pitcher_reliability"] = train["asof_pitcher_success_rate"] * (train["asof_pitcher_n"] > 20).astype(int)

new_features = [
    "count_pressure", "pitcher_batter_success", "is_full_count", "pitcher_is_new",
    "pitcher_recent5_gap", "pitcher_success_squared", "fastball_success", "batter_reliability",
    "pitcher_reverse_squared", "batter_success_squared", "pitcher_reliability",
]
FEATURES = FEATURES + new_features
NUM_COLS = NUM_COLS + new_features

trackman = pd.read_csv("data/trackman_history.csv", encoding="utf-8-sig", usecols=["season", "spin_rate", "rel_speed"])
season_stats = trackman.groupby("season")[["spin_rate", "rel_speed"]].mean()
train["league_spin_avg"] = train["season"].map(season_stats["spin_rate"])
train["league_speed_avg"] = train["season"].map(season_stats["rel_speed"])
FEATURES = FEATURES + ["league_spin_avg", "league_speed_avg"]
NUM_COLS = NUM_COLS + ["league_spin_avg", "league_speed_avg"]

print("최종 피처 개수:", len(FEATURES))

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUM_COLS),
])

model = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42, C=3.16e-06)),
])

# 검증: 2022 제외 학습 -> 2024 검증 (버전 차이로 인한 결과 재현성 확인용)
TRAIN_SEASONS = [2019, 2020, 2021, 2023]
is_val = train["season"] == 2024
is_train = train["season"].isin(TRAIN_SEASONS)
X_train, y_train = train.loc[is_train, FEATURES], train.loc[is_train, TARGET]
X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]

t = time.time()
model.fit(X_train, y_train)
print(f"검증용 학습 완료 :: {time.time() - t:.1f}s")

val_pred = model.predict_proba(X_val)[:, 1]
r = y_val.mean()
brier = ((val_pred - y_val) ** 2).mean()
baseline_brier = r * (1 - r)
score = max(0, 100000 * (1 - brier / baseline_brier))
print(f"[재현성 확인] Validation Score: {score:.2f}  (기존 Python 3.13 환경 기준값: 424.93)")

# 최종 제출용: 2022 제외 전체(2019,2020,2021,2023,2024)로 재학습 -> 저장
FINAL_SEASONS = [2019, 2020, 2021, 2023, 2024]
is_final_train = train["season"].isin(FINAL_SEASONS)

t = time.time()
model.fit(train.loc[is_final_train, FEATURES], train.loc[is_final_train, TARGET])
print(f"최종 재학습 완료 :: {time.time() - t:.1f}s (n={is_final_train.sum()})")

os.makedirs("./model_py311", exist_ok=True)
joblib.dump(model, "./model_py311/lr.pkl", compress=3)
print("저장 완료: ./model_py311/lr.pkl")
