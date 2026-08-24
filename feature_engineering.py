# lgaimers9/code/train.py의 CatBoost 파이프라인(BSS 0.00819 / 환산 818.54점)에서
# 그대로 옮겨온 피처 엔지니어링. XGBoost 스크립트(retrain_xgb.py, tune_xgb.py)에서
# 공유해서 씁니다. 로직은 원본과 동일하게 유지 (수정 없이 그대로 이식).
import numpy as np
import pandas as pd


def process_trackman_features_safe(df_main, df_trm, is_train_split=True):
    """타임 리크가 차단된 10-Key 상황 지문 기반 트랙맨 전처리 결합 엔진"""
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    if is_train_split:
        max_season = df_main_copy['season'].max()
        max_month = df_main_copy[df_main_copy['season'] == max_season]['game_month'].max()
        future_mask = (df_trm_copy['season'] > max_season) | \
                      ((df_trm_copy['season'] == max_season) & (df_trm_copy['game_month'] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)
        print(f"⏳ [Time Filter] {max_season}년 {max_month}월 이전의 트랙맨 데이터만 잘라내어 피처를 산출합니다.")

    match_cols = [dfc for dfc in df_main_copy.columns if (dfc in df_trm_copy.columns) and dfc != 'row_id']

    grouped = df_trm_copy.groupby(match_cols + ['pitch_type_group', 'auto_pitch_type'])
    grouped_phase1 = grouped[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']].agg(['mean', 'std'])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [col for col in grouped_phase2.columns if 'std' in col]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ['_'.join(col).strip('_') for col in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns='auto_pitch_type')
    grouped_phase3 = grouped_phase3.groupby(match_cols + ['pitch_type_group']).agg(['mean'])

    pivoted = grouped_phase3.unstack(level='pitch_type_group')
    pivoted.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)

    df_main_copy['top_bottom'] = df_main_copy['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    tm_final['top_bottom'] = tm_final['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)

    for col in ['batter_hand', 'pitcher_hand']:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({'Left': 1, 'Right': 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how='left')
    new_feature_cols = [col for col in tm_final.columns if col not in match_cols]
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(tr_final[new_feature_cols].mean())

    return tr_final, match_cols


SEASON_PROGRESSION_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def build_season_end_lookup(df):
    """투수/타자별 "시즌 마지막 행" 누적치 정적 lookup 테이블 (동섭님 피처, 2026-08-19
    창현님 저장소 채택, 2026-08-24 이식). control_success가 있는 학습 데이터(스플릿
    이전 전체)에서만 계산 가능 — test.csv엔 과거 시즌 행 자체가 없으므로, 이 테이블을
    한 번 만들어 정적으로 재사용해야 한다(대회 규칙: 평가 데이터 독립 예측 원칙).
    반환: role별 next_season(이 값이 적용될 시즌) 키의 DataFrame."""
    tables = []
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col, "control_success"]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate", "end_success"]
        season_end["season"] = season_end["season"] + 1  # 다음 시즌 키로 이동
        season_end.insert(0, "role", role)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_season_progression_features(df, lookup):
    """투수/타자 "시즌 진행분" 8개. asof_{pitcher,batter}_success_rate는 커리어 전체
    누적값이라 베테랑일수록 최근 시즌 컨디션 신호가 희석되는데, lookup(직전 시즌 마지막
    행의 누적치)을 시즌 시작 시점 기준값으로 빼서 "이번 시즌만의" 성공률과 커리어 누적
    대비 격차(rate_gap)를 분리해낸다. df 자신의 다른 행이나 control_success에 의존하지
    않으므로 test.csv에도 안전. 창현님 저장소 실측: CatBoost 단독 cutoff7 +34.66/
    season==2023 +150.76, 블렌드 cutoff7 +35.90/season==2023 +164.90, 실전 리더보드
    748(로컬)→1027.54(+42) — 트랙맨류와 달리 두 레짐·두 모델 전부 같은 방향으로
    개선되어 채택됨(EXPERIMENTS.md §7 참고)."""
    df = df.copy()
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
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


def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다.

    df는 control_success를 포함한 학습 데이터(스플릿 이전 전체)여야 한다 — 시즌
    진행분 피처가 lookup을 df 자신으로부터 만들기 때문."""
    df = df.copy()
    df = apply_season_progression_features(df, build_season_end_lookup(df))

    df['pitcher_recent1_gap'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent3_gap'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent5_gap'] = df['asof_pitcher_prev5_game_success_rate'] - df['asof_pitcher_success_rate']

    df['pitcher_relative_success'] = df['asof_pitcher_success_rate'] - league_success_mean

    df['count_diff'] = df['strikes_before'] - df['balls_before']
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(np.int64)

    df['pitcher_count_advantage_raw'] = df['asof_pitcher_success_rate'] * df['count_diff']
    df['pitcher_count_advantage_rel'] = df['pitcher_relative_success'] * df['count_diff']

    df['pitcher_trend'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_consistency'] = df[[
        'asof_pitcher_prev1_game_success_rate',
        'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate',
    ]].std(axis=1)

    df['matchup'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']

    pressure_signal = df['li'] * ((df['strikes_before'] >= 2) | (df['balls_before'] >= 3)).astype(np.int64)
    df['count_pressure'] = df['pitcher_relative_success'] * pressure_signal

    return df


TE_K_SMOOTH = 200

# (피처명, 그룹 컬럼, 잔차를 뺄 주효과 컬럼명)
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
    """source_df(F1 필터 적용된 학습 파티션)로 시즌-causal 스무딩 타겟인코딩을 만들어
    query_df(train_split 자신 또는 val_split)에 적용합니다. query_df 행 자신의
    시즌보다 엄격히 앞선 시즌들의 데이터만 씀(`allow_exact_matches=False`가 핵심) —
    학습 파티션 자기 자신에 적용해도 self-leakage가 없고, F1 필터가 이미 적용된
    source_df만 쓰므로 F1 오염 재유입도 구조적으로 막힘."""
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
    """투수/타자 상황별 target-encoding 잔차 6개(팀원 제보 "Track A", 2026-08-19
    창현님 저장소 채택, 2026-08-24 이식). asof_pitcher_success_rate(주효과, 이미 공식
    피처)와의 중복을 줄이려고 상황별 인코딩에서 주효과(p_main/b_main)를 뺀 잔차만
    씀. **CatBoost 전용** — MLP에 같이 먹이면 MLP 단독이 크게 무너져 블렌드 이득이
    거의 사라짐(창현님 저장소 실측: cutoff7 MLP -24.51 → 블렌드 -2.38). CatBoost
    전용으로 두면 cutoff7 +6.97/season==2023 +20.49, rolling-origin 3-fold(2021/
    2022/2023) 평균 +28.01(3/3 fold 승)로 확인됨(EXPERIMENTS.md §7 참고).

    호출부 관례(원작 기준): `te_prior = train_split[target].mean()`(F1 필터 적용된
    train_split 자신의 평균), `te_source = train_split`(F1 필터 적용된 자기 자신)로
    train_split/val_split 양쪽에 동일하게 적용."""
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


def add_experimental_features(df):
    """add_engineered_features() 이후에 호출. count_diff/is_full_count 등에 의존하므로 순서 주의.
    창현님 원본 피처와 구분하기 위해 별도 함수로 분리한 신규 실험 피처."""
    df = df.copy()

    # 1. 주자 상황 x 투수 성향 교차 (기존엔 볼카운트 압박만 있었고 주자 압박은 미반영)
    df['runner_pressure_reverse'] = df['num_runners_on'] * df['asof_pitcher_reverse_rate']
    df['runner_pressure_middle'] = df['num_runners_on'] * df['asof_pitcher_middle_rate']
    scoring_position = ((df['runner_on_2b'] == 1) | (df['runner_on_3b'] == 1)).astype(np.int64)
    df['scoring_position_reverse'] = scoring_position * df['asof_pitcher_reverse_rate']

    # 2. home/away_win_expectancy 미활용 컬럼 활용 (li와 별개로 "얼마나 접전인지" 표현)
    df['win_expectancy_gap'] = (df['home_win_expectancy'] - df['away_win_expectancy']).abs()

    # 3. 구종 믹스 x 카운트 교차 (풀카운트/유리한카운트에서 구종 비중이 다르게 작용하는지)
    df['fastball_count_interaction'] = df['asof_pitcher_fastball_rate'] * df['count_diff']
    df['breaking_fullcount_interaction'] = df['asof_pitcher_breaking_rate'] * df['is_full_count']
    df['offspeed_pressure'] = df['asof_pitcher_offspeed_rate'] * df['li']

    return df
