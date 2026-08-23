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


def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다."""
    df = df.copy()

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
