"""로컬 검증 프로토콜 통일 유틸 (2026-08-23 팀 회의 결정, EXPERIMENTS.md §6-15).

기존 `season==2024` 단일 홀드아웃 / 2023+2024 평균 방식을 대체. 두 개의 기본 fold +
필요시 expanding window로 확장:

- Fold A(`half_2024`): 2019 ~ 2024/06 학습 -> 2024/07~10 검증 (시즌 도중 cutoff).
  트랙맨 매칭이 "같은 시즌이지만 아직 못 본 미래 월" 상황에서 어떻게 동작하는지를
  로컬에서 잡아내려는 목적 (기존 season 단위 홀드아웃으로는 이 상황이 재현 안 됐음).
- Fold B(`expand_2023`): 2019~2022 학습 -> 2023 검증.
- expanding: 통과하면 2019~2020->2021, 2019~2021->2022 ... 로 검증 연도를 늘림.

전부 train.csv만 사용(행 단위 시간순 분할), test.csv는 관여하지 않으므로 "평가 데이터
독립 예측 원칙"과 무관 — CLAUDE.md 규칙은 이 유틸 자체에는 해당 없음(추론 스크립트가
아니라 학습용 로컬 검증 분할 도구).
"""
import numpy as np
import pandas as pd

DEFAULT_EXPANDING_VAL_SEASONS = (2023, 2021, 2022)


def half_2024_fold(df, cutoff_month=6):
    """Fold A: 2019~2024/cutoff_month 학습 -> 2024/(cutoff_month+1)~10 검증."""
    season = df["season"]
    month = df["game_month"]
    train_mask = (season < 2024) | ((season == 2024) & (month <= cutoff_month))
    val_mask = (season == 2024) & (month > cutoff_month)
    return train_mask, val_mask


def expanding_fold(df, val_season):
    """val_season 하나를 검증으로, 그 이전 시즌 전부를 학습으로 쓰는 fold."""
    season = df["season"]
    train_mask = season < val_season
    val_mask = season == val_season
    return train_mask, val_mask


def iter_folds(df, val_seasons=DEFAULT_EXPANDING_VAL_SEASONS, include_half_2024=True):
    """(fold_name, train_mask, val_mask) 튜플을 순서대로 yield.

    기본 순서: half_2024 -> expand_2023 -> (통과 시) expand_2021 -> expand_2022 ...
    val_seasons 순서를 그대로 따름(기본값은 회의에서 정한 순서: 2023 먼저, 그다음 2021/2022).
    """
    if include_half_2024:
        train_mask, val_mask = half_2024_fold(df)
        yield "half_2024", train_mask, val_mask
    for val_season in val_seasons:
        train_mask, val_mask = expanding_fold(df, val_season)
        yield f"expand_{val_season}", train_mask, val_mask


def compute_bss(preds, y):
    preds = np.asarray(preds)
    y = np.asarray(y)
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def score(preds, y):
    return max(0.0, 100000 * compute_bss(preds, y))


def summarize(fold_scores):
    """{fold_name: score} -> (평균, 보기 좋은 표 문자열)."""
    avg = sum(fold_scores.values()) / len(fold_scores)
    lines = [f"  {name}: {s:.2f}" for name, s in fold_scores.items()]
    lines.append(f"  평균: {avg:.2f}")
    return avg, "\n".join(lines)
