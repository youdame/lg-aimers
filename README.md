# LG Aimers 9기 — 투구 제구 성공 확률 예측

> 5인 팀 온라인 해커톤 (2026.06.22 ~ 2026.09.02). 투수의 투구가 제구에 성공할 확률을 예측하고, **Brier Skill Score**로 평가했습니다.
> 이 저장소는 팀 프로젝트 중 제가 직접 실험·검증한 개인 파이프라인과 실험 기록입니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 문제 | 투구별 `control_success` 확률 예측 (평가지표: Brier Skill Score) |
| 모델 | CatBoost + Tabular MLP, 로지스틱 회귀 스태킹 블렌드 |
| 피처 | 141개 (원본 47 + 트랙맨 집계 64 + 구종비중 4 + 파생 12 + 시즌 진행분 8 + Track A 6) |
| 개인 리더보드 기록 | 927 → 968.15 → 1059.72 → 1085.24 (`docs/EXPERIMENTS.md` 기록 기준) |
| 가장 중요하게 본 것 | 하나의 검증 시즌만 믿지 않기, 평가 데이터 누수 막기 |

## 제가 한 일

- **구종비중(pitch-mix) 피처:** 트랙맨 기록을 볼카운트와 투수·타자 손 기준으로 묶어 구종(직구/변화구/오프스피드/기타) 비율 4개를 만들었습니다. 검증 시점 이후의 트랙맨 기록이 섞이지 않게 시간 누수를 막았습니다.
- **F1 필터와 결합, 효과 규명:** 2022년 이하 퓨처스리그(F) 데이터를 학습에서만 제외하는 필터를 구종비중과 결합해 다시 학습·제출해 968.15점을 만들었습니다. 2024 검증에서는 손해(XGBoost −38.74), 2023 검증에서는 큰 이득(+283.75)으로 상반되게 나오는 이유를 확인해, 단일 시즌 검증이 효과를 왜곡한다는 걸 밝혔습니다.
- **메타모델 스태킹 이식·검증:** 고정 가중치 대신 2-입력 로지스틱 회귀로 CatBoost와 MLP 예측을 합쳤습니다. 검증 구간을 같은 레짐 안에서 나눠야 팀 저장소 결과와 같은 효과가 재현된다는 걸 확인했습니다.
- **원인을 못 찾은 하락을 정리하고 되돌림:** 개선 사항을 모두 합쳤더니 968.15 → 879.54로 떨어졌고, 하나씩 빼봐도 원인을 못 찾았습니다. 추측으로 제출하지 않고 968.15 커밋으로 되돌린 뒤 새 검증 프로토콜(half_2024 / expand_2023)을 만들었습니다.
- **평가 데이터 독립 예측 규칙 준수:** test 행끼리 통계를 섞는 코드가 없는지 점검하는 원칙을 `CLAUDE.md`에 정리했습니다.

## 구조

```text
.
├── src/                          공통 모듈
│   ├── feature_engineering.py    피처 정의 (트랙맨 조인, 시즌 진행분, Track A)
│   ├── mlp_model.py              Tabular MLP 모델·전처리·앙상블
│   └── local_validation.py       검증 fold 정의(half_2024 / expand_2023)와 BSS 계산
├── pipeline/                     최종 모델 학습·분석
│   ├── full_retrain_blend_f1.py  실제 학습 파이프라인 (CatBoost + MLP + 블렌드)
│   ├── cache_fold_predictions*.py  메타모델 실험용 fold 예측 캐시
│   └── analyze_feature_importance.py
├── tuning/                       하이퍼파라미터 탐색(Optuna)·재검증
├── experiments/                  가설별 검증 실험 (하나의 가설에 하나의 파일)
│   └── experiment_pitchmix_coarse.py  구종비중 피처 (다른 스크립트들이 import)
├── submit/                       대회 서버용 독립 추론 코드 (script.py, requirements.txt)
├── notebooks/                    대회 제공 베이스라인 노트북
├── docs/
│   ├── EXPERIMENTS.md            전체 실험 기록: 효과 있었던 것/없었던 것과 이유
│   ├── data_description.md       데이터 명세와 공식 규칙
│   └── feature_importance_results.txt
└── CLAUDE.md                     대회 규칙과 코드 작성 시 지킨 원칙
```

## 실행 방법

모든 스크립트는 저장소 루트에서 모듈로 실행합니다 (`./data`, `./model_py311` 상대경로 기준).

```bash
# 데이터: 대회 사이트에서 받은 train.csv, trackman_history.csv, test.csv를 data/에 배치
python -m pipeline.full_retrain_blend_f1          # 최종 모델 학습
python -m experiments.experiment_same_hand        # 개별 실험 재현
```

## 모델

**CatBoost** — 피처 141개 전부 사용, 하이퍼파라미터는 Optuna 탐색, Brier Score 기준.

**Tabular MLP** — Track A 6개를 뺀 135개 피처(수치형 128 + 임베딩 범주형 7). 임베딩 concat → Linear(128) → BN → ReLU → Dropout → Linear(64) → BN → ReLU → Linear(1) → Sigmoid. 시드 7개 평균.

**블렌드**

```text
final_pred = sigmoid(w_cat * cat_pred + w_mlp * mlp_pred + intercept)
w_cat=2.002827, w_mlp=1.988260, intercept=-2.022844
```

## 실험 기록을 읽는 법

[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)는 잘 된 것뿐 아니라 **안 된 것과 그 이유**를 같이 적었습니다. 예를 들면 다음과 같습니다.

- 트랙맨 물리량을 빼면 좋아질 거라는 가설 → 실전 955점으로 효과 없음, 종결
- 개선 사항 전체 반영 시 879.54로 하락 → 범인 후보 3개를 하나씩 빼서 확인했으나 모두 기각

## 한계

- 최종 점수는 팀 제출 기록과 이 저장소 기록이 다릅니다. 이 저장소에는 제가 직접 제출한 점수만 적었습니다.
- 대회 데이터는 약관상 재배포할 수 없어 저장소에 포함하지 않았습니다. 실행하려면 대회 사이트에서 받아 `data/`에 넣어야 합니다 (`.gitignore` 처리됨).
