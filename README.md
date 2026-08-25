# LG Aimers 9기 — 투구 제구 성공 확률 예측

CatBoost + Tabular MLP 블렌드. 대회 규칙은 `CLAUDE.md`, 전체 실험 기록은 `EXPERIMENTS.md` 참고.

## 파일 네비게이션

| 보고 싶은 것 | 파일 |
|---|---|
| 피처 정의 전체(트랙맨 조인, 구종비중, 시즌진행분, Track A) | `feature_engineering.py` |
| MLP 모델/전처리/앙상블 | `mlp_model.py` |
| 실제 학습 파이프라인(전체 조립) | `full_retrain_blend_f1.py` |
| 실제 추론 코드(독립 실행용) | `submit/script.py` |
| 실험 기록 전체 | `EXPERIMENTS.md` |

## CatBoost

- 피처: `feature_engineering.py`에서 만들어지는 141개 전부(원본 47 + 트랙맨 물리량 64 + 구종비중 4 + 파생피처 12 + 시즌진행분 8 + Track A 6)
- 하이퍼파라미터: `model_py311/best_catboost_pitchmix_hparams.json`(로컬 전용, gitignore) — `depth=7, learning_rate=0.02034, l2_leaf_reg=14.806, random_strength=7.992, bagging_temperature=0.00872, border_count=167, min_data_in_leaf=35, bootstrap_type=Bayesian, loss_function=Logloss, eval_metric=BrierScore, iterations=1462`
- 시드: 1개(앙상블 없음)
- 학습: `full_retrain_blend_f1.py`

## MLP (Tabular MLP)

- 피처: CatBoost 141개 중 Track A 6개 제외한 135개 — 수치형 128(StandardScaler) + 임베딩 범주형 7(`top_bottom, game_type, base_state, pitcher_hand, batter_hand, pitcher_team_id, batter_team_id`)
- 하이퍼파라미터(`mlp_model.py`): `hidden1=128, hidden2=64, dropout=0.3, lr=0.003, weight_decay=0.01, batch_size=4096, max_epochs=60(홀드아웃)/25(전체재학습), patience=7`
- 구조: 임베딩 concat + 수치 concat → Linear(128)→BN→ReLU→Dropout → Linear(64)→BN→ReLU → Linear(1)→Sigmoid
- 시드: **7개** `[42, 123, 7, 2024, 99, 555, 31337]` — 확률 단순 평균. 시드 15개로 늘려도 추가 개선 없음이 확인된 상태라(`mlp_model.py` 주석 참고) 7개로 고정
- 학습: `full_retrain_blend_f1.py`

## 블렌드

2-입력 로지스틱회귀 스태킹(2026-08-25 채택, 고정 alpha 방식 대체):

```
final_pred = sigmoid(w_cat * cat_pred + w_mlp * mlp_pred + intercept)
w_cat=2.002827, w_mlp=1.988260, intercept=-2.022844
```

## 학습 데이터 필터

`season<=2022 & game_type=='F'` 행을 학습에서만 제거(검증/추론 데이터엔 적용 안 함).

## 최고 기록

로컬(새 검증 프로토콜, half_2024+expand_2023 평균) 792.22 → 실전 리더보드 **1059.72**.
