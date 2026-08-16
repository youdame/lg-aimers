# LG Aimers 9기 온라인 해커톤 — 대회 규칙 (항상 확인할 것)

투구 제구 성공 확률 예측 AI 온라인 해커톤. 평가지표: Brier Skill Score.

## ⚠️ 코드 작성 시 반드시 지킬 규칙

### 1. 평가 데이터 독립 예측 원칙 (제일 중요, 실격 사유)

> 평가 데이터(test.csv)의 각 행은 하나의 독립적인 예측 대상입니다.
> 참가자는 각 행에 포함된 입력 변수와 주최 측이 제공한 공식 학습 데이터만을 이용하여
> 해당 행의 예측값을 추론해야 합니다. 평가 데이터의 다른 행이나 전체 평가 데이터의
> 분포를 이용해 특정 행의 예측값을 보정하거나 생성하는 방식은 정상적인 추론 절차로
> 인정되지 않습니다. (2026-08-13 데이콘 재안내 공지)

**허용되는 것**: 행 A의 입력 변수 / 행 A의 입력만으로 만든 파생변수 / 공식 학습 데이터(train.csv, trackman_history.csv) / 공식 학습 데이터만으로 만든 통계·모델·파생변수.

**금지되는 것 (예시)**:
- test.csv 내 다른 행을 이용한 누적 통계·rolling/lag feature
- test.csv 전체의 평균/분포/빈도/순위로 예측값 보정 — **`fillna(df.mean())`을 test 유래 데이터프레임에 쓰면 이거에 해당함, 반드시 train.csv/trackman_history.csv에서 미리 계산한 고정값으로 채울 것**
- 같은 선수/팀/월/경기로 평가 데이터 내 다른 행을 집계
- 평가 데이터상 더 과거로 보이는 다른 행을 이용해 현재 행 피처 생성

**판별법**: "이 행의 예측값이 test.csv에 이 행 하나만 있을 때와 전체가 다 있을 때 동일한가?" 다르면 위반.

**실제 발생했던 사고 (2026-08-13 발견, 수정함)**: `submit/script.py`의 `process_trackman_features()`, `add_coarse_pitchmix()`에서 트랙맨 매칭 결측치를 `tr_final[cols].fillna(tr_final[cols].mean())`로 채우고 있었음 — 이 mean이 test.csv(추론 대상) 자체에서 계산됨 → 규칙 위반. 원인: 학습/검증용 함수(`feature_engineering.py::process_trackman_features_safe`, 거기선 train/val 자체 평균이라 문제없음)를 추론 스크립트에 그대로 복사하면서 fillna 로직을 안 바꿈. 창현님 원본 제출 코드(`lgaimers9/submit/script.py`)는 `fillna(0)`(고정값)으로 정상 처리하고 있었음 — 이식 과정에서 생긴 버그였음. **추론 스크립트에서 `.fillna(...mean())` 패턴을 보면 무조건 의심할 것.**

### 2. 투구 시점 이후 확정 정보 사용 금지 (`data_description.md` §6 공식 규칙)

> 다음 정보는 입력으로 사용할 수 없습니다: 현재 투구 이후에 확정되는 모든 정보 / 현재
> 투구의 실제 위치 또는 코스 정보 / 현재 투구의 실제 판정·결과·제구 성공 여부 / 현재
> 투구의 실제 구종 / **현재 투구의 실제 Trackman 측정값** / 2025년 Trackman 데이터 /
> 평가 데이터 내부의 다른 행을 이용해 만든 누적·빈도·분포·rolling·target encoding 피처.

`trackman_history.csv`의 `rel_speed`, `spin_rate`, `induced_vert_break` 등은 해당 투구가
릴리스된 시점(=투구 결과가 사실상 정해지는 시점) 이후에나 확정되는 값이라, **"이 행이
가리키는 바로 그 투구"의 실측값을 그 행 자신의 예측 입력으로 쓰면 안 됨** (실격 사유급
리크). 트랙맨 데이터를 피처로 쓰려면 반드시 **다른 여러 투구들을 모은 집계값**(상황별
평균/표준편차/구종비중 등)이어야 하고, 예측 대상 행 자신의 실측값이 섞이면 안 됨.

**현재 코드는 안전함(확인 완료, 2026-08-16)**: `feature_engineering.py::process_trackman_features_safe`의
`match_cols`는 시즌/월/이닝/볼카운트/손 등 **상황(situational) 컬럼만**이고 선수 ID처럼
개별 투구를 특정할 수 있는 컬럼은 없음(train.csv의 `pitcher_id`와 trackman의
`pitcher_trackman_id`는 컬럼명이 달라 자동으로 매칭 기준에 안 들어감) — 그룹당 표본이
많아서(수백~수만 건) 항상 여러 투구의 평균/표준편차로 집계됨, 특정 한 투구의 실측값이
그대로 새어 들어갈 수 없는 구조. `구종비중(coarse pitch-mix)`도 마찬가지로 볼카운트+손
기준 집계라 동일하게 안전함.

**과거에 시도했다가 폐기한 접근 중 이 규칙에 위배될 소지가 있었던 것**: 경기 단위 행
정렬로 투수를 1:1 매핑하는 "크로스워크" 실험(§ 트랙맨 활용 3가지 접근, `EXPERIMENTS.md`)은
개별 투구 단위로 실측값을 연결하려 한 시도라, 만약 "실시간 상태 지표"에 해당 투구 자신의
트랙맨 실측값이 섞여 들어갔다면 이 규칙 위반 소지가 있었음(결과적으로 성능 개선이 없어
전부 기각·미채택됐고 프로덕션 코드에는 반영된 적 없음 — 사고로 이어지진 않았으나, 이
방향을 다시 시도할 경우 반드시 유의할 것).

### 3. 사전학습모델/가중치

MIT, Apache 2.0 등 비상업적 이용 허용 라이선스로 공개된 것만 사용 가능.

### 4. 외부 API 금지

OpenAI API, Gemini API 등 원격 서버 기반 API 형태 모델 사용 불가. 모든 작업은 로컬에서 코드로 직접 실행·재현 가능해야 함.

### 5. 외부 데이터 금지

대회 공식 제공 데이터(train.csv, test.csv, trackman_history.csv, sample_submission.csv) 외의 외부 데이터 사용 금지.

## 평가 산식

```
Score = max(0, 100000 × (1 - Brier / 평균제구율_Brier))
Brier = mean((p_i - y_i)^2)
r = mean(y_i)  # 비공개 수치
평균제구율_Brier = r × (1 - r)
```

- Public Score = 테스트 데이터 100%. Private Score = 대회 종료 시점의 Public Score(별도 홀드아웃 아님).
- **수료 조건**: Phase1 이수 + Phase2 Public Score ≥ **549.51** (baseline)
- **1차 평가**: Private Score 100% 기준 순위
- **2차 평가**: Private 상위 약 100팀만 코드+PPT 제출·검증 후 Phase3(오프라인) 진출

## 제출 파일(zip) 형식

`baseline_submit/`(대회 공식 템플릿)와 동일한 구조. **디렉토리명·파일명 완전히 일치해야
함** — 최상위에 추가 폴더 있으면 설치 오류.

```
submit.zip
├── model/             # 모델 가중치 (예: *.pkl, *.pt)
├── script.py          # 추론 실행 코드 (제출 시 서버가 자동 실행, code/ 패키지 미의존)
└── requirements.txt   # 필요 패키지+버전 (pip install -r로 설치 가능한 형태)
```

**서버가 자동으로 덧붙이는 것** (건드릴 필요 없음, 읽기전용):
```
├── data/                    # 평가용 진짜 데이터 (읽기전용, 수정 불가)
└── output/submission.csv    # script.py가 반드시 이 경로/파일명으로 결과 저장해야 함
```

**용량/시간 제한**: zip ≤10GB (압축해제 후 ≤32GB) · 패키지 설치 ≤10분 · 추론 실행 ≤10분 (초과 시 오류)

**평가 서버 사양**: Ubuntu 22.04.5, GPU NVIDIA L4(22.4GiB), 6vCPU, RAM 28GB, Python 3.11.15,
CUDA 12.8, **인터넷 접속 비활성화**(패키지 설치 외 외부 다운로드 전부 불가 — 사전학습 가중치도
로컬/zip 안에 포함시켜야 함).

**서버 기본 설치 패키지** (버전 명시된 건 requirements.txt에 재명시하지 말 것 — 충돌 위험):
`torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`, `scipy==1.15.3`, `scikit-learn==1.8.0`,
`joblib==1.5.3`, `threadpoolctl==3.6.0`, `transformers==4.46.3`, `accelerate==1.9.0` 등.
**`catboost`, `xgboost`, `lightgbm`은 기본 설치 안 됨 → requirements.txt에 반드시 버전 명시.**

**오류 두 종류, 취급이 다름 (중요)**:
- **설치 오류** (zip 구조 불일치, 패키지 설치 실패) → **일일 제출 횟수에 반영 안 됨**
- **제출 오류** (script.py 실행 중 발생하는 모든 오류) → **일일 제출 횟수 소모됨**
- → zip 구조·requirements.txt는 특히 신중하게, 로컬 스모크테스트 필수 (아래)

- `script.py`는 test.csv를 다루는 모든 로직에서 "평가 데이터 독립 예측 원칙"(위 1번) 준수 필수
- 제출 전 로컬 5행 test.csv로 스모크 테스트 (에러 없이 돌아가는지) 확인 후 zip

## 기타 유의사항

- **1일 최대 제출 횟수: 5회** (단, 위 "설치 오류"는 미반영)
- 사용 가능 언어: Python
- Phase 3 진출 희망 시 제출 필요: 학습 환경(OS/라이브러리 버전), Private Score 재현용 학습 코드, 솔루션 PPT, 팀원 참가 여부
- 부정 제출/규칙 위반 발견 시 실격, 제출 후 문제 인지해도 소급 면책 안 됨

## 프로젝트 구조 참고

- `feature_engineering.py`: 학습/검증용 트랙맨 조인 + 파생피처 (시간 마스킹 있음, `.fillna(mean())` 써도 됨 — train/val 자체 데이터라 규칙과 무관)
- `submit/`, `xgb_submit/`: 실제 제출용 독립 실행 스크립트 (code/ 패키지 미의존 관례) — **여기서는 test.csv 유래 통계로 fillna하면 안 됨, 항상 train.csv/trackman_history.csv에서 미리 계산한 고정값 사용**
- `experiment_*.py`: 로컬 실험/검증 스크립트, 제출용 아님
