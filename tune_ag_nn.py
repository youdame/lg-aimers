"""AutoGluon NN_TORCH에 실제 HPO(하이퍼파라미터 탐색)를 적용해서,
아까 기본값 결과(445.11)가 튜닝 부재 때문인지 확인한다.
CatBoost가 튜닝만으로 +52.5점 오른 전례(736.61->818.54 중 765.99->818.54 구간)와 비교선상에 둠.
"""
import os
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.common import space

from feature_engineering import process_trackman_features_safe, add_engineered_features

DATA_DIR = "./data"
TARGET = "control_success"
NUM_TRIALS = int(os.environ.get("AG_NUM_TRIALS", 8))
TIME_LIMIT = int(os.environ.get("AG_TIME_LIMIT", 1200))
TRAIN_SAMPLE = int(os.environ.get("AG_TRAIN_SAMPLE", 300_000))


def brier_score_func(y_true, y_pred):
    return float(((np.asarray(y_pred) - np.asarray(y_true)) ** 2).mean())


from autogluon.core.metrics import make_scorer

brier_scorer = make_scorer(
    name="brier",
    score_func=brier_score_func,
    optimum=0,
    greater_is_better=False,
    needs_proba=True,
)


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return brier, 1.0 - (brier / baseline_brier)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    keep_cols = [c for c in train_df.columns if c != 'row_id']
    train_part = train_df.loc[train_mask, keep_cols].reset_index(drop=True)
    val_part = train_df.loc[val_mask, keep_cols].reset_index(drop=True)
    y_val = val_part[TARGET].values

    if TRAIN_SAMPLE and TRAIN_SAMPLE < len(train_part):
        train_part = train_part.sample(n=TRAIN_SAMPLE, random_state=42).reset_index(drop=True)
    print(f"train={len(train_part)} val={len(val_part)} | trials={NUM_TRIALS} time_limit={TIME_LIMIT}s", flush=True)

    nn_search = {
        'num_epochs': 30,
        'learning_rate': space.Real(1e-4, 1e-2, log=True),
        'weight_decay': space.Real(1e-12, 1e-2, log=True),
        'dropout_prob': space.Real(0.0, 0.4),
        'num_layers': space.Int(2, 5),
        'hidden_size': space.Categorical(64, 128, 256, 512),
    }

    t0 = time.time()
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric=brier_scorer,
        problem_type='binary',
        verbosity=2,
    ).fit(
        train_data=train_part,
        tuning_data=val_part,
        use_bag_holdout=True,
        hyperparameters={'NN_TORCH': nn_search},
        hyperparameter_tune_kwargs={
            'num_trials': NUM_TRIALS,
            'scheduler': 'local',
            'searcher': 'random',
        },
        time_limit=TIME_LIMIT,
    )
    print(f"[Tune] 완료: {time.time()-t0:.1f}s", flush=True)

    lb = predictor.leaderboard(val_part, silent=True)
    print("\n[Leaderboard on val]")
    print(lb.to_string(), flush=True)

    best_pred = predictor.predict_proba(val_part)[1].values
    _, bss = compute_bss(best_pred, y_val)
    score = max(0, 100000 * bss)
    print(f"\n튜닝된 NN_TORCH 최고 성능: score={score:.2f}")
    print(f"참고: 기본값(튜닝 X) NeuralNetTorch = 445.11")
    print(f"참고: CatBoost 단독 = 818.54")
    print(f"참고: CatBoost도 튜닝만으로 +52.5점 올랐음 (765.99 -> 818.54)")


if __name__ == "__main__":
    main()
