"""MLP 신규 하이퍼파라미터(tune_mlp_v2.py 1등 후보) 정식 검증 (2026-08-26).

tune_mlp_v2.py의 2-seed 탐색 순위는 노이즈가 커서(트라이얼 내 시드간 최대 59점 차이
관측됨) 그대로 최종 재학습에 반영하지 않고, 1등 후보를 기존 프로덕션 하이퍼파라미터와
동일하게 ENSEMBLE_SEEDS 7개로 정식 비교한 뒤 진짜 이길 때만 채택한다.
"""
import json
import os

import numpy as np
import torch

import mlp_model as mlp_module
from mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, embed_dim_for_cardinality,
    train_ensemble, predict_ensemble,
    HIDDEN1 as PROD_HIDDEN1, HIDDEN2 as PROD_HIDDEN2, DROPOUT as PROD_DROPOUT,
    LR as PROD_LR, WEIGHT_DECAY as PROD_WEIGHT_DECAY, BATCH_SIZE as PROD_BATCH_SIZE,
)
from local_validation import score
from tune_mlp_v2 import build_data

CANDIDATE_PATH = "./model_py311/best_mlp_hparams_v2.json"


def run_config(name, hidden1, hidden2, dropout, lr, weight_decay, batch_size, data):
    (X_tr_cat, X_tr_num, y_tr, X_val_cat, X_val_num, y_val_t, y_val,
     cat_dims, embed_dims, num_numeric_feats) = data

    mlp_module.HIDDEN1 = hidden1
    mlp_module.HIDDEN2 = hidden2
    mlp_module.DROPOUT = dropout

    print(f"\n[{name}] hidden1={hidden1} hidden2={hidden2} dropout={dropout:.4f} "
          f"lr={lr:.5f} weight_decay={weight_decay:.5f} batch_size={batch_size}", flush=True)

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=num_numeric_feats,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t,
        seeds=ENSEMBLE_SEEDS, embed_dims=embed_dims,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, verbose=False,
    )
    preds = predict_ensemble(members, cat_dims, num_numeric_feats, embed_dims, X_val_cat, X_val_num)
    s = score(preds, y_val)
    print(f"[{name}] 7-seed 앙상블 점수 = {s:.2f}", flush=True)
    return s


def main():
    print("[Verify] half_2024 fold 데이터 준비 중...", flush=True)
    data = build_data()

    with open(CANDIDATE_PATH) as f:
        cand = json.load(f)
    cp = cand["best_params"]
    print(f"[Verify] 탐색 1등 후보(2-seed avg_score={cand['best_score']:.2f}): {cp}", flush=True)

    prod_score = run_config(
        "기존 프로덕션", PROD_HIDDEN1, PROD_HIDDEN2, PROD_DROPOUT,
        PROD_LR, PROD_WEIGHT_DECAY, PROD_BATCH_SIZE, data,
    )
    cand_score = run_config(
        "신규 후보", cp["hidden1"], cp["hidden2"], cp["dropout"],
        cp["lr"], cp["weight_decay"], cp["batch_size"], data,
    )

    print("\n" + "=" * 60, flush=True)
    print(f"기존 프로덕션 (7-seed) = {prod_score:.2f}", flush=True)
    print(f"신규 후보     (7-seed) = {cand_score:.2f}", flush=True)
    print(f"Δ = {cand_score - prod_score:+.2f}", flush=True)
    if cand_score > prod_score:
        print("=> 신규 후보 채택 권장", flush=True)
    else:
        print("=> 기존 프로덕션 유지 권장", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
