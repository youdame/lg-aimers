"""experiment_track_a_port.py(시즌진행분+Track A, ball/strike 없음)를 expand_2021/2022
fold에 대해 실행 -- ball/strike의 순수 효과를 가려내기 위한 중간 기준선.
"""
import numpy as np

from experiment_track_a_port import build_base_df, run_fold
from local_validation import iter_folds


def main():
    df, df_trm, train_df, tr_final = build_base_df()
    results = {}
    for fold_name, train_mask, val_mask in iter_folds(train_df, val_seasons=(2021, 2022), include_half_2024=False):
        results[fold_name] = run_fold(df, df_trm, train_df, tr_final, train_mask, val_mask, fold_name)

    print("\n=== 요약 (시즌진행분+Track A만, ball/strike 없음, expand_2021/2022) ===")
    for key in ["cat", "mlp", "blend"]:
        vals = [results[f][key] for f in results]
        print(f"{key}: " + ", ".join(f"{f}={results[f][key]:.2f}" for f in results) + f" | 평균={np.mean(vals):.2f}")
    print("\n참고: +ball/strike 버전(이미 완료) = "
          "CatBoost 805.54/1123.18, MLP 592.48/1072.22, Blend 759.90/1130.18")


if __name__ == "__main__":
    main()
