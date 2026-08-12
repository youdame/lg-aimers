import os
import time

import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from feature_engineering import process_trackman_features_safe, add_engineered_features

N_TRIALS = int(os.environ.get("TUNE_TRIALS", 40))
DATA_DIR = "./data"
TARGET = "control_success"
OUT_PATH = "./model_py311/best_lgb_hparams.json"


def compute_bss(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return 1.0 - (brier / baseline_brier)


def lgb_brier_feval(y_true, y_pred):
    return 'brier', float(((y_pred - y_true) ** 2).mean()), False


def build_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)

    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024

    league_success_mean = train_df.loc[train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET]
    features = [c for c in train_df.columns if c not in drop_cols]
    categorical_features = [c for c in tr_final.select_dtypes(include="object").columns if c != "row_id" and c in features]
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna("missing").astype("category")

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET]
    X_val = train_df.loc[val_mask, features]
    y_val = train_df.loc[val_mask, TARGET]
    return X_train, y_train, X_val, y_val, categorical_features


def make_objective(X_train, y_train, X_val, y_val, categorical_features):
    def objective(trial):
        t0 = time.time()
        params = dict(
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.008, 0.05, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            subsample=trial.suggest_float("subsample", 0.55, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.55, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.02, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.02, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )
        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=lgb_brier_feval,
            callbacks=[early_stopping(50, verbose=False)],
            categorical_feature=categorical_features,
        )
        preds = model.predict_proba(X_val)[:, 1]
        bss = compute_bss(preds, y_val)
        trial.set_user_attr("best_iteration", model.best_iteration_)
        print(f"[Trial {trial.number:03d}] BSS={bss:.5f} (score={max(0,100000*bss):.2f}) "
              f"best_iter={model.best_iteration_} took={time.time()-t0:.1f}s params={trial.params}", flush=True)
        return bss
    return objective


def main():
    print("[Tune-LGB] 데이터 로드 및 피처 엔지니어링...", flush=True)
    t = time.time()
    X_train, y_train, X_val, y_val, categorical_features = build_data()
    print(f"[Tune-LGB] train: {len(X_train)}행 | val: {len(X_val)}행 | 피처: {X_train.shape[1]}개 ({time.time()-t:.1f}s)", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(X_train, y_train, X_val, y_val, categorical_features), n_trials=N_TRIALS)

    print("\n" + "=" * 60, flush=True)
    print(f"[Tune-LGB 완료] Best BSS: {study.best_value:.5f} (score: {max(0, 100000*study.best_value):.2f})", flush=True)
    print(f"Best params: {study.best_params}", flush=True)
    print(f"참고: CatBoost 818.54 | XGBoost 752.70 | LightGBM 튜닝전 725.87", flush=True)
    print("=" * 60, flush=True)

    import json
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "best_bss": study.best_value,
            "best_score": max(0, 100000 * study.best_value),
            "best_params": study.best_params,
            "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        }, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
