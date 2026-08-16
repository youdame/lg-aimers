# code/mlp_model.py
"""CatBoost를 대체하는 Tabular MLP(PyTorch) 공용 모델/전처리 유틸리티.

`train.py`, `test.py`, `dopip.py`가 공유해서 사용합니다.
`submit/script.py`는 대회 규칙상 이 모듈을 import하지 않고 필요한 부분을
자체적으로 복제해 둡니다 (기존 CatBoost 파이프라인과 동일한 관례).

모델/전처리기는 커스텀 클래스 인스턴스가 아니라 순수 dict(bundle)로 저장합니다.
submit.zip에는 `code/` 패키지가 포함되지 않으므로, pickle이 프로젝트 전용 클래스에
의존하면 대회 서버에서 unpickle이 실패합니다. dict + torch 텐서 + sklearn
전처리기 조합은 별도 클래스 정의 없이도 어디서나 unpickle이 가능합니다.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 저카디널리티 상황/식별 플래그를 임베딩으로 학습합니다. pitcher_id(792)/batter_id(830)는
# 실험 결과 제외했습니다: 학습 데이터(~122만 행)에 비해 카디널리티가 높아 개별 선수 임베딩이
# 몇 epoch만에 과적합되고(best_epoch=3) 검증 점수가 660 -> 304로 급락했습니다.
#
# season도 제외했습니다: 검증셋(2024)이 학습셋(2019~2023)에 전혀 없던 시즌값이라, season을
# 임베딩으로 두면 검증의 모든 행이 "미학습(unknown) 카테고리"의 랜덤 임베딩을 공유하게 되어
# 전체 예측이 체계적으로 miscalibrate됩니다(660 -> 351로 하락, season만 다시 빼도 동일하게
# 재현됨). 실제 제출 환경도 test.csv가 2025 시즌으로 train.csv(2019~2024)에 없는 값이라 같은
# 문제가 그대로 발생합니다. season처럼 "미래에 반드시 새 값이 나오는" 순서형 컬럼은 임베딩이
# 아니라 숫자(스케일링)로 다뤄야 외삽이 가능합니다. team_id/hand는 카디널리티가 낮고 향후에도
# 새 값이 나올 가능성이 낮아 임베딩으로 유지합니다.
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]

MAX_EMBED_DIM = 50
MIN_EMBED_DIM = 4
# Optuna 재튜닝 결과(2026-08-16, tune_mlp_pitchmix_f1.py, 구종비중+F1 반영 후 2023+2024
# 이중검증, 27trial 중 trial 016 best). 단일시드 프록시 기준 값이라 7-seed 앙상블로
# 재검증 전까지는 잠정치. model_py311/best_mlp_pitchmix_f1_hparams.json 참고.
HIDDEN1 = 128
HIDDEN2 = 96
DROPOUT = 0.4259134876019591
LR = 0.004256541133797278
WEIGHT_DECAY = 0.007764092438820441
BATCH_SIZE = 1024
MAX_EPOCHS = 60
PATIENCE = 7
SEED = 42

# 단일 모델은 epoch마다 검증 점수가 크게 요동치는 고분산 특성을 보였습니다
# (동일 설정으로도 Val Score가 490~740점 사이로 흔들림). 서로 다른 시드로 학습한
# 모델 여러 개의 예측을 평균 내는 앙상블로 이 분산을 줄였더니 실측 데이터에서
# 단일 모델 최고점(약 715~740점)보다 앙상블 평균이 꾸준히 높게 나왔습니다
# (7-seed 앙상블 실측 BSS 점수 780.92, 단일 모델 690.62). 시드 수를 15개로 늘려도
# 추가 개선은 없었고(768.93, 정체) 네트워크를 넓혀도(256/128) 비슷한 수준(777.92)이라
# 7개로 확정했습니다.
ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 555, 31337]

# 수치형 피처를 표준화만 해서 그대로 concat하는 대신, Gorishniy et al.의 quantile 기반
# piecewise-linear 인코딩(PLE, Q-LR)으로 감싼 뒤 concat합니다. n_bins=24, d=8은 팀원(창현님)
# 저장소(lgaimers9-ensemble-v1)의 그리드 스윕(4/8/12/16/24/32, 7-seed paired 비교)에서
# n_bins=24가 delta(+55.78)·승률(7/7)·std(23.49) 세 지표 모두 최적이라 채택했다는 값을 이식.
QUANTILE_N_BINS = 24
QUANTILE_D = 8


def embed_dim_for_cardinality(cardinality):
    """fastai 스타일 임베딩 크기 휴리스틱: min(50, (card+1)//2), 하한 4."""
    return max(MIN_EMBED_DIM, min(MAX_EMBED_DIM, (cardinality + 1) // 2))


def fit_quantile_edges(X_num, n_bins=QUANTILE_N_BINS):
    """(N, num_numeric) 학습 텐서/배열에서 피처별 quantile 경계 (num_numeric, n_bins+1)를 계산합니다.
    반드시 학습 split에서만 계산해야 합니다(리크 방지, fit_preprocessing과 동일한 컨벤션).
    중복 경계(카디널리티가 낮은 피처)는 단조 증가를 보장하도록 아주 작은 epsilon으로 밀어냅니다."""
    X_np = X_num.numpy() if torch.is_tensor(X_num) else np.asarray(X_num)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(X_np, qs, axis=0).T  # (num_numeric, n_bins+1)
    edges = np.maximum.accumulate(edges, axis=1)
    for i in range(1, edges.shape[1]):
        tie = edges[:, i] <= edges[:, i - 1]
        edges[tie, i] = edges[tie, i - 1] + 1e-6
    return torch.tensor(edges, dtype=torch.float32)


class QuantileEmbedding(nn.Module):
    """수치형 피처별 quantile 기반 piecewise-linear 인코딩(PLE) + 피처별 독립 Linear + ReLU
    (Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning")."""

    def __init__(self, bin_edges, d_embed=QUANTILE_D, use_relu=True):
        super().__init__()
        self.register_buffer("edges", bin_edges)  # (num_numeric, n_bins+1), 학습 안 함
        num_numeric, n_bins_plus1 = bin_edges.shape
        n_bins = n_bins_plus1 - 1
        self.num_numeric = num_numeric
        self.n_bins = n_bins
        self.use_relu = use_relu
        self.weight = nn.Parameter(torch.empty(num_numeric, n_bins, d_embed))
        self.bias = nn.Parameter(torch.zeros(num_numeric, d_embed))
        bound = 1.0 / math.sqrt(n_bins)
        nn.init.uniform_(self.weight, -bound, bound)

    def encode(self, x_num):
        left = self.edges[:, :-1].unsqueeze(0)   # (1, num_numeric, n_bins)
        right = self.edges[:, 1:].unsqueeze(0)    # (1, num_numeric, n_bins)
        x = x_num.unsqueeze(-1)                   # (batch, num_numeric, 1)
        width = (right - left).clamp_min(1e-6)
        frac = (x - left) / width
        return frac.clamp(0.0, 1.0)                # (batch, num_numeric, n_bins)

    def forward(self, x_num):
        p = self.encode(x_num)
        e = torch.einsum("bnf,nfd->bnd", p, self.weight) + self.bias
        if self.use_relu:
            e = torch.relu(e)
        return e.reshape(e.shape[0], -1)


class TabularMLP(nn.Module):
    def __init__(self, num_numeric_feats=None, cat_dims=None, embed_dims=None,
                 hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT,
                 bin_edges=None, quantile_d=QUANTILE_D):
        """`bin_edges`가 주어지면 수치형 입력을 QuantileEmbedding(PLE)으로 인코딩하고,
        `bin_edges=None`이면 예전처럼 표준화된 수치형을 그대로 concat합니다(하위 호환,
        `num_numeric_feats` 필요)."""
        super().__init__()
        if embed_dims is None:
            embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])

        if bin_edges is not None:
            self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d)
            numeric_out_dim = bin_edges.shape[0] * quantile_d
        else:
            self.quantile = None
            numeric_out_dim = num_numeric_feats
        total_input_dim = sum(self.embed_dims) + numeric_out_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_num_enc = self.quantile(x_num) if self.quantile is not None else x_num
        x_all = torch.cat([x_embed, x_num_enc], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_preprocessing(df, cat_cols, num_cols):
    """train split(또는 전체 학습 데이터)에 전처리기를 fit하고 변환된 df를 반환합니다."""
    df = df.copy()
    df[cat_cols] = df[cat_cols].astype(str)

    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()

    df[cat_cols] = cat_encoder.fit_transform(df[cat_cols]) + 1
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[num_cols] = num_scaler.fit_transform(df[num_cols])

    cat_dims = [int(df[col].max() + 1) for col in cat_cols]
    return df, cat_encoder, num_imputer, num_scaler, cat_dims


def apply_preprocessing(df, cat_cols, num_cols, cat_encoder, num_imputer, num_scaler):
    """이미 fit된 전처리기를 새 df(검증/추론용)에 적용합니다."""
    df = df.copy()
    df[cat_cols] = df[cat_cols].astype(str)
    df[cat_cols] = cat_encoder.transform(df[cat_cols]) + 1
    df[num_cols] = num_imputer.transform(df[num_cols])
    df[num_cols] = num_scaler.transform(df[num_cols])
    return df


def to_tensors(df, cat_cols, num_cols, target_col=None):
    X_cat = torch.tensor(df[cat_cols].values.astype(np.float32))
    X_num = torch.tensor(df[num_cols].values.astype(np.float32))
    if target_col is not None:
        y = torch.tensor(df[target_col].values.astype(np.float32))
        return X_cat, X_num, y
    return X_cat, X_num


def compute_bss(preds, y):
    preds = np.asarray(preds, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    brier = ((preds - y) ** 2).mean()
    r = y.mean()
    baseline_brier = r * (1 - r)
    bss = 1.0 - (brier / baseline_brier)
    score = max(0.0, 100000.0 * bss)
    return brier, bss, score


def train_mlp(
    X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric_feats=None,
    X_val_cat=None, X_val_num=None, y_val=None,
    embed_dims=None, max_epochs=MAX_EPOCHS, patience=PATIENCE,
    batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
    device=None, verbose=True, seed=SEED,
    hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT,
    bin_edges=None, quantile_d=QUANTILE_D,
):
    """TabularMLP를 학습합니다.

    `bin_edges`가 주어지면 수치형 입력을 QuantileEmbedding(PLE)으로 인코딩합니다.
    `bin_edges=None`이면 `num_numeric_feats`를 이용해 예전처럼 표준화된 수치형을
    그대로 concat합니다(하위 호환).

    검증 텐서(X_val_*)가 주어지면 Val Brier 기준 early stopping을 수행하고,
    best epoch의 가중치로 복원해 (model, best_epoch)을 반환합니다.
    검증 텐서가 없으면 (전체 데이터 재학습용) max_epochs를 그대로 모두 돌리고
    마지막 epoch 가중치를 사용합니다.
    """
    torch.manual_seed(seed)
    device = device or get_device()
    model = TabularMLP(
        num_numeric_feats=num_numeric_feats, cat_dims=cat_dims, embed_dims=embed_dims,
        hidden1=hidden1, hidden2=hidden2, dropout=dropout,
        bin_edges=bin_edges, quantile_d=quantile_d,
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    has_val = X_val_cat is not None and y_val is not None and len(y_val) > 0

    best_state = None
    best_epoch = 0
    best_val_brier = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_cat, batch_num, batch_y in loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_cat, batch_num)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
            val_brier, val_bss, val_score = compute_bss(val_preds, y_val)
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f} | Val Brier: {val_brier:.6f} | Val Score: {val_score:.2f}")

            if val_brier < best_val_brier - 1e-9:
                best_val_brier = val_brier
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"[EarlyStopping] Val Brier 개선 없음 {patience} epoch 지속 -> epoch {epoch}에서 종료 (best epoch: {best_epoch})")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f}")
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def train_ensemble(
    X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric_feats=None,
    X_val_cat=None, X_val_num=None, y_val=None,
    seeds=ENSEMBLE_SEEDS, embed_dims=None,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
    lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True,
    hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT,
    bin_edges=None, quantile_d=QUANTILE_D,
):
    """서로 다른 시드로 TabularMLP를 여러 개 학습해 멤버 리스트를 반환합니다.

    단일 모델은 epoch별 Val Brier 분산이 커서(같은 설정으로도 Val Score가
    수백 점 단위로 흔들림), 여러 시드의 예측을 평균 내는 앙상블이 실측
    데이터에서 안정적으로 더 높은 점수를 냈습니다 (ENSEMBLE_SEEDS 주석 참고).
    """
    device = device or get_device()
    members = []
    for seed in seeds:
        model, best_epoch = train_mlp(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=num_numeric_feats,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
            embed_dims=embed_dims, max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            device=device, verbose=verbose, seed=seed,
            hidden1=hidden1, hidden2=hidden2, dropout=dropout,
            bin_edges=bin_edges, quantile_d=quantile_d,
        )
        members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": best_epoch,
            "seed": seed,
        })
        if verbose:
            print(f"[Ensemble] seed={seed} 학습 완료 (best_epoch={best_epoch})")
    return members


def predict_ensemble(members, cat_dims, num_numeric_feats, embed_dims, X_cat, X_num, device=None,
                      hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT,
                      bin_edges=None, quantile_d=QUANTILE_D):
    """멤버별 예측 확률을 평균해 앙상블 예측을 반환합니다."""
    device = device or get_device()
    preds_list = []
    with torch.no_grad():
        for member in members:
            model = TabularMLP(
                num_numeric_feats=num_numeric_feats, cat_dims=cat_dims, embed_dims=embed_dims,
                hidden1=hidden1, hidden2=hidden2, dropout=dropout,
                bin_edges=bin_edges, quantile_d=quantile_d,
            ).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            preds_list.append(model(X_cat.to(device), X_num.to(device)).cpu().numpy())
    return np.mean(preds_list, axis=0)


def make_bundle(members, cat_cols, num_cols, cat_dims, embed_dims, cat_encoder, num_imputer, num_scaler,
                 hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT,
                 bin_edges=None, quantile_d=QUANTILE_D):
    """pickle로 저장 가능한 순수 dict 형태의 앙상블 모델 번들을 만듭니다.

    ("members" 키의 존재로 구버전 단일-모델 번들("state_dict" 키만 있음)과 구분합니다.)
    `bin_edges`가 없으면(과거 raw-concat 번들과의 하위 호환) "bin_edges" 키를 아예 담지
    않습니다 — `predict_bundle`이 `.get("bin_edges")`로 조회해 자동으로 raw-concat 경로로
    빠지므로 구버전 번들도 그대로 로드/추론 가능합니다.
    """
    avg_best_epoch = int(round(np.mean([m["best_epoch"] for m in members]))) if members else 0
    bundle = {
        "members": members,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_dims": cat_dims,
        "embed_dims": list(embed_dims),
        "cat_encoder": cat_encoder,
        "num_imputer": num_imputer,
        "num_scaler": num_scaler,
        "best_epoch_": avg_best_epoch,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "dropout": dropout,
    }
    if bin_edges is not None:
        bundle["bin_edges"] = bin_edges
        bundle["quantile_d"] = quantile_d
    return bundle


def predict_bundle(bundle, df, device=None):
    device = device or get_device()
    df_proc = apply_preprocessing(
        df, bundle["cat_cols"], bundle["num_cols"],
        bundle["cat_encoder"], bundle["num_imputer"], bundle["num_scaler"],
    )
    X_cat, X_num = to_tensors(df_proc, bundle["cat_cols"], bundle["num_cols"])
    return predict_ensemble(
        bundle["members"], bundle["cat_dims"], len(bundle["num_cols"]), bundle["embed_dims"],
        X_cat, X_num, device=device,
        hidden1=bundle.get("hidden1", HIDDEN1), hidden2=bundle.get("hidden2", HIDDEN2),
        dropout=bundle.get("dropout", DROPOUT),
        bin_edges=bundle.get("bin_edges"), quantile_d=bundle.get("quantile_d", QUANTILE_D),
    )
