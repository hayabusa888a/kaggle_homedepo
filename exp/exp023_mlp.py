"""exp023: アンサンブルのメンバーのうち、MLP（ニューラルネット）を学習する。

PyTorch を使うので、GPU 用の仮想環境で実行する。特徴量の読み込みと保存の形式は exp023_members.py と同じ。
    .venv-gpu/Scripts/python.exe exp/exp023_mlp.py
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import nn

from exp023_members import MEMBER_DIR, OUT_DIR, Data, impute_scale
from src.utils import get_logger, rmse

# (名前, 使う特徴量, 隠れ層のユニット数, ドロップアウト, 乱数の種)
MLP_MEMBERS = [
    ("mlp_num_emb_512", "num_emb", [512, 128], 0.2, 42),
    ("mlp_num_emb_1024", "num_emb", [1024, 256, 64], 0.3, 7),
    ("mlp_num_256", "num", [256, 64], 0.1, 42),
]


def make_mlp(n_in, hidden, dropout):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def fit_predict(X_tr, y_tr, X_va, y_va, X_te, hidden, dropout, seed, max_epochs=40, patience=4, batch=512):
    """検証 fold の RMSE が patience エポック改善しなければ止め、最良のエポックの重みで予測する"""
    torch.manual_seed(seed)
    dev = "cuda"
    model = make_mlp(X_tr.shape[1], hidden, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    Xt, yt = torch.tensor(X_tr, device=dev), torch.tensor(y_tr, dtype=torch.float32, device=dev)
    Xv = torch.tensor(X_va, device=dev)

    def predict(X):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(torch.tensor(X[i:i + 8192], device=dev)).squeeze(1)
                              for i in range(0, len(X), 8192)]).cpu().numpy()

    best, best_state, bad = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(Xt[idx]).squeeze(1), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = rmse(y_va, np.clip(predict(X_va), 1, 3))
        sched.step(score)
        if score < best:
            best, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return predict(X_va), predict(X_te)


def main():
    logger = get_logger(OUT_DIR / "mlp_log.txt")
    data = Data()
    cache = {}
    for name, kind, hidden, dropout, seed in MLP_MEMBERS:
        if (MEMBER_DIR / f"{name}.npz").exists():
            logger.info(f"{name}: cached")
            continue
        if kind not in cache:
            cache.clear()
            n_cols = data.n_dense + (data.n_emb if kind == "num_emb" else 0)
            cache[kind] = impute_scale(*data.dense(n_cols))
        X, X_test = cache[kind]
        logger.info(f"{name}: start ({kind}, {X.shape[1]} cols, hidden {hidden}, dropout {dropout})")
        y, folds = data.y, data.folds
        oof, pred_test = np.zeros(len(y)), np.zeros(len(X_test))
        t0 = time.time()
        for k in range(5):
            tr, va = folds != k, folds == k
            p_va, p_te = fit_predict(X[tr], y[tr], X[va], y[va], X_test, hidden, dropout, seed + k)
            oof[va], pred_test = p_va, pred_test + p_te / 5
            logger.info(f"{name}: fold {k} rmse={rmse(y[va], np.clip(p_va, 1, 3)):.5f} ({time.time() - t0:.0f}s)")
        oof, pred_test = np.clip(oof, 1, 3), np.clip(pred_test, 1, 3)
        np.savez(MEMBER_DIR / f"{name}.npz", oof=oof, test=pred_test)
        logger.info(f"{name}: CV rmse={rmse(y, oof):.5f} ({(time.time() - t0) / 60:.1f}min)")


if __name__ == "__main__":
    main()
