"""exp024: exp023 のアンサンブルのメンバーを、fold ごとにそろえた test の特徴量で学習し直す。

fold k で学習したモデルは、test の予測に「fold k のファインチューニングのモデルが出したスコア」を使った特徴量
（exp024_build_fold_test.py で作成）を使い、5 つの予測を平均する。train（OOF）の特徴量は exp023 と同じ。
MLP（--mlp）は GPU 用の仮想環境で実行する:
    python exp/exp024_members.py
    .venv-gpu/Scripts/python.exe exp/exp024_members.py --mlp
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from scipy import sparse

import exp023_members as m23
from src.utils import OUTPUT_DIR, get_logger, rmse

OUT_DIR = OUTPUT_DIR / "exp024"
MEMBER_DIR = OUT_DIR / "members"


class FoldTest:
    """使う特徴量の種類ごとに、train の特徴量と、fold k の test の特徴量を作る"""

    def __init__(self, data: m23.Data):
        self.data = data
        self.dense_k = [np.load(m23.FEAT_DIR / f"X_test_dense_fold{k}.npy") for k in range(5)]

    def prepare(self, kind):
        d, n = self.data, self.data.n_dense
        X, X_test = d.sparse
        if kind == "sparse":
            rest = X_test[:, n:].tocsr()
            return X, lambda k: sparse.hstack([sparse.csr_matrix(self.dense_k[k]), rest], format="csr")
        if kind in ("dense", "num_emb"):
            Xd, Xd_test = d.dense(n + d.n_emb)
            emb_test = Xd_test[:, n:]
            make = lambda k: np.hstack([self.dense_k[k], emb_test])
        elif kind == "num":
            Xd = d.dense(n)[0]
            make = lambda k: self.dense_k[k]
        elif kind == "knn_a":
            idx = [d.names.index(c) for c in m23.KNN_COLS_A]
            Xd = X[:, idx].toarray()
            make = lambda k: self.dense_k[k][:, idx]
        if kind == "dense":
            return Xd, make
        # 欠損の補完と標準化は、train の統計量で test にも同じ変換をかける
        med = np.nanmedian(Xd, axis=0)
        med = np.where(np.isnan(med), 0, med)
        Xf = np.where(np.isnan(Xd), med, Xd)
        mu, sd = Xf.mean(axis=0), Xf.std(axis=0)
        sd[sd == 0] = 1
        tf = lambda A: ((np.where(np.isnan(A), med, A) - mu) / sd).astype(np.float32)
        return tf(Xd), lambda k: tf(make(k))


def run_cv(name, data, X, make_test, fit_predict, logger):
    y, folds = data.y, data.folds
    oof, pred_test = np.zeros(len(y)), None
    t0 = time.time()
    for k in range(5):
        tr, va = folds != k, folds == k
        X_te = make_test(k)
        p_va, p_te = fit_predict(X[tr], y[tr], X[va], y[va], X_te)
        oof[va] = p_va
        pred_test = p_te / 5 if pred_test is None else pred_test + p_te / 5
        logger.info(f"{name}: fold {k} rmse={rmse(y[va], np.clip(p_va, 1, 3)):.5f} ({time.time() - t0:.0f}s)")
    oof, pred_test = np.clip(oof, 1, 3), np.clip(pred_test, 1, 3)
    MEMBER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MEMBER_DIR / f"{name}.npz", oof=oof, test=pred_test)
    logger.info(f"{name}: CV rmse={rmse(y, oof):.5f} ({(time.time() - t0) / 60:.1f}min)")


def mlp_members():
    import exp023_mlp as mlp
    return [(name, kind, (lambda h, dr, s: (lambda X_tr, y_tr, X_va, y_va, X_te:
                                             mlp.fit_predict(X_tr, y_tr, X_va, y_va, X_te, h, dr, s)))(hidden, dropout, seed))
            for name, kind, hidden, dropout, seed in mlp.MLP_MEMBERS]


def main():
    use_mlp = "--mlp" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(OUT_DIR / ("mlp_log.txt" if use_mlp else "members_log.txt"))
    data = m23.Data()
    ft = FoldTest(data)
    if use_mlp:
        members = mlp_members()
    else:
        # exp022 の LightGBM と同じ設定（シード 42）も、fold ごとの test で学習し直してメンバーにする
        members = [("lgb_base", "sparse", m23.lgb_member(m23.LGB_BASE, seed=42))] + m23.build_members()
    cache = {}
    for name, kind, fp in sorted(members, key=lambda m: m[1]):
        if (MEMBER_DIR / f"{name}.npz").exists():
            logger.info(f"{name}: cached")
            continue
        if kind not in cache:
            cache.clear()
            cache[kind] = ft.prepare(kind)
        X, make_test = cache[kind]
        logger.info(f"{name}: start ({kind}, {X.shape[1]} cols)")
        run_cv(name, data, X, make_test, fp, logger)


if __name__ == "__main__":
    main()
