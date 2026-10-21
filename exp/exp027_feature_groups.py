"""exp027: LightGBM 単体で、使う特徴量のグループを減らしたときの精度を比べる。

exp026 の特徴量（数値 80 列 → 埋め込みベクトル 2,048 列 → TF-IDF ベクトル約 20 万列）から列のグループを選び、
exp026 の LightGBM（lgb_base）と同じパラメータ・同じ fold・fold ごとにそろえた test の特徴量で学習する。
全体の CV に加えて、商品の新しさ（product_uid）の帯ごとの誤差と、test の帯の割合に合わせた重み付きの CV を出す。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import sparse

import exp023_members as m23
from src.utils import OUTPUT_DIR, ROOT, get_logger, rmse

UID_BINS = [100000, 150000, 180000, 195000, 206650]


def main(feat_dir=ROOT / "data" / "features" / "exp026", out_dir=OUTPUT_DIR / "exp027", group_names=None):
    """feat_dir の特徴量で、group_names（None なら全部）の構成を比べ、out_dir に保存する"""
    m23.FEAT_DIR, OUT_DIR = feat_dir, out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(OUT_DIR / "log.txt")
    data = m23.Data()
    y, folds, n, n_emb = data.y, data.folds, data.n_dense, data.n_emb
    X, X_test = data.sparse
    n_all = X.shape[1]
    dense_k = [np.load(m23.FEAT_DIR / f"X_test_dense_fold{k}.npy") for k in range(5)]
    test_rest = X_test[:, n:].tocsr()

    groups = {
        "all": np.arange(n_all),
        "num_emb": np.arange(n + n_emb),
        "num_tfidf": np.r_[np.arange(n), np.arange(n + n_emb, n_all)],
        "num": np.arange(n),
    }
    if group_names:
        groups = {k: groups[k] for k in group_names}

    tr = pd.read_csv(ROOT / "input" / "train.csv", encoding="ISO-8859-1")
    te = pd.read_csv(ROOT / "input" / "test.csv", encoding="ISO-8859-1")
    bins = UID_BINS + [300000]
    tr_bin, te_bin = pd.cut(tr.product_uid, bins), pd.cut(te.product_uid, bins)
    # test の帯の割合に合わせた重み（train にない新しさの帯は、train で最も新しい帯で代用する）
    te_share = te_bin.value_counts(normalize=True)
    te_share.iloc[-2] += te_share.iloc[-1]
    w = tr_bin.map(te_share / tr_bin.value_counts(normalize=True)).astype(float).to_numpy()

    fp = m23.lgb_member(m23.LGB_BASE, seed=42)
    rows = []
    for name, cols in groups.items():
        Xg = X[:, cols].tocsr()
        oof, pred_test = np.zeros(len(y)), np.zeros(X_test.shape[0])
        for k in range(5):
            te_k = sparse.hstack([sparse.csr_matrix(dense_k[k]), test_rest], format="csr")[:, cols]
            trm, va = folds != k, folds == k
            p_va, p_te = fp(Xg[trm], y[trm], Xg[va], y[va], te_k)
            oof[va], pred_test = p_va, pred_test + p_te / 5
        oof, pred_test = np.clip(oof, 1, 3), np.clip(pred_test, 1, 3)
        r = {"group": name, "n_cols": len(cols), "cv": rmse(y, oof),
             "cv_weighted": np.sqrt(np.average((oof - y) ** 2, weights=w))}
        for b in tr_bin.cat.categories[:-1]:
            m = (tr_bin == b).to_numpy()
            r[f"rmse_{b.left // 1000}k-{b.right // 1000}k"] = rmse(y[m], oof[m])
        rows.append(r)
        logger.info(f"{name}: " + ", ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
        np.savez(OUT_DIR / f"{name}.npz", oof=oof, test=pred_test)
        pd.DataFrame({"id": data.meta["test_id"], "relevance": pred_test}).to_csv(OUT_DIR / f"submission_{name}.csv", index=False)
    logger.info("summary:\n" + pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
