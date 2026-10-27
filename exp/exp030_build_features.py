"""exp030（準備）: exp029 の特徴量に、ファインチューニングした DeBERTa（mxbai-rerank-base-v1、検索語 × タイトル）
のスコアと、その同じ検索語の中での相対値を加えた特徴量を data/features/exp030/ に作る。

- train: OOF（その行を学習に使っていない fold のモデルの予測）
- test（X_test.npz）: 5 つの fold のモデルの予測の平均
- test（X_test_dense_fold{k}.npy）: fold k のモデルの予測（exp024 の方式）
数値の特徴量の末尾に 4 列を加えるので、n_dense が 84 → 88 になる。
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import sparse

from exp024_build_fold_test import per_fold_test_scores
from src.data import load_data
from src.features import group_relative_features
from src.finetune import fold_ids
from src.preprocess import _map_unique, normalize
from src.spell import build_symspell, make_query_corrector
from src.utils import ROOT, get_logger

SRC_DIR = ROOT / "data" / "features" / "exp029"
DST_DIR = ROOT / "data" / "features" / "exp030"
FT_DIR = ROOT / "data" / "processed" / "ft_ce_mxbai_base_title"
COL = "ft_ce_deberta_title"


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(ROOT / "output" / "exp030_features_log.txt")
    meta = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n = meta["feature_names"], meta["n_dense"]
    folds = fold_ids(pd.Series(meta["groups"]))
    y = np.load(SRC_DIR / "y.npy")

    train, test = load_data()
    df = pd.concat([train, test], ignore_index=True)
    products = df.drop_duplicates("product_uid")
    corrector = make_query_corrector(build_symspell(
        _map_unique(products["product_title"] + " " + products["product_description"], normalize)))
    test_keys = corrector(_map_unique(test["search_term"], normalize)) + "\x1f" + test["product_title"]

    oof = np.zeros(len(y), dtype=np.float32)
    for k in range(5):
        oof[folds == k] = np.load(FT_DIR / f"oof_fold{k}.npy")
    test_k = per_fold_test_scores(FT_DIR, test_keys)
    logger.info(f"{COL}: standalone OOF rmse {np.sqrt(np.mean((oof * 2 + 1 - y) ** 2)):.5f}")

    X, X_test = sparse.load_npz(SRC_DIR / "X_train.npz").tocsr(), sparse.load_npz(SRC_DIR / "X_test.npz").tocsr()
    dense_tr = pd.DataFrame(X[:, :n].toarray(), columns=names[:n])
    group_key = pd.Series(meta["groups"] + test["search_term"].tolist())

    def add(tr: pd.DataFrame, te: pd.DataFrame, te_scores: np.ndarray):
        """新しい列とその相対値を末尾に加える（相対値は train + test を合わせて集計する）"""
        tr, te = tr.copy(), te.copy()
        tr[COL], te[COL] = oof, te_scores
        grp = group_relative_features(pd.concat([tr, te], ignore_index=True), group_key, [COL], prefix="q")
        for c in [f"q_{COL}_diff_mean", f"q_{COL}_diff_max", f"q_{COL}_rank"]:
            tr[c], te[c] = grp[c].to_numpy()[: len(tr)], grp[c].to_numpy()[len(tr):]
        return tr, te

    tr_new, te_new = add(dense_tr, pd.DataFrame(X_test[:, :n].toarray(), columns=names[:n]), np.mean(test_k, axis=0))
    sparse.save_npz(DST_DIR / "X_train.npz", sparse.hstack([sparse.csr_matrix(tr_new.to_numpy(np.float32)), X[:, n:]], format="csr"), compressed=False)
    sparse.save_npz(DST_DIR / "X_test.npz", sparse.hstack([sparse.csr_matrix(te_new.to_numpy(np.float32)), X_test[:, n:]], format="csr"), compressed=False)
    np.save(DST_DIR / "y.npy", y)
    new_names = list(tr_new.columns)
    meta.update({"feature_names": new_names + names[n:], "n_dense": len(new_names)})
    (DST_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    for k in range(5):
        te_k = pd.DataFrame(np.load(SRC_DIR / f"X_test_dense_fold{k}.npy"), columns=names[:n])
        _, te_k = add(dense_tr, te_k, test_k[k])
        np.save(DST_DIR / f"X_test_dense_fold{k}.npy", te_k.to_numpy(dtype=np.float32))
    logger.info(f"saved: {DST_DIR} (n_dense {len(new_names)})")


if __name__ == "__main__":
    main()
