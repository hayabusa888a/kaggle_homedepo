"""exp025（準備）: v2-m3（タイトル + 説明文）のスコアを、2 つの乱数の種のモデルの平均に置き換えた特徴量を作る。

exp022 の特徴量（data/features/exp022/）をもとに、ft_ce_large_title_desc と、その同じ検索語の中での相対値
（q_ft_ce_large_title_desc_*）だけを置き換えて、data/features/exp025/ に保存する。
- train: 2 つの種の OOF の平均
- test（X_test.npz）: 2 つの種 × 5 fold の平均
- test（X_test_dense_fold{k}.npy、exp024 の方式）: 2 つの種の fold k の予測の平均
"""
import json
import shutil
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

PROC = ROOT / "data" / "processed"


def main(src_dir=ROOT / "data" / "features" / "exp022", dst_dir=ROOT / "data" / "features" / "exp025",
         seed_dirs=(PROC / "ft_ce_bge_v2m3_title_desc", PROC / "ft_ce_bge_v2m3_title_desc_seed2"),
         col="ft_ce_large_title_desc", doc="title_desc", log_name="exp025_features_log.txt"):
    """src_dir の特徴量の col を、seed_dirs のモデルの予測の平均に置き換えて dst_dir に保存する。
    doc はファインチューニングの入力の文書（"title" または "title_desc"）"""
    SRC_DIR, DST_DIR, SEED_DIRS, COL = src_dir, dst_dir, list(seed_dirs), col
    DST_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(ROOT / "output" / log_name)
    meta = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n = meta["feature_names"], meta["n_dense"]
    folds = fold_ids(pd.Series(meta["groups"]))

    train, test = load_data()
    df = pd.concat([train, test], ignore_index=True)
    products = df.drop_duplicates("product_uid")
    corrector = make_query_corrector(build_symspell(
        _map_unique(products["product_title"] + " " + products["product_description"], normalize)))
    doc_text = test["product_title"] + (". " + test["product_description"] if doc == "title_desc" else "")
    test_keys = corrector(_map_unique(test["search_term"], normalize)) + "\x1f" + doc_text

    # train の OOF（fold k の行の順に保存されている）と、fold ごとの test の予測を、種ごとに読み込む
    oofs, tests = [], []
    for d in SEED_DIRS:
        oof = np.zeros(len(folds), dtype=np.float32)
        for k in range(5):
            oof[folds == k] = np.load(d / f"oof_fold{k}.npy")
        oofs.append(oof)
        tests.append(per_fold_test_scores(d, test_keys))
    y = np.load(SRC_DIR / "y.npy")
    for i, o in enumerate(oofs):
        logger.info(f"seed {i + 1}: standalone OOF rmse {np.sqrt(np.mean((o * 2 + 1 - y) ** 2)):.5f}")
    oof_mean = np.mean(oofs, axis=0)
    logger.info(f"mean of 2 seeds: standalone OOF rmse {np.sqrt(np.mean((oof_mean * 2 + 1 - y) ** 2)):.5f}, "
                f"corr between seeds {np.corrcoef(oofs[0], oofs[1])[0, 1]:.4f}")

    X, X_test = sparse.load_npz(SRC_DIR / "X_train.npz").tocsr(), sparse.load_npz(SRC_DIR / "X_test.npz").tocsr()
    dense_tr = pd.DataFrame(X[:, :n].toarray(), columns=names[:n])
    dense_te = pd.DataFrame(X_test[:, :n].toarray(), columns=names[:n])
    group_key = pd.Series(meta["groups"] + test["search_term"].tolist())
    rel_cols = [f"q_{COL}_diff_mean", f"q_{COL}_diff_max", f"q_{COL}_rank"]

    def replace(tr: pd.DataFrame, te: pd.DataFrame, te_scores: np.ndarray):
        """COL を置き換え、その相対値を train + test を合わせて作り直す"""
        tr, te = tr.copy(), te.copy()
        tr[COL], te[COL] = oof_mean, te_scores
        grp = group_relative_features(pd.concat([tr, te], ignore_index=True), group_key, [COL], prefix="q")
        for c in rel_cols:
            tr[c], te[c] = grp[c].to_numpy()[: len(tr)], grp[c].to_numpy()[len(tr):]
        return tr, te

    test_mean = np.mean([s for seed in tests for s in seed], axis=0)
    tr_new, te_new = replace(dense_tr, dense_te, test_mean)
    sparse.save_npz(DST_DIR / "X_train.npz", sparse.hstack([sparse.csr_matrix(tr_new.to_numpy(np.float32)), X[:, n:]], format="csr"), compressed=False)
    sparse.save_npz(DST_DIR / "X_test.npz", sparse.hstack([sparse.csr_matrix(te_new.to_numpy(np.float32)), X_test[:, n:]], format="csr"), compressed=False)
    shutil.copy(SRC_DIR / "y.npy", DST_DIR / "y.npy")
    shutil.copy(SRC_DIR / "meta.json", DST_DIR / "meta.json")

    for k in range(5):
        te_k = pd.DataFrame(np.load(SRC_DIR / f"X_test_dense_fold{k}.npy"), columns=names[:n])
        _, te_k = replace(dense_tr, te_k, np.mean([tests[0][k], tests[1][k]], axis=0))
        np.save(DST_DIR / f"X_test_dense_fold{k}.npy", te_k.to_numpy(dtype=np.float32))
    logger.info(f"saved: {DST_DIR}")


if __name__ == "__main__":
    main()
