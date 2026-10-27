"""exp031〜（準備）: もとの特徴量に、ファインチューニングした cross-encoder のスコア（複数可）と、その同じ検索語の中での
相対値を加えた特徴量を作る（exp029・exp030 の準備を、複数のモデルに使えるようにしたもの）。

- train: OOF（その行を学習に使っていない fold のモデルの予測）
- test（X_test.npz）: 5 つの fold のモデルの予測の平均
- test（X_test_dense_fold{k}.npy）: fold k のモデルの予測（exp024 の方式）
数値の特徴量の末尾に、モデルごとに 4 列を加える。
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

PROC = ROOT / "data" / "processed"
FEAT = ROOT / "data" / "features"
# 実験ごとの (もとの特徴量, 加えるモデルの [(列名, フォルダ, 入力の文書の種類)])
CONFIGS = {
    "exp031": ("exp029", [("ft_ce_modernbert_title_desc", "ft_ce_gte_modernbert_title_desc", "title_desc")]),
    "exp032": ("exp029", [("ft_ce_qwen3_title", "ft_ce_qwen3_06b_title", "title")]),
    "exp033": ("exp029", [("ft_ce_deberta_title", "ft_ce_mxbai_base_title", "title"),
                          ("ft_ce_modernbert_title_desc", "ft_ce_gte_modernbert_title_desc", "title_desc"),
                          ("ft_ce_qwen3_title", "ft_ce_qwen3_06b_title", "title")]),
}


def main(exp):
    src_name, sources = CONFIGS[exp]
    SRC_DIR, DST_DIR = FEAT / src_name, FEAT / exp
    DST_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(ROOT / "output" / f"{exp}_features_log.txt")
    meta = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n = meta["feature_names"], meta["n_dense"]
    folds = fold_ids(pd.Series(meta["groups"]))
    y = np.load(SRC_DIR / "y.npy")

    train, test = load_data()
    df = pd.concat([train, test], ignore_index=True)
    products = df.drop_duplicates("product_uid")
    corrector = make_query_corrector(build_symspell(
        _map_unique(products["product_title"] + " " + products["product_description"], normalize)))
    query_text = corrector(_map_unique(test["search_term"], normalize))
    docs = {"title": test["product_title"], "title_desc": test["product_title"] + ". " + test["product_description"]}

    oofs, tests = {}, {}
    for col, ft_name, doc in sources:
        oof = np.zeros(len(y), dtype=np.float32)
        for k in range(5):
            oof[folds == k] = np.load(PROC / ft_name / f"oof_fold{k}.npy")
        oofs[col], tests[col] = oof, per_fold_test_scores(PROC / ft_name, query_text + "\x1f" + docs[doc])
        logger.info(f"{col}: standalone OOF rmse {np.sqrt(np.mean((oof * 2 + 1 - y) ** 2)):.5f}, "
                    f"pearson {np.corrcoef(oof, y)[0, 1]:.4f}")
    cols = list(oofs)

    X, X_test = sparse.load_npz(SRC_DIR / "X_train.npz").tocsr(), sparse.load_npz(SRC_DIR / "X_test.npz").tocsr()
    dense_tr = pd.DataFrame(X[:, :n].toarray(), columns=names[:n])
    ft_old = [c for c in names[:n] if c.startswith("ft_ce_")]
    corr = pd.concat([pd.DataFrame(oofs), dense_tr[ft_old]], axis=1).corr()
    logger.info("correlation of OOF scores:\n" + corr.loc[cols].round(3).to_string())
    group_key = pd.Series(meta["groups"] + test["search_term"].tolist())

    def add(tr: pd.DataFrame, te: pd.DataFrame, te_scores: dict):
        """新しい列とその相対値を末尾に加える（相対値は train + test を合わせて集計する）"""
        tr, te = tr.copy(), te.copy()
        for c in cols:
            tr[c], te[c] = oofs[c], te_scores[c]
        grp = group_relative_features(pd.concat([tr, te], ignore_index=True), group_key, cols, prefix="q")
        for c in cols:
            for g in [f"q_{c}_diff_mean", f"q_{c}_diff_max", f"q_{c}_rank"]:
                tr[g], te[g] = grp[g].to_numpy()[: len(tr)], grp[g].to_numpy()[len(tr):]
        return tr, te

    te_mean = {c: np.mean(tests[c], axis=0) for c in cols}
    tr_new, te_new = add(dense_tr, pd.DataFrame(X_test[:, :n].toarray(), columns=names[:n]), te_mean)
    sparse.save_npz(DST_DIR / "X_train.npz", sparse.hstack([sparse.csr_matrix(tr_new.to_numpy(np.float32)), X[:, n:]], format="csr"), compressed=False)
    sparse.save_npz(DST_DIR / "X_test.npz", sparse.hstack([sparse.csr_matrix(te_new.to_numpy(np.float32)), X_test[:, n:]], format="csr"), compressed=False)
    np.save(DST_DIR / "y.npy", y)
    new_names = list(tr_new.columns)
    meta.update({"feature_names": new_names + names[n:], "n_dense": len(new_names)})
    (DST_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    for k in range(5):
        te_k = pd.DataFrame(np.load(SRC_DIR / f"X_test_dense_fold{k}.npy"), columns=names[:n])
        _, te_k = add(dense_tr, te_k, {c: tests[c][k] for c in cols})
        np.save(DST_DIR / f"X_test_dense_fold{k}.npy", te_k.to_numpy(dtype=np.float32))
    logger.info(f"saved: {DST_DIR} (n_dense {len(new_names)})")


if __name__ == "__main__":
    main(sys.argv[1])
