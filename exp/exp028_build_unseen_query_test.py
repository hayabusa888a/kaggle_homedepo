"""exp028（準備）: test のファインチューニングのスコアを、すべて「その検索語を学習で見ていないモデル」の予測にそろえる。

ファインチューニングは検索語単位の GroupKFold で行ったので、train の OOF のスコアは、すべて「その検索語を学習で
見ていないモデル」の予測。一方 test の 27% の行は train にある検索語で、そのままでは検索語を学習で見たモデルの
スコアになり、高めに出る（exp026 の誤差分析）。train の各検索語は 5 つの fold のどれか 1 つにだけ入るので、
その fold のモデルはその検索語を見ていない。そこで、fold k ごとの test の特徴量を次のように作る。
- train にある検索語の行: その検索語の fold（q_fold）のモデルのスコア（stage-2 の fold k によらない）
- train にない検索語の行: fold k のモデルのスコア（exp024 と同じ）
2 つの種で平均している列は、種ごとに同じ選び方をしてから平均する。相対値（q_*）も作り直す。
もとの特徴量は exp026（data/features/exp026/）。train の特徴量は変えない。
"""
import json
import os
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

SRC_DIR = ROOT / "data" / "features" / "exp026"
DST_DIR = ROOT / "data" / "features" / "exp028"
PROC = ROOT / "data" / "processed"
# (列名, 種ごとのフォルダ, 入力の文書の種類)
FT_SOURCES = [
    ("ft_ce_title", [PROC / "ft_ce_bge_base_title"], "title"),
    ("ft_ce_title_desc", [PROC / "ft_ce_bge_base_title_desc"], "title_desc"),
    ("ft_ce_large_title", [PROC / "ft_ce_bge_v2m3_title", PROC / "ft_ce_bge_v2m3_title_seed2"], "title"),
    ("ft_ce_large_title_desc", [PROC / "ft_ce_bge_v2m3_title_desc", PROC / "ft_ce_bge_v2m3_title_desc_seed2"], "title_desc"),
]


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(ROOT / "output" / "exp028_features_log.txt")
    meta = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n = meta["feature_names"], meta["n_dense"]

    # train の特徴量・test の残りの列は exp026 と同じなので、ファイルはハードリンクで共有する
    for f in ["X_train.npz", "X_test.npz", "y.npy", "meta.json"]:
        if not (DST_DIR / f).exists():
            os.link(SRC_DIR / f, DST_DIR / f)

    train, test = load_data()
    groups = pd.Series(meta["groups"])
    q_fold_map = dict(zip(groups, fold_ids(groups)))  # 検索語 -> その検索語が検証側に入る fold
    q_fold = test["search_term"].map(q_fold_map)
    seen = q_fold.notna().to_numpy()
    q_fold = q_fold.fillna(-1).astype(int).to_numpy()
    logger.info(f"test rows with a train query: {seen.mean():.1%}")

    df = pd.concat([train, test], ignore_index=True)
    products = df.drop_duplicates("product_uid")
    corrector = make_query_corrector(build_symspell(
        _map_unique(products["product_title"] + " " + products["product_description"], normalize)))
    query_text = corrector(_map_unique(test["search_term"], normalize))
    docs = {"title": test["product_title"], "title_desc": test["product_title"] + ". " + test["product_description"]}

    # 列ごと・fold k ごとの test のスコア（種の平均）
    scores = {}
    rows = np.arange(len(test))
    for col, dirs, doc in FT_SOURCES:
        per_seed = [np.stack(per_fold_test_scores(d, query_text + "\x1f" + docs[doc])) for d in dirs]  # (5, n_test)
        S = np.mean(per_seed, axis=0)
        unseen_q = S[q_fold, rows]  # その検索語を見ていない fold のモデルのスコア
        scores[col] = [np.where(seen, unseen_q, S[k]) for k in range(5)]
        diff = (S.mean(axis=0) - unseen_q)[seen].mean()
        logger.info(f"{col}: seen-query rows, mean score of all folds - unseen-query fold = {diff:+.4f} (0-1 scale)")

    X = sparse.load_npz(SRC_DIR / "X_train.npz")
    dense_tr = pd.DataFrame(X[:, :n].toarray(), columns=names[:n])
    group_key = pd.Series(meta["groups"] + test["search_term"].tolist())
    ft_cols = [c for c, _, _ in FT_SOURCES]
    for k in range(5):
        te_k = pd.DataFrame(np.load(SRC_DIR / f"X_test_dense_fold{k}.npy"), columns=names[:n])
        for col in ft_cols:
            te_k[col] = scores[col][k]
        grp = group_relative_features(pd.concat([dense_tr, te_k], ignore_index=True), group_key, ft_cols, prefix="q").iloc[len(dense_tr):]
        for c in grp.columns:
            if c != "q_size":
                te_k[c] = grp[c].to_numpy()
        np.save(DST_DIR / f"X_test_dense_fold{k}.npy", te_k.to_numpy(dtype=np.float32))
    logger.info(f"saved: {DST_DIR}")


if __name__ == "__main__":
    main()
