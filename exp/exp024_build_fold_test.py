"""exp024（準備）: test のファインチューニングのスコアを、fold ごとのモデルの予測にそろえた特徴量を作る。

exp022 の特徴量では、ファインチューニングした cross-encoder のスコアが、train は「その行を学習に使っていない 1 つの
モデル」の予測（OOF）、test は「5 つのモデルの平均」になっていて、性質が違う。そこで fold k ごとに、test の
スコアを「fold k のモデルの予測」に置き換えた数値の特徴量（先頭 80 列）を作り、
data/features/exp022/X_test_dense_fold{k}.npy に保存する。train の特徴量は変えない。

置き換えるのは、ファインチューニングした 4 つのスコアと、それらの同じ検索語の中での相対値（q_*）だけ。
相対値は、train（OOF）と test（fold k のスコア）を合わせて、元の検索語ごとに集計し直す。
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import load_data
from src.features import group_relative_features
from src.preprocess import _map_unique, normalize
from src.spell import build_symspell, make_query_corrector
from src.utils import ROOT, get_logger

FEAT_DIR = ROOT / "data" / "features" / "exp022"
PROC = ROOT / "data" / "processed"
# (特徴量の列名, ファインチューニングの結果のフォルダ, 入力の文書の種類)
FT_SOURCES = [
    ("ft_ce_title", PROC / "ft_ce_bge_base_title", "title"),
    ("ft_ce_title_desc", PROC / "ft_ce_bge_base_title_desc", "title_desc"),
    ("ft_ce_large_title", PROC / "ft_ce_bge_v2m3_title", "title"),
    ("ft_ce_large_title_desc", PROC / "ft_ce_bge_v2m3_title_desc", "title_desc"),
]
N_FOLDS = 5


def per_fold_test_scores(ft_dir: Path, test_keys: pd.Series) -> list[np.ndarray]:
    """fold ごとの test の予測（ユニークなペアの順）を、test の行の順に並べ直して返す"""
    pairs = json.loads((ft_dir / "input.json").read_text(encoding="utf-8"))["test"]
    index = {q + "\x1f" + d: i for i, (q, d) in enumerate(pairs)}
    rows = test_keys.map(index).to_numpy()
    assert not np.isnan(rows.astype(float)).any(), f"{ft_dir.name}: test のペアが見つからない"
    return [np.load(ft_dir / f"test_fold{k}.npy")[rows.astype(int)] for k in range(N_FOLDS)]


def main():
    logger = get_logger(ROOT / "output" / "exp024_fold_test_log.txt")
    meta = json.loads((FEAT_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n_dense = meta["feature_names"], meta["n_dense"]
    dense_names = names[:n_dense]

    # 検索語は正規化 + SymSpell 後（ファインチューニングの入力と同じ）、文書は元のテキスト
    train, test = load_data()
    assert train["id"].tolist() == meta["train_id"] and test["id"].tolist() == meta["test_id"]
    df = pd.concat([train, test], ignore_index=True)
    products = df.drop_duplicates("product_uid")
    corrector = make_query_corrector(build_symspell(
        _map_unique(products["product_title"] + " " + products["product_description"], normalize)))
    query_text = corrector(_map_unique(test["search_term"], normalize))
    docs = {"title": test["product_title"], "title_desc": test["product_title"] + ". " + test["product_description"]}

    X, X_test = sparse.load_npz(FEAT_DIR / "X_train.npz"), sparse.load_npz(FEAT_DIR / "X_test.npz")
    dense_tr = pd.DataFrame(X[:, :n_dense].toarray(), columns=dense_names)
    dense_te = pd.DataFrame(X_test[:, :n_dense].toarray(), columns=dense_names)
    group_key = pd.Series(meta["groups"] + test["search_term"].tolist())

    fold_scores = {}
    for col, ft_dir, doc in FT_SOURCES:
        scores = per_fold_test_scores(ft_dir, query_text + "\x1f" + docs[doc])
        # 確認: fold の平均は、exp022 の test の特徴量と一致するはず
        diff = np.abs(np.mean(scores, axis=0) - dense_te[col].to_numpy()).max()
        logger.info(f"{col}: per-fold test scores loaded, max |mean - exp022| = {diff:.2e}, "
                    f"std of fold mean {np.mean(scores, axis=0).std():.4f} vs single fold {np.mean([s.std() for s in scores]):.4f}")
        assert diff < 1e-4
        fold_scores[col] = scores

    ft_cols = [c for c, _, _ in FT_SOURCES]
    for k in range(N_FOLDS):
        te_k = dense_te.copy()
        for col in ft_cols:
            te_k[col] = fold_scores[col][k]
        combined = pd.concat([dense_tr, te_k], ignore_index=True)
        grp = group_relative_features(combined, group_key, ft_cols, prefix="q").iloc[len(dense_tr):]
        for c in grp.columns:
            if c != "q_size":
                te_k[c] = grp[c].to_numpy()
        np.save(FEAT_DIR / f"X_test_dense_fold{k}.npy", te_k.to_numpy(dtype=np.float32))
        changed = [c for c in dense_names if not np.allclose(te_k[c], dense_te[c], equal_nan=True)]
        logger.info(f"fold {k}: saved, changed columns ({len(changed)}): {changed}")


if __name__ == "__main__":
    main()
