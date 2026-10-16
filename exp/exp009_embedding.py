"""exp009: exp007 + 埋め込み（nomic-embed-text）の類似度 + 検索語・タイトルの埋め込みベクトル"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import load_attribute_texts, load_brands, load_data
from src.embedding import embed_texts
from src.features import (attr_match_features, bm25_features, brand_features, char_cosine_features,
                          cosine_features, match_features, tfidf_vectors)
from src.preprocess import _map_unique, normalize, preprocess, stem
from src.spell import build_symspell, make_query_corrector
from src.train import run_cv_lgb
from src.utils import ROOT, OUTPUT_DIR, get_logger, seed_everything, timer

EXP_NAME = "exp009"
SEED = 42
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "seed": SEED,
    "verbose": -1,
}
# 2文字以上の単語 + 1文字の数字（寸法 "2 x 4" の数字を残すため）
TOKEN_PATTERN = r"(?u)\b(?:\w\w+|\d)\b"
EMB_DIR = ROOT / "data" / "processed" / "emb_nomic"
EMB_DIM = 768


def main():
    out_dir = OUTPUT_DIR / EXP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(out_dir / "log.txt")
    seed_everything(SEED)

    with timer("load data", logger):
        train, test = load_data()
        df = pd.concat([train, test], ignore_index=True)
        # ハイフンは正規化の「連結形も残す」処理を通さず、単純にスペースにする ("Strong-Tie" -> "strong tie")
        raw_brands = load_brands()
        brands = raw_brands.str.replace("-", " ").map(lambda b: stem(normalize(b)))

    with timer("spell dictionary", logger):
        products = df.drop_duplicates("product_uid")
        product_texts = _map_unique(products["product_title"] + " " + products["product_description"], normalize)
        corrector = make_query_corrector(build_symspell(product_texts))

    with timer("embeddings", logger):
        # 埋め込みにはステミング前の自然な文を使う。検索語は正規化 + SymSpell 後（ステミング前）、商品側は元のテキスト
        query_text = corrector(_map_unique(df["search_term"], normalize))
        emb_q = embed_texts(query_text, "search_query: ", EMB_DIR / "query.npz", logger=logger)
        emb_t = embed_texts(df["product_title"], "search_document: ", EMB_DIR / "title.npz", logger=logger)
        emb_d = embed_texts(df["product_description"], "search_document: ", EMB_DIR / "desc.npz", batch_size=16,
                            logger=logger)
        brand_raw = df["product_uid"].map(raw_brands)
        has_brand = brand_raw.notna().to_numpy()
        emb_b = embed_texts(brand_raw[has_brand], "search_document: ", EMB_DIR / "brand.npz", logger=logger)
        emb = pd.DataFrame({
            "emb_cos_title": (emb_q * emb_t).sum(axis=1),
            "emb_cos_desc": (emb_q * emb_d).sum(axis=1),
            "emb_cos_brand": np.nan,
        }, index=df.index)
        emb.loc[has_brand, "emb_cos_brand"] = (emb_q[has_brand] * emb_b).sum(axis=1)
        del emb_d, emb_b

    with timer("preprocess", logger):
        df = preprocess(df, query_fn=corrector)
        attr_text = load_attribute_texts()
        df["attr_text"] = _map_unique(df["product_uid"].map(attr_text).fillna(""), lambda s: stem(normalize(s)))

    with timer("tfidf features", logger):
        vecs, vocab = tfidf_vectors(df, token_pattern=TOKEN_PATTERN)
        cos = cosine_features(vecs, df.index)

    with timer("match features", logger):
        match = match_features(df)

    with timer("brand features", logger):
        brand = brand_features(df, brands)

    with timer("bm25 features", logger):
        bm25 = bm25_features(df, {"title": "product_title", "desc": "product_description", "attr": "attr_text"},
                             token_pattern=TOKEN_PATTERN)

    with timer("char ngram features", logger):
        char = char_cosine_features(df)

    with timer("attr match features", logger):
        attr = attr_match_features(df)

    dense = pd.concat([cos, match, brand, bm25, char, attr, emb], axis=1)
    # 埋め込みベクトル（検索語・タイトル、各 768 次元）は密な行列だが、TF-IDF ベクトルと横に並べるため疎行列にする
    X_all = sparse.hstack(
        [sparse.csr_matrix(dense.to_numpy(dtype=np.float32)), sparse.csr_matrix(emb_q), sparse.csr_matrix(emb_t),
         vecs["query"].astype(np.float32), vecs["title"].astype(np.float32)], format="csr"
    )
    del emb_q, emb_t
    n_dim = EMB_DIM
    feature_names = (list(dense.columns) + [f"eq_{i}" for i in range(n_dim)] + [f"et_{i}" for i in range(n_dim)]
                     + [f"q_{w}" for w in vocab] + [f"t_{w}" for w in vocab])
    X, X_test = X_all[: len(train)], X_all[len(train) :]
    y = train["relevance"].to_numpy()
    logger.info(f"vocab size: {len(vocab)}, n_features: {X.shape[1]} (dense {dense.shape[1]})")

    # 追加した特徴量の値ごとの平均 relevance（train）
    dense_train = dense.iloc[: len(train)].assign(relevance=y)
    for col in []:
        stats = dense_train.groupby(col)["relevance"].agg(["mean", "count"]).round(3)
        logger.info(f"relevance by {col}:\n{stats.to_string()}")
    new_cols = list(emb.columns)
    logger.info("new feature stats (train):\n" + dense.iloc[: len(train)][new_cols].describe().T.round(3).to_string())
    corr = dense_train[new_cols + ["cos_title", "ratio_match_title"]].corrwith(dense_train["relevance"])
    logger.info("corr with relevance (train):\n" + corr.round(3).to_string())

    oof, pred_test, importance = run_cv_lgb(X, y, X_test, train["search_term"], LGB_PARAMS, logger)

    imp = pd.Series(importance, index=feature_names)
    logger.info(f"features used (gain > 0): {(imp > 0).sum()}")
    for prefix, name in [("eq_", "query emb"), ("et_", "title emb"), ("q_", "query tfidf"), ("t_", "title tfidf")]:
        part = imp[imp.index.str.startswith(prefix)]
        logger.info(f"{name}: used {(part > 0).sum()} columns, total gain {part.sum():.1f}")
    logger.info("feature importance top 40 (gain):\n" + imp.sort_values(ascending=False).head(40).round(1).to_string())

    pd.DataFrame({"id": train["id"], "relevance": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "relevance": pred_test}).to_csv(out_dir / "submission.csv", index=False)
    logger.info(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
