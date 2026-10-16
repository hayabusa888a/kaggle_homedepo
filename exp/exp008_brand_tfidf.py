"""exp008: exp007 + ブランド名の TF-IDF（検索語との類似度 + ベクトル）"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import load_attribute_texts, load_brands, load_data
from src.features import (attr_match_features, bm25_features, brand_features, char_cosine_features,
                          cosine_features, match_features, rowwise_cosine, tfidf_vectors)
from src.preprocess import _map_unique, normalize, preprocess, stem
from src.spell import build_symspell, make_query_corrector
from src.train import run_cv_lgb
from src.utils import OUTPUT_DIR, get_logger, seed_everything, timer

EXP_NAME = "exp008"
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


def main():
    out_dir = OUTPUT_DIR / EXP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(out_dir / "log.txt")
    seed_everything(SEED)

    with timer("load data", logger):
        train, test = load_data()
        df = pd.concat([train, test], ignore_index=True)
        # ハイフンは正規化の「連結形も残す」処理を通さず、単純にスペースにする ("Strong-Tie" -> "strong tie")
        brands = load_brands().str.replace("-", " ").map(lambda b: stem(normalize(b)))

    with timer("spell dictionary", logger):
        products = df.drop_duplicates("product_uid")
        product_texts = _map_unique(products["product_title"] + " " + products["product_description"], normalize)
        corrector = make_query_corrector(build_symspell(product_texts))

    with timer("preprocess", logger):
        df = preprocess(df, query_fn=corrector)
        attr_text = load_attribute_texts()
        df["attr_text"] = _map_unique(df["product_uid"].map(attr_text).fillna(""), lambda s: stem(normalize(s)))
        # ブランドがない商品は空文字（TF-IDF ベクトルは 0 になる）
        df["brand_text"] = df["product_uid"].map(brands).fillna("")

    with timer("tfidf features", logger):
        vecs, vocab = tfidf_vectors(df, extra_cols={"brand": "brand_text"}, token_pattern=TOKEN_PATTERN)
        cos = cosine_features(vecs, df.index)
        cos["cos_brand"] = rowwise_cosine(vecs["query"], vecs["brand"])

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

    dense = pd.concat([cos, match, brand, bm25, char, attr], axis=1)
    X_all = sparse.hstack(
        [sparse.csr_matrix(dense.to_numpy(dtype=np.float64)), vecs["query"], vecs["title"], vecs["brand"]], format="csr"
    )
    feature_names = list(dense.columns) + [f"q_{w}" for w in vocab] + [f"t_{w}" for w in vocab] + [f"b_{w}" for w in vocab]
    X, X_test = X_all[: len(train)], X_all[len(train) :]
    y = train["relevance"].to_numpy()
    logger.info(f"vocab size: {len(vocab)}, n_features: {X.shape[1]} (dense {dense.shape[1]})")

    # 追加した特徴量の値ごとの平均 relevance（train）
    dense_train = dense.iloc[: len(train)].assign(relevance=y)
    for col in ["has_attr"]:
        stats = dense_train.groupby(col)["relevance"].agg(["mean", "count"]).round(3)
        logger.info(f"relevance by {col}:\n{stats.to_string()}")
    new_cols = ["cos_brand", "brand_in_query"]
    logger.info("new feature stats (train):\n" + dense.iloc[: len(train)][new_cols].describe().T.round(3).to_string())
    corr = dense_train[new_cols + ["cos_title", "ratio_match_title"]].corrwith(dense_train["relevance"])
    logger.info("corr with relevance (train):\n" + corr.round(3).to_string())

    oof, pred_test, importance = run_cv_lgb(X, y, X_test, train["search_term"], LGB_PARAMS, logger)

    imp = pd.Series(importance, index=feature_names)
    logger.info(f"features used (gain > 0): {(imp > 0).sum()}")
    b_imp = imp[imp.index.str.startswith("b_")]
    logger.info(f"brand vector: used {(b_imp > 0).sum()} columns, total gain {b_imp.sum():.1f}, "
                f"top {b_imp.sort_values(ascending=False).head(10).round(1).to_dict()}")
    logger.info("feature importance top 40 (gain):\n" + imp.sort_values(ascending=False).head(40).round(1).to_string())

    pd.DataFrame({"id": train["id"], "relevance": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "relevance": pred_test}).to_csv(out_dir / "submission.csv", index=False)
    logger.info(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
