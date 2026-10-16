"""exp006: exp004 + 単語一致の数え上げ特徴量 + ブランド一致の特徴量"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import load_brands, load_data
from src.features import brand_features, cosine_features, match_features, tfidf_vectors
from src.preprocess import _map_unique, normalize, preprocess, stem
from src.spell import build_symspell, make_query_corrector
from src.train import run_cv_lgb
from src.utils import OUTPUT_DIR, get_logger, seed_everything, timer

EXP_NAME = "exp006"
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

    with timer("tfidf features", logger):
        vecs, vocab = tfidf_vectors(df, token_pattern=TOKEN_PATTERN)
        cos = cosine_features(vecs, df.index)

    with timer("match features", logger):
        match = match_features(df)

    with timer("brand features", logger):
        brand = brand_features(df, brands)

    dense = pd.concat([cos, match, brand], axis=1)
    X_all = sparse.hstack(
        [sparse.csr_matrix(dense.to_numpy(dtype=np.float64)), vecs["query"], vecs["title"]], format="csr"
    )
    feature_names = list(dense.columns) + [f"q_{w}" for w in vocab] + [f"t_{w}" for w in vocab]
    X, X_test = X_all[: len(train)], X_all[len(train) :]
    y = train["relevance"].to_numpy()
    logger.info(f"vocab size: {len(vocab)}, n_features: {X.shape[1]} (dense {dense.shape[1]})")

    # 追加した特徴量の値ごとの平均 relevance（train）
    dense_train = dense.iloc[: len(train)].assign(relevance=y)
    for col in ["last_in_title", "q_in_title", "has_brand", "brand_in_query", "query_mentions_brand", "other_brand_in_query"]:
        stats = dense_train.groupby(col)["relevance"].agg(["mean", "count"]).round(3)
        logger.info(f"relevance by {col}:\n{stats.to_string()}")
    logger.info("dense feature stats (train):\n" + dense.iloc[: len(train)].describe().T.round(3).to_string())

    oof, pred_test, importance = run_cv_lgb(X, y, X_test, train["search_term"], LGB_PARAMS, logger)

    imp = pd.Series(importance, index=feature_names)
    logger.info(f"features used (gain > 0): {(imp > 0).sum()}")
    logger.info("feature importance top 40 (gain):\n" + imp.sort_values(ascending=False).head(40).round(1).to_string())

    pd.DataFrame({"id": train["id"], "relevance": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "relevance": pred_test}).to_csv(out_dir / "submission.csv", index=False)
    logger.info(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
