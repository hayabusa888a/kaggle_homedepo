"""exp003: exp002 + テキストの前処理（表記の正規化 + Snowball ステミング）"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy import sparse

from src.data import load_data
from src.features import cosine_features, tfidf_vectors
from src.preprocess import preprocess
from src.train import run_cv_lgb
from src.utils import OUTPUT_DIR, get_logger, seed_everything, timer

EXP_NAME = "exp003"
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

    with timer("preprocess", logger):
        df = preprocess(df)
    for i in [0, 1, 9]:
        logger.info(f"example: {df.loc[i, 'search_term']!r} | {df.loc[i, 'product_title']!r}")

    with timer("tfidf features", logger):
        vecs, vocab = tfidf_vectors(df, token_pattern=TOKEN_PATTERN)
        cos = cosine_features(vecs, df.index)
        X_all = sparse.hstack(
            [sparse.csr_matrix(cos.to_numpy()), vecs["query"], vecs["title"]], format="csr"
        )
    feature_names = list(cos.columns) + [f"q_{w}" for w in vocab] + [f"t_{w}" for w in vocab]
    X, X_test = X_all[: len(train)], X_all[len(train) :]
    y = train["relevance"].to_numpy()
    logger.info(f"vocab size: {len(vocab)}, n_features: {X.shape[1]}")
    logger.info(f"cos_title == 0 (train): {(cos['cos_title'][: len(train)] == 0).mean():.3%}")
    logger.info("feature stats (train):\n" + cos.iloc[: len(train)].describe().round(4).to_string())

    oof, pred_test, importance = run_cv_lgb(X, y, X_test, train["search_term"], LGB_PARAMS, logger)

    imp = pd.Series(importance, index=feature_names)
    logger.info(f"features used (gain > 0): {(imp > 0).sum()}")
    logger.info("feature importance top 30 (gain):\n" + imp.sort_values(ascending=False).head(30).round(1).to_string())

    pd.DataFrame({"id": train["id"], "relevance": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "relevance": pred_test}).to_csv(out_dir / "submission.csv", index=False)
    logger.info(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
