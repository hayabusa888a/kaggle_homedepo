"""exp002: exp001 の TF-IDF コサイン類似度 + 検索語・タイトルの TF-IDF ベクトル（疎行列のまま）+ LightGBM"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold

from src.data import load_data
from src.features import cosine_features, tfidf_vectors
from src.utils import OUTPUT_DIR, get_logger, rmse, seed_everything, timer

EXP_NAME = "exp002"
SEED = 42
N_FOLDS = 5
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "seed": SEED,
    "verbose": -1,
}
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 100


def main():
    out_dir = OUTPUT_DIR / EXP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(out_dir / "log.txt")
    seed_everything(SEED)

    with timer("load data", logger):
        train, test = load_data()
    logger.info(f"train: {train.shape}, test: {test.shape}")

    with timer("tfidf features", logger):
        df = pd.concat([train, test], ignore_index=True)
        vecs, vocab = tfidf_vectors(df)
        cos = cosine_features(vecs, df.index)
        # 列の並び: [cos_title, cos_desc, 検索語ベクトル(語彙数), タイトルベクトル(語彙数)]
        X_all = sparse.hstack(
            [sparse.csr_matrix(cos.to_numpy()), vecs["query"], vecs["title"]], format="csr"
        )
    # 単語には LightGBM が扱えない文字が含まれうるので、モデルには番号の列名を渡し、ログ表示時に単語へ戻す
    feature_names = list(cos.columns) + [f"q_{w}" for w in vocab] + [f"t_{w}" for w in vocab]
    X, X_test = X_all[: len(train)], X_all[len(train) :]
    y = train["relevance"].to_numpy()
    logger.info(f"vocab size: {len(vocab)}, n_features: {X.shape[1]}")
    logger.info(f"nonzero per row (train): query={vecs['query'][: len(train)].getnnz(axis=1).mean():.2f}, "
                f"title={vecs['title'][: len(train)].getnnz(axis=1).mean():.2f}")

    oof = np.zeros(len(train))
    pred_test = np.zeros(len(test))
    importance = np.zeros(X.shape[1])
    folds = GroupKFold(n_splits=N_FOLDS).split(X, y, groups=train["search_term"])
    for fold, (tr_idx, va_idx) in enumerate(folds):
        with timer(f"fold {fold}", logger):
            model = lgb.train(
                LGB_PARAMS,
                lgb.Dataset(X[tr_idx], y[tr_idx]),
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[lgb.Dataset(X[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof[va_idx] = model.predict(X[va_idx], num_iteration=model.best_iteration)
            pred_test += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
            importance += model.feature_importance("gain") / N_FOLDS
        logger.info(
            f"fold {fold}: rmse={rmse(y[va_idx], oof[va_idx]):.5f} best_iter={model.best_iteration}"
        )

    oof = np.clip(oof, 1, 3)
    pred_test = np.clip(pred_test, 1, 3)
    logger.info(f"CV rmse: {rmse(y, oof):.5f}")
    logger.info(f"oof pred std: {oof.std():.4f}, test pred std: {pred_test.std():.4f}")

    imp = pd.Series(importance, index=feature_names)
    logger.info(f"features used (gain > 0): {(imp > 0).sum()}")
    logger.info("feature importance top 30 (gain):\n" + imp.sort_values(ascending=False).head(30).round(1).to_string())

    pd.DataFrame({"id": train["id"], "relevance": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "relevance": pred_test}).to_csv(out_dir / "submission.csv", index=False)
    logger.info(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
