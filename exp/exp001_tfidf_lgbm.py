"""exp001: 検索語×タイトル / 検索語×説明文の TF-IDF コサイン類似度 + LightGBM"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data import load_data
from src.features import tfidf_cosine_features
from src.utils import OUTPUT_DIR, get_logger, rmse, seed_everything, timer

EXP_NAME = "exp001"
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
        feats = tfidf_cosine_features(df)
    X = feats.iloc[: len(train)].reset_index(drop=True)
    X_test = feats.iloc[len(train) :].reset_index(drop=True)
    y = train["relevance"].to_numpy()
    logger.info(f"features: {list(X.columns)}")
    logger.info("feature stats (train):\n" + X.describe().round(4).to_string())

    # test の行の 73% は train にない検索語なので、検索語単位でグループ分割して条件を合わせる
    oof = np.zeros(len(train))
    pred_test = np.zeros(len(test))
    importance = pd.DataFrame(index=X.columns)
    folds = GroupKFold(n_splits=N_FOLDS).split(X, y, groups=train["search_term"])
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X.iloc[tr_idx], y[tr_idx]),
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[lgb.Dataset(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        pred_test += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        importance[f"fold{fold}"] = model.feature_importance("gain")
        logger.info(
            f"fold {fold}: rmse={rmse(y[va_idx], oof[va_idx]):.5f} best_iter={model.best_iteration}"
        )

    oof = np.clip(oof, 1, 3)
    pred_test = np.clip(pred_test, 1, 3)
    logger.info(f"CV rmse: {rmse(y, oof):.5f}")
    logger.info(f"(reference) predict mean only: {rmse(y, np.full_like(y, y.mean())):.5f}")
    logger.info("feature importance (gain):\n" + importance.mean(axis=1).round(1).to_string())

    pd.DataFrame({"id": train["id"], "relevance": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "relevance": pred_test}).to_csv(out_dir / "submission.csv", index=False)
    logger.info(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
