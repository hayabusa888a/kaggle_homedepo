import logging

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from src.utils import rmse, timer


def run_cv_lgb(X, y, X_test, groups, params: dict, logger: logging.Logger, n_folds: int = 5,
               num_boost_round: int = 10000, early_stopping_rounds: int = 100):
    """GroupKFold で LightGBM を学習し、(oof, test 予測, 平均 gain 重要度) を返す。予測は 1〜3 にクリップする"""
    oof = np.zeros(X.shape[0])
    pred_test = np.zeros(X_test.shape[0])
    importance = np.zeros(X.shape[1])
    for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups=groups)):
        with timer(f"fold {fold}", logger):
            model = lgb.train(
                params,
                lgb.Dataset(X[tr_idx], y[tr_idx]),
                num_boost_round=num_boost_round,
                valid_sets=[lgb.Dataset(X[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            oof[va_idx] = model.predict(X[va_idx], num_iteration=model.best_iteration)
            pred_test += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
            importance += model.feature_importance("gain") / n_folds
        logger.info(f"fold {fold}: rmse={rmse(y[va_idx], oof[va_idx]):.5f} best_iter={model.best_iteration}")

    oof, pred_test = np.clip(oof, 1, 3), np.clip(pred_test, 1, 3)
    logger.info(f"CV rmse: {rmse(y, oof):.5f}")
    logger.info(f"oof pred std: {oof.std():.4f}, test pred std: {pred_test.std():.4f}")
    return oof, pred_test, importance
