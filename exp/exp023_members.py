"""exp023: アンサンブルのメンバーを学習する。

exp023_build_features.py で保存した exp022 の特徴量を読み込み、メンバーごとに LightGBM の CV と同じ 5 fold
（検索語単位の GroupKFold）で学習して、OOF と test の予測を output/exp023/members/<名前>.npz に保存する。
保存済みのメンバーは学習し直さない。MLP は GPU 用の仮想環境で exp023_mlp.py を使う。
    python exp/exp023_members.py                 # すべてのメンバー
    python exp/exp023_members.py rf_a knn_a20    # 指定したメンバーだけ
"""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse

from src.finetune import fold_ids
from src.utils import OUTPUT_DIR, ROOT, get_logger, rmse

FEAT_DIR = ROOT / "data" / "features" / "exp022"
OUT_DIR = OUTPUT_DIR / "exp023"
MEMBER_DIR = OUT_DIR / "members"
SEED = 42

# exp016 で調整した LightGBM のパラメータ（exp022 と同じ）
LGB_BASE = {"objective": "regression", "metric": "rmse", "learning_rate": 0.02, "num_leaves": 15,
            "min_child_samples": 66, "feature_fraction": 0.3137, "bagging_fraction": 0.9252, "bagging_freq": 1,
            "lambda_l1": 9.04, "lambda_l2": 0.0046, "verbose": -1}

# k-NN で使う、重要度の高い特徴量（少数の列で距離を測る）
KNN_COLS_A = ["ft_ce_large_title_desc", "ft_ce_large_title", "ft_ce_title_desc", "ft_ce_title", "ce_title", "ce_desc",
              "emb_cos_title", "emb_cos_desc", "ratio_match_title", "cos_title", "q_ft_ce_large_title_desc_diff_max",
              "q_ft_ce_large_title_desc_diff_mean", "q_size"]


class Data:
    """特徴量を必要になったときに読み込む"""

    def __init__(self):
        self.meta = json.loads((FEAT_DIR / "meta.json").read_text(encoding="utf-8"))
        self.y = np.load(FEAT_DIR / "y.npy")
        self.folds = fold_ids(pd.Series(self.meta["groups"]))
        self.names = self.meta["feature_names"]
        self.n_dense, self.n_emb = self.meta["n_dense"], self.meta["n_emb"]
        self._sparse = None

    @property
    def sparse(self):
        if self._sparse is None:
            self._sparse = (sparse.load_npz(FEAT_DIR / "X_train.npz").tocsr(), sparse.load_npz(FEAT_DIR / "X_test.npz").tocsr())
        return self._sparse

    def dense(self, n_cols: int):
        """先頭 n_cols 列（数値の特徴量、または数値の特徴量 + 埋め込みベクトル）を密な行列で返す"""
        X, X_test = self.sparse
        return X[:, :n_cols].toarray(), X_test[:, :n_cols].toarray()

    def columns(self, cols: list[str]):
        idx = [self.names.index(c) for c in cols]
        X, X_test = self.sparse
        return X[:, idx].toarray(), X_test[:, idx].toarray()


def impute_scale(X, X_test):
    """欠損を train の中央値で埋め、train の平均と標準偏差で標準化する（k-NN、Ridge、ランダムフォレスト用）"""
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0, med)
    X, X_test = np.where(np.isnan(X), med, X), np.where(np.isnan(X_test), med, X_test)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    return ((X - mu) / sd).astype(np.float32), ((X_test - mu) / sd).astype(np.float32)


def run_cv(name, data, X, X_test, fit_predict, logger):
    """fold ごとに fit_predict(X_tr, y_tr, X_va, y_va, X_test) -> (va の予測, test の予測) を呼び、OOF と test の予測を保存する"""
    y, folds = data.y, data.folds
    oof, pred_test = np.zeros(len(y)), np.zeros(X_test.shape[0])
    t0 = time.time()
    for k in range(5):
        tr, va = folds != k, folds == k
        p_va, p_te = fit_predict(X[tr], y[tr], X[va], y[va], X_test)
        oof[va] = p_va
        pred_test += p_te / 5
        logger.info(f"{name}: fold {k} rmse={rmse(y[va], np.clip(p_va, 1, 3)):.5f} ({time.time() - t0:.0f}s)")
    oof, pred_test = np.clip(oof, 1, 3), np.clip(pred_test, 1, 3)
    MEMBER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MEMBER_DIR / f"{name}.npz", oof=oof, test=pred_test)
    logger.info(f"{name}: CV rmse={rmse(y, oof):.5f} ({(time.time() - t0) / 60:.1f}min)")


# ---- メンバーの定義 -------------------------------------------------------------------------------

def lgb_member(params, seed):
    import lightgbm as lgb

    def fp(X_tr, y_tr, X_va, y_va, X_te):
        m = lgb.train({**params, "seed": seed}, lgb.Dataset(X_tr, y_tr), num_boost_round=10000,
                      valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict(X_va, num_iteration=m.best_iteration), m.predict(X_te, num_iteration=m.best_iteration)
    return fp


def xgb_member(params):
    import xgboost as xgb

    def fp(X_tr, y_tr, X_va, y_va, X_te):
        m = xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=100, device="cuda", tree_method="hist",
                             random_state=SEED, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m.predict(X_va), m.predict(X_te)
    return fp


def cat_member(params):
    from catboost import CatBoostRegressor

    def fp(X_tr, y_tr, X_va, y_va, X_te):
        m = CatBoostRegressor(iterations=10000, od_type="Iter", od_wait=100, loss_function="RMSE", task_type="GPU",
                              random_seed=SEED, verbose=0, **params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        return m.predict(X_va), m.predict(X_te)
    return fp


def sk_member(make_model):
    def fp(X_tr, y_tr, X_va, y_va, X_te):
        m = make_model()
        m.fit(X_tr, y_tr)
        return m.predict(X_va), m.predict(X_te)
    return fp


def build_members():
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    rf = lambda **kw: sk_member(lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED, **kw))
    et = lambda **kw: sk_member(lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=SEED, **kw))
    knn = lambda k, w: sk_member(lambda: KNeighborsRegressor(n_neighbors=k, weights=w, algorithm="brute", n_jobs=-1))

    # (名前, 使う特徴量, 学習関数)。特徴量: "sparse" = 全列（疎行列）、"dense" = 数値の特徴量 + 埋め込みベクトル、
    # "num" = 数値の特徴量だけ（欠損補完 + 標準化）、"knn_a" = 重要度の高い 13 列（同）、"num_emb" = 数値 + 埋め込み（同）
    return [
        # 勾配ブースティング（LightGBM は exp022 そのものに加えて 2 つ）
        ("lgb_seed7_ff025", "sparse", lgb_member({**LGB_BASE, "feature_fraction": 0.25}, seed=7)),
        ("lgb_leaves63", "sparse", lgb_member({**LGB_BASE, "num_leaves": 63, "min_child_samples": 100,
                                               "feature_fraction": 0.4, "lambda_l1": 5.0}, seed=11)),
        ("xgb_d6", "sparse", xgb_member({"max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.4,
                                         "min_child_weight": 50, "reg_lambda": 1.0})),
        ("xgb_d4", "sparse", xgb_member({"max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.3,
                                         "min_child_weight": 20, "reg_alpha": 5.0})),
        ("cat_d6", "dense", cat_member({"depth": 6, "learning_rate": 0.05})),
        ("cat_d8", "dense", cat_member({"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10})),
        # 線形回帰
        ("ridge_num", "num", sk_member(lambda: Ridge(alpha=10.0))),
        ("ridge_num_emb", "num_emb", sk_member(lambda: Ridge(alpha=100.0))),
        # ランダムフォレスト・Extra Trees（数値の特徴量だけ）
        ("rf_mf03_leaf20", "num", rf(max_features=0.3, min_samples_leaf=20)),
        ("rf_mf06_leaf50", "num", rf(max_features=0.6, min_samples_leaf=50)),
        ("rf_sqrt_leaf10", "num", rf(max_features="sqrt", min_samples_leaf=10)),
        ("et_mf05_leaf20", "num", et(max_features=0.5, min_samples_leaf=20)),
        ("et_mf08_leaf50", "num", et(max_features=0.8, min_samples_leaf=50)),
        # k-NN
        ("knn_a_k20_dist", "knn_a", knn(20, "distance")),
        ("knn_a_k100_unif", "knn_a", knn(100, "uniform")),
        ("knn_num_k30_dist", "num", knn(30, "distance")),
        ("knn_num_k100_unif", "num", knn(100, "uniform")),
        ("knn_num_k300_unif", "num", knn(300, "uniform")),
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(OUT_DIR / "members_log.txt")
    data = Data()
    members = build_members()
    wanted = set(sys.argv[1:])
    cache = {}

    def features(kind):
        if kind not in cache:
            cache.clear()  # メモリ節約のため、使う特徴量の種類ごとに作り直す
            if kind == "sparse":
                cache[kind] = data.sparse
            elif kind == "dense":
                cache[kind] = data.dense(data.n_dense + data.n_emb)
            elif kind == "num":
                cache[kind] = impute_scale(*data.dense(data.n_dense))
            elif kind == "num_emb":
                cache[kind] = impute_scale(*data.dense(data.n_dense + data.n_emb))
            elif kind == "knn_a":
                cache[kind] = impute_scale(*data.columns(KNN_COLS_A))
        return cache[kind]

    # 種類ごとにまとめて学習する（特徴量の作り直しを減らす）
    for name, kind, fp in sorted(members, key=lambda m: m[1]):
        if wanted and name not in wanted:
            continue
        if (MEMBER_DIR / f"{name}.npz").exists():
            logger.info(f"{name}: cached")
            continue
        X, X_test = features(kind)
        logger.info(f"{name}: start ({kind}, {X.shape[1]} cols)")
        run_cv(name, data, X, X_test, fp, logger)


if __name__ == "__main__":
    main()
