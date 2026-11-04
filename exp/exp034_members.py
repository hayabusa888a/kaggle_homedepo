"""exp034: スタッキングのメンバーを、exp033 の特徴量で学習する。

exp023・exp024 のメンバーに、損失関数や解き方の違うモデルを加える。test の特徴量は exp024 と同じく fold ごとにそろえる。
exp023・exp024 との違い: early stopping に CV の検証 fold を使わない。学習側の fold から検索語の 1 割を取り分けて
early stopping に使い、検証 fold は予測だけに使う（OOF の予測が楽観的になり、混ぜ方が OOF に合わせすぎるのを防ぐ）。
結果は output/exp034/members/<名前>.npz（oof, test）。保存済みのメンバーは学習し直さない。
    python exp/exp034_members.py                         # MLP 以外のすべて
    .venv-gpu/Scripts/python.exe exp/exp034_members.py --mlp
    python exp/exp034_members.py --exp=exp035 lgb_base ...  # exp035 の特徴量で、指定したメンバーだけ（output/exp035/members）
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from scipy import sparse

import exp023_members as m23
from src.utils import OUTPUT_DIR, ROOT, get_logger, rmse

# 実験名（--exp=...）ごとの特徴量のフォルダ。exp035 は exp033 に product_uid を加えたもの
EXP = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--exp=")), "exp034")
m23.FEAT_DIR = ROOT / "data" / "features" / {"exp034": "exp033", "exp035": "exp035", "exp036": "exp036"}[EXP]
m23.KNN_COLS_A = ["ft_ce_large_title_desc", "ft_ce_large_title", "ft_ce_title_desc", "ft_ce_title",
                  "ft_ce_deberta_title_desc", "ft_ce_deberta_title", "ft_ce_modernbert_title_desc", "ft_ce_qwen3_title",
                  "ce_title", "ce_desc", "emb_cos_title", "emb_cos_desc", "ratio_match_title", "cos_title",
                  "q_ft_ce_large_title_desc_diff_max", "q_ft_ce_large_title_desc_diff_mean", "q_size"]
import exp024_members as m24  # noqa: E402

OUT_DIR = OUTPUT_DIR / EXP
MEMBER_DIR = OUT_DIR / "members"
FT_COLS = ["ft_ce_title", "ft_ce_title_desc", "ft_ce_large_title", "ft_ce_large_title_desc",
           "ft_ce_deberta_title_desc", "ft_ce_deberta_title", "ft_ce_modernbert_title_desc", "ft_ce_qwen3_title"]
# relevance を 1/3 刻みの 7 段階に丸めたときの値（分類で解くメンバー用）
CLASS_VALUES = np.array([1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3])


def inner_split(groups: np.ndarray, seed: int):
    """学習側の行を、検索語単位で 9 : 1 に分ける（1 の側を early stopping に使う）"""
    uniq = np.unique(groups)
    es_groups = np.random.default_rng(seed).choice(uniq, size=len(uniq) // 10, replace=False)
    es = np.isin(groups, es_groups)
    return ~es, es


def stack_rows(A, B):
    return sparse.vstack([A, B], format="csr") if sparse.issparse(A) else np.vstack([A, B])


def run_cv(name, data, X, make_test, fit_predict, use_es, logger):
    """fold ごとに学習し、OOF と test の予測を保存する。use_es のメンバーは fit_predict(X_fit, y_fit, X_es, y_es, X_pred) で
    early stopping し、そうでないメンバーは学習側の fold 全部で学習する（X_es, y_es は使わない）"""
    y, folds = data.y, data.folds
    groups = np.asarray(data.meta["groups"])
    oof, pred_test = np.zeros(len(y)), None
    t0 = time.time()
    for k in range(5):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        X_te = make_test(k)
        X_pred = stack_rows(X[va], X_te)
        if use_es:
            fit, es = inner_split(groups[tr], seed=k)
            p = fit_predict(X[tr[fit]], y[tr[fit]], X[tr[es]], y[tr[es]], X_pred)
        else:
            p = fit_predict(X[tr], y[tr], None, None, X_pred)
        p_va, p_te = p[: len(va)], p[len(va):]
        oof[va] = p_va
        pred_test = p_te / 5 if pred_test is None else pred_test + p_te / 5
        logger.info(f"{name}: fold {k} rmse={rmse(y[va], np.clip(p_va, 1, 3)):.5f} ({time.time() - t0:.0f}s)")
    oof, pred_test = np.clip(oof, 1, 3), np.clip(pred_test, 1, 3)
    MEMBER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MEMBER_DIR / f"{name}.npz", oof=oof, test=pred_test)
    logger.info(f"{name}: CV rmse={rmse(y, oof):.5f} ({(time.time() - t0) / 60:.1f}min)")


# ---- メンバーの定義（fit_predict(X_fit, y_fit, X_es, y_es, X_pred) -> X_pred の予測） --------------------

def lgb_member(params, seed):
    import lightgbm as lgb

    def fp(X_fit, y_fit, X_es, y_es, X_pred):
        m = lgb.train({**m23.LGB_BASE, **params, "seed": seed}, lgb.Dataset(X_fit, y_fit), num_boost_round=10000,
                      valid_sets=[lgb.Dataset(X_es, y_es)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict(X_pred, num_iteration=m.best_iteration)
    return fp


def lgb_class_member(seed):
    """relevance を 7 段階に丸めて多クラス分類で解き、各段階の確率 × 値の和（期待値）を予測にする"""
    import lightgbm as lgb

    def fp(X_fit, y_fit, X_es, y_es, X_pred):
        to_cls = lambda v: np.abs(v[:, None] - CLASS_VALUES[None, :]).argmin(axis=1)
        params = {**m23.LGB_BASE, "objective": "multiclass", "num_class": len(CLASS_VALUES), "metric": "multi_logloss",
                  "learning_rate": 0.05, "seed": seed}
        m = lgb.train(params, lgb.Dataset(X_fit, to_cls(y_fit)), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X_es, to_cls(y_es))], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict(X_pred, num_iteration=m.best_iteration) @ CLASS_VALUES
    return fp


def xgb_member(params):
    import xgboost as xgb

    def fp(X_fit, y_fit, X_es, y_es, X_pred):
        m = xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=100, device="cuda", tree_method="hist",
                             random_state=m23.SEED, **params)
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        return m.predict(X_pred)
    return fp


def cat_member(params):
    from catboost import CatBoostRegressor

    def fp(X_fit, y_fit, X_es, y_es, X_pred):
        m = CatBoostRegressor(iterations=10000, od_type="Iter", od_wait=100, loss_function="RMSE", task_type="GPU",
                              random_seed=m23.SEED, verbose=0, train_dir=str(OUT_DIR / "catboost_info"), **params)
        m.fit(X_fit, y_fit, eval_set=(X_es, y_es), use_best_model=True)
        return m.predict(X_pred)
    return fp


def sk_member(make_model):
    def fp(X_fit, y_fit, X_es, y_es, X_pred):
        m = make_model()
        m.fit(X_fit, y_fit)
        return m.predict(X_pred)
    return fp


def ft_mean_member(data):
    """ファインチューニングした 8 本のスコアの単純平均（学習しない。1〜3 に換算）"""
    idx = [data.names.index(c) for c in FT_COLS]

    def fp(X_fit, y_fit, X_es, y_es, X_pred):
        return X_pred[:, idx].mean(axis=1) * 2 + 1
    return fp


def build_members(data):
    """(名前, 使う特徴量, 学習関数, early stopping を使うか)。特徴量の種類は exp024_members.FoldTest.prepare と同じ
    （"sparse" = 全列、"dense" = 数値 + 埋め込み（補完・標準化なし）、"num" / "num_emb" / "knn_a" = 補完 + 標準化）"""
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline

    seed = m23.SEED
    rf = lambda **kw: sk_member(lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed, **kw))
    et = lambda **kw: sk_member(lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=seed, **kw))
    knn = lambda k, w: sk_member(lambda: KNeighborsRegressor(n_neighbors=k, weights=w, algorithm="brute", n_jobs=-1))
    return [
        # 勾配ブースティング（全列）
        ("lgb_base", "sparse", lgb_member({}, seed=42), True),
        ("lgb_seed7_ff025", "sparse", lgb_member({"feature_fraction": 0.25}, seed=7), True),
        ("lgb_leaves63", "sparse", lgb_member({"num_leaves": 63, "min_child_samples": 100, "feature_fraction": 0.4,
                                               "lambda_l1": 5.0}, seed=11), True),
        ("lgb_huber", "sparse", lgb_member({"objective": "huber", "alpha": 0.5}, seed=13), True),
        ("lgb_l1", "sparse", lgb_member({"objective": "regression_l1"}, seed=17), True),
        ("xgb_d6", "sparse", xgb_member({"max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.4,
                                         "min_child_weight": 50, "reg_lambda": 1.0}), True),
        ("xgb_d4", "sparse", xgb_member({"max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.3,
                                         "min_child_weight": 20, "reg_alpha": 5.0}), True),
        # 勾配ブースティング（数値 + 埋め込み）
        ("lgb_num_emb", "dense", lgb_member({}, seed=23), True),
        ("lgb_class", "dense", lgb_class_member(seed=29), True),
        ("cat_d6", "dense", cat_member({"depth": 6, "learning_rate": 0.05}), True),
        ("cat_d8", "dense", cat_member({"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10}), True),
        # 勾配ブースティング（数値だけ）
        ("hgb_num", "num", sk_member(lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, max_iter=400,
            random_state=seed)), False),
        # 線形・カーネル
        ("ft_mean", "num_raw", None, False),
        ("ridge_num", "num", sk_member(lambda: Ridge(alpha=10.0)), False),
        ("ridge_num_emb", "num_emb", sk_member(lambda: Ridge(alpha=100.0)), False),
        ("kridge_num", "num", sk_member(lambda: make_pipeline(
            Nystroem(kernel="rbf", gamma=1 / 200, n_components=1500, random_state=seed), Ridge(alpha=3.0))), False),
        # ランダムフォレスト・Extra Trees（数値だけ）
        ("rf_mf03_leaf20", "num", rf(max_features=0.3, min_samples_leaf=20), False),
        ("rf_mf06_leaf50", "num", rf(max_features=0.6, min_samples_leaf=50), False),
        ("rf_sqrt_leaf10", "num", rf(max_features="sqrt", min_samples_leaf=10), False),
        ("et_mf05_leaf20", "num", et(max_features=0.5, min_samples_leaf=20), False),
        ("et_mf08_leaf50", "num", et(max_features=0.8, min_samples_leaf=50), False),
        # k-NN
        ("knn_a_k20_dist", "knn_a", knn(20, "distance"), False),
        ("knn_a_k100_unif", "knn_a", knn(100, "uniform"), False),
        ("knn_num_k30_dist", "num", knn(30, "distance"), False),
        ("knn_num_k100_unif", "num", knn(100, "uniform"), False),
        ("knn_num_k300_unif", "num", knn(300, "uniform"), False),
    ]


# (名前, 使う特徴量, 隠れ層のユニット数, ドロップアウト, 乱数の種)
MLP_MEMBERS = [
    ("mlp_num_emb_512", "num_emb", [512, 128], 0.2, 42),
    ("mlp_num_emb_1024", "num_emb", [1024, 256, 64], 0.3, 7),
    ("mlp_num_256", "num", [256, 64], 0.1, 42),
    ("mlp_num_512", "num", [512, 128], 0.2, 19),
]


def mlp_members():
    import exp023_mlp as mlp
    members = []
    for name, kind, hidden, dropout, seed in MLP_MEMBERS:
        def fp(X_fit, y_fit, X_es, y_es, X_pred, h=hidden, dr=dropout, s=seed):
            # exp023_mlp.fit_predict は (検証側の予測, test の予測) を返す。検証側に early stopping 用の行を渡す
            return mlp.fit_predict(X_fit, y_fit, X_es, y_es, X_pred, h, dr, s)[1]
        members.append((name, kind, fp, True))
    return members


class FoldTest(m24.FoldTest):
    """exp024 の FoldTest に、補完・標準化をしない数値の特徴量（ft_mean 用）を加える"""

    def prepare(self, kind):
        if kind == "num_raw":
            return self.data.dense(self.data.n_dense)[0], lambda k: self.dense_k[k]
        return super().prepare(kind)


def main():
    use_mlp = "--mlp" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(OUT_DIR / ("mlp_log.txt" if use_mlp else "members_log.txt"))
    data = m23.Data()
    ft = FoldTest(data)
    members = mlp_members() if use_mlp else build_members(data)
    members = [(n, k, ft_mean_member(data) if fp is None else fp, es) for n, k, fp, es in members]
    wanted = {a for a in sys.argv[1:] if not a.startswith("--")}
    cache = {}
    for name, kind, fp, use_es in sorted(members, key=lambda m: m[1]):
        if wanted and name not in wanted:
            continue
        if (MEMBER_DIR / f"{name}.npz").exists():
            logger.info(f"{name}: cached")
            continue
        if kind not in cache:
            cache.clear()
            cache[kind] = ft.prepare(kind)
        X, make_test = cache[kind]
        logger.info(f"{name}: start ({kind}, {X.shape[1]} cols, early stopping {'inner 10%' if use_es else 'none'})")
        run_cv(name, data, X, make_test, fp, use_es, logger)


if __name__ == "__main__":
    main()
