"""exp034: exp034_members.py で学習したメンバーの予測を、2 段目のモデルで混ぜる（スタッキング）。

2 段目のモデル × メンバーの組み合わせで、6 通りを作る。
- 2 段目のモデル
  - linear: 重みを 0 以上・合計 1 とした重み付き平均（exp023 と同じ混ぜ方）
  - lgb: 浅い LightGBM。メンバーの予測に加えて、少数のメタ特徴量（8 本のスコアのばらつき、検索語の商品数、
    商品の新しさなど）を入力にし、行の性質に応じてメンバーの信頼度を変えられるようにする
  - lgb_nouid: lgb から商品の新しさ（product_uid）を除いたもの
- メンバー
  - trees: 全列または数値 + 埋め込みを使う勾配ブースティングだけ
  - all: すべて
評価は入れ子の CV（外側の fold k の 2 段目のモデルは、ほかの 4 fold の OOF で学習して fold k を予測する）。
test は全部の OOF で学習した 2 段目のモデルで予測する。メタ特徴量は fold ごとの test の特徴量で作り、5 つの予測を平均する。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import exp034_members as m34
from exp023_blend import fit_weights
from src.utils import get_logger, rmse

m23 = m34.m23
TREES = ["lgb_base", "lgb_seed7_ff025", "lgb_leaves63", "lgb_huber", "lgb_l1", "xgb_d6", "xgb_d4",
         "lgb_num_emb", "lgb_class", "cat_d6", "cat_d8"]
META_LGB = {"objective": "regression", "metric": "rmse", "learning_rate": 0.02, "num_leaves": 7,
            "min_child_samples": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 10.0, "verbose": -1, "seed": 42}


def meta_features(dense: np.ndarray, names: list[str], uid: np.ndarray) -> pd.DataFrame:
    ft = dense[:, [names.index(c) for c in m34.FT_COLS]] * 2 + 1
    col = lambda c: dense[:, names.index(c)]
    return pd.DataFrame({"meta_ft_std": ft.std(axis=1), "meta_ft_range": ft.max(axis=1) - ft.min(axis=1),
                         "meta_q_size": col("q_size"), "meta_uid": uid, "meta_ratio_match_title": col("ratio_match_title"),
                         "meta_q_n_words": col("q_n_words")})


def fit_lgb(X, y, groups, seed=0):
    """検索語の 1 割で early stopping した浅い LightGBM を返す"""
    import lightgbm as lgb
    fit, es = m34.inner_split(groups, seed)
    return lgb.train(META_LGB, lgb.Dataset(X[fit], y[fit]), num_boost_round=5000,
                     valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    logger = get_logger(m34.OUT_DIR / "stack_log.txt")
    data = m23.Data()
    y, folds, names, n = data.y, data.folds, data.names, data.n_dense
    groups = np.asarray(data.meta["groups"])
    oof, test = {}, {}
    for path in sorted(m34.MEMBER_DIR.glob("*.npz")):
        d = np.load(path)
        oof[path.stem], test[path.stem] = d["oof"], d["test"]
    oof, test = pd.DataFrame(oof), pd.DataFrame(test)
    logger.info(f"members: {oof.shape[1]}")
    logger.info("single CV rmse / corr with lgb_base:\n" + pd.DataFrame({
        "rmse": {c: rmse(y, oof[c]) for c in oof}, "corr_lgb_base": oof.corr()["lgb_base"]}).sort_values("rmse").round(5).to_string())

    train_csv = pd.read_csv(m23.ROOT / "input" / "train.csv", encoding="ISO-8859-1")
    test_csv = pd.read_csv(m23.ROOT / "input" / "test.csv", encoding="ISO-8859-1")
    meta_tr = meta_features(data.dense(n)[0], names, train_csv["product_uid"].to_numpy())
    meta_te = [meta_features(np.load(m23.FEAT_DIR / f"X_test_dense_fold{k}.npy"), names, test_csv["product_uid"].to_numpy())
               for k in range(5)]

    rows, preds = [], {}
    for set_name, cols in [("trees", [c for c in TREES if c in oof]), ("all", list(oof.columns))]:
        P, P_test = oof[cols].to_numpy(), test[cols].to_numpy()
        # linear
        nested = np.zeros(len(y))
        for k in range(5):
            tr, va = folds != k, folds == k
            nested[va] = P[va] @ fit_weights(P[tr], y[tr])
        w = pd.Series(fit_weights(P, y), index=cols)
        preds[f"linear_{set_name}"] = np.clip(P_test @ w.to_numpy(), 1, 3)
        rows.append({"method": f"linear_{set_name}", "n_members": len(cols), "nested_cv": rmse(y, np.clip(nested, 1, 3))})
        logger.info(f"linear_{set_name}: nested CV {rows[-1]['nested_cv']:.5f}, weights:\n"
                    + w[w > 1e-4].sort_values(ascending=False).round(4).to_string())
        # lgb（メンバーの予測 + メタ特徴量）。商品の新しさ（meta_uid）の効果を切り分けるため、使わない版も作る
        for method, meta_cols in [("lgb", list(meta_tr.columns)), ("lgb_nouid", [c for c in meta_tr.columns if c != "meta_uid"])]:
            name = f"{method}_{set_name}"
            Xm = np.hstack([P, meta_tr[meta_cols].to_numpy()])
            nested = np.zeros(len(y))
            for k in range(5):
                tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
                m = fit_lgb(Xm[tr], y[tr], groups[tr], seed=k)
                nested[va] = m.predict(Xm[va], num_iteration=m.best_iteration)
            m = fit_lgb(Xm, y, groups, seed=99)
            pred = np.mean([m.predict(np.hstack([P_test, meta_te[k][meta_cols].to_numpy()]), num_iteration=m.best_iteration)
                            for k in range(5)], axis=0)
            preds[name] = np.clip(pred, 1, 3)
            rows.append({"method": name, "n_members": len(cols), "nested_cv": rmse(y, np.clip(nested, 1, 3))})
            gain = pd.Series(m.feature_importance("gain"), index=cols + meta_cols)
            logger.info(f"{name}: nested CV {rows[-1]['nested_cv']:.5f}, best_iteration {m.best_iteration}, gain share:\n"
                        + (gain / gain.sum()).sort_values(ascending=False).head(15).round(4).to_string())

    rows.append({"method": "lgb_base (single)", "n_members": 1, "nested_cv": rmse(y, oof["lgb_base"])})
    summary = pd.DataFrame(rows)
    test_ids = data.meta["test_id"]
    for name, p in preds.items():
        pd.DataFrame({"id": test_ids, "relevance": p}).to_csv(m34.OUT_DIR / f"submission_{name}.csv", index=False)
    pd.DataFrame({"id": test_ids, "relevance": test["lgb_base"]}).to_csv(m34.OUT_DIR / "submission_lgb_base.csv", index=False)
    sub33 = pd.read_csv(m34.OUTPUT_DIR / "exp033" / "submission_all.csv").set_index("id").loc[test_ids, "relevance"].to_numpy()
    summary["test_mean"] = [preds[r].mean() if r in preds else test["lgb_base"].mean() for r in summary["method"]]
    summary["rmse_vs_exp033_test"] = [rmse(sub33, preds[r] if r in preds else test["lgb_base"]) for r in summary["method"]]
    logger.info("summary:\n" + summary.round(5).to_string(index=False))
    # 商品の新しさの帯ごとの test の予測の平均（2 段目のモデルが、新しい商品の予測をどう補正したか）
    band = pd.cut(test_csv["product_uid"], [100000, 150000, 180000, 195000, 206650, 300000])
    by_band = pd.DataFrame({"lgb_base": test["lgb_base"].to_numpy(), **preds}).groupby(band.to_numpy(), observed=True).mean()
    logger.info("test pred mean by product_uid:\n" + by_band.round(4).to_string())


if __name__ == "__main__":
    main()
