"""exp023: アンサンブルのメンバーの予測を混ぜる。

- 重みを「0 以上、合計 1」として、OOF の RMSE が最小になるように決める
- 重みを決めた OOF でそのまま評価すると楽観的になるので、4 つの fold で重みを決めて残り 1 つの fold で評価する
  操作を 5 回繰り返す（入れ子の CV）。提出用の重みは OOF 全体で決める
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from exp023_members import MEMBER_DIR, OUT_DIR, Data
from src.utils import OUTPUT_DIR, get_logger, rmse


def load_members(data: Data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """メンバーごとの OOF と test の予測を、列 = メンバーの DataFrame で返す"""
    oof, test = {}, {}
    for path in sorted(MEMBER_DIR.glob("*.npz")):
        d = np.load(path)
        oof[path.stem], test[path.stem] = d["oof"], d["test"]

    # exp022 の LightGBM（そのままメンバーにする）
    tr_id, te_id = data.meta["train_id"], data.meta["test_id"]
    o = pd.read_csv(OUTPUT_DIR / "exp022" / "oof.csv").set_index("id").loc[tr_id, "pred"].to_numpy()
    s = pd.read_csv(OUTPUT_DIR / "exp022" / "submission.csv").set_index("id").loc[te_id, "relevance"].to_numpy()
    oof["lgb_exp022"], test["lgb_exp022"] = o, s

    # ファインチューニングした cross-encoder 単独の予測（0〜1 のスコアを 1〜3 に戻す）
    for col in ["ft_ce_large_title_desc", "ft_ce_large_title"]:
        X, X_test = data.columns([col])
        oof[f"ce_{col}"], test[f"ce_{col}"] = X[:, 0] * 2 + 1, X_test[:, 0] * 2 + 1
    return pd.DataFrame(oof), pd.DataFrame(test)


def fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """重みを 0 以上・合計 1 として、sum(w * P) の RMSE を最小にする"""
    n = P.shape[1]
    res = minimize(lambda w: np.sqrt(np.mean((P @ w - y) ** 2)), np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 1000, "ftol": 1e-10})
    return res.x


def main():
    logger = get_logger(OUT_DIR / "blend_log.txt")
    data = Data()
    y, folds = data.y, data.folds
    oof, test = load_members(data)
    logger.info(f"members: {oof.shape[1]}")

    single = pd.Series({c: rmse(y, oof[c]) for c in oof.columns}).sort_values()
    logger.info("single CV rmse:\n" + single.round(5).to_string())
    corr = oof.corr()
    logger.info("correlation with lgb_exp022:\n" + corr["lgb_exp022"].sort_values().round(4).to_string())

    # 単純平均（全メンバー / 上位 5 つ）
    logger.info(f"simple average (all): {rmse(y, oof.mean(axis=1)):.5f}")
    logger.info(f"simple average (top 5): {rmse(y, oof[single.index[:5]].mean(axis=1)):.5f}")

    # 入れ子の CV で、重みを最適化したアンサンブルの効果を評価する
    nested = np.zeros(len(y))
    P = oof.to_numpy()
    for k in range(5):
        tr, va = folds != k, folds == k
        w = fit_weights(P[tr], y[tr])
        nested[va] = P[va] @ w
        logger.info(f"nested fold {k}: rmse={rmse(y[va], nested[va]):.5f} "
                    f"(lgb_exp022 {rmse(y[va], oof['lgb_exp022'][va]):.5f})")
    logger.info(f"optimized blend, nested CV rmse: {rmse(y, nested):.5f}")

    # 提出用: OOF 全体で重みを決める
    w = pd.Series(fit_weights(P, y), index=oof.columns)
    blend_oof, blend_test = np.clip(P @ w.to_numpy(), 1, 3), np.clip(test.to_numpy() @ w.to_numpy(), 1, 3)
    logger.info(f"optimized blend, in-sample CV rmse: {rmse(y, blend_oof):.5f}")
    logger.info("weights:\n" + w[w > 1e-4].sort_values(ascending=False).round(4).to_string())

    pd.DataFrame({"id": data.meta["train_id"], "relevance": y, "pred": blend_oof}).to_csv(OUT_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": data.meta["test_id"], "relevance": blend_test}).to_csv(OUT_DIR / "submission.csv", index=False)
    w.to_csv(OUT_DIR / "weights.csv", header=["weight"])
    logger.info(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
