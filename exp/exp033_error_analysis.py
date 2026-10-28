"""exp033（現状のベスト、LightGBM 単体・全部の列）の誤差分析。

exp026 の LightGBM 単体（output/exp027/all.npz の all。exp026 と同じ構成を学習し直したもの）と比べて、
どこが改善し、どこが外れたままかを調べる。結果は output/exp033_analysis/log.txt に出す。
"""
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import sparse

import exp023_members as m23
from src.finetune import fold_ids
from src.utils import OUTPUT_DIR, ROOT, get_logger

FEAT_DIR = ROOT / "data" / "features" / "exp033"
OUT_DIR = OUTPUT_DIR / "exp033_analysis"
FT_COLS = ["ft_ce_title", "ft_ce_title_desc", "ft_ce_large_title", "ft_ce_large_title_desc",
           "ft_ce_deberta_title_desc", "ft_ce_deberta_title", "ft_ce_modernbert_title_desc", "ft_ce_qwen3_title"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def segment_table(df, key, cols=("pred33", "pred26")):
    """key の値ごとに、行数・正解の平均・予測の平均・RMSE（exp033 / exp026）を返す"""
    g = df.groupby(key, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "share": g.size() / len(df),
        "y_mean": g["y"].mean(),
        "pred_mean": g["pred33"].mean(),
        "rmse33": g.apply(lambda d: rmse(d["y"], d["pred33"])),
        "rmse26": g.apply(lambda d: rmse(d["y"], d["pred26"])),
        "err_share": g.apply(lambda d: ((d["y"] - d["pred33"]) ** 2).sum()) / ((df["y"] - df["pred33"]) ** 2).sum(),
    })
    out["diff"] = out["rmse33"] - out["rmse26"]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(OUT_DIR / "log.txt")
    pd.set_option("display.width", 200)
    meta = json.loads((FEAT_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n = meta["feature_names"], meta["n_dense"]
    X = sparse.load_npz(FEAT_DIR / "X_train.npz").tocsr()
    dense = pd.DataFrame(X[:, :n].toarray(), columns=names[:n])
    y = np.load(FEAT_DIR / "y.npy")

    train = pd.read_csv(ROOT / "input" / "train.csv", encoding="ISO-8859-1")
    test = pd.read_csv(ROOT / "input" / "test.csv", encoding="ISO-8859-1")
    z33, z26 = np.load(OUTPUT_DIR / "exp033" / "all.npz"), np.load(OUTPUT_DIR / "exp027" / "all.npz")
    df = pd.DataFrame({"search_term": train["search_term"], "title": train["product_title"],
                       "uid": train["product_uid"], "y": y, "pred33": z33["oof"], "pred26": z26["oof"]})
    df["res"] = df["y"] - df["pred33"]
    logger.info(f"CV: exp033 {rmse(df.y, df.pred33):.5f} / exp026 {rmse(df.y, df.pred26):.5f}")

    # 1. 正解の値ごと
    df["y_bin"] = pd.cut(df["y"], [0, 1.34, 1.67, 2.0, 2.34, 2.67, 3.0], labels=["1.00-1.33", "1.67", "2.00", "2.33", "2.67", "3.00"])
    logger.info(f"pred std: exp033 {df.pred33.std():.3f} / exp026 {df.pred26.std():.3f} / y {df.y.std():.3f}")
    logger.info("1. by true relevance:\n" + segment_table(df, "y_bin").round(4).to_string())

    # 2. 検索語・商品の性質ごと
    df["q_words"] = dense["q_n_words"].clip(upper=5).astype(int).astype(str).replace("5", "5+")
    logger.info("2a. by number of query words:\n" + segment_table(df, "q_words").round(4).to_string())
    df["match"] = pd.cut(dense["ratio_match_title"], [-0.01, 0, 0.5, 0.99, 1.0], labels=["0", "(0,0.5]", "(0.5,1)", "1"])
    logger.info("2b. by share of query words in title:\n" + segment_table(df, "match").round(4).to_string())
    df["uid_bin"] = pd.cut(df["uid"], [100000, 150000, 180000, 195000, 206650])
    logger.info("2c. by product_uid:\n" + segment_table(df, "uid_bin").round(4).to_string())
    df["q_size_bin"] = pd.cut(dense["q_size"], [0, 3, 6, 10, 1000], labels=["1-3", "4-6", "7-10", "11+"])
    logger.info("2d. by number of products per query (train + test):\n" + segment_table(df, "q_size_bin").round(4).to_string())
    df["has_digit"] = train["search_term"].str.contains(r"\d").map({True: "digit", False: "no digit"})
    logger.info("2e. query contains a digit:\n" + segment_table(df, "has_digit").round(4).to_string())

    # 3. ファインチューニングした 8 本のスコアの、意見の割れ方（1〜3 に換算した標準偏差）
    ft = dense[FT_COLS] * 2 + 1
    df["ft_mean"], df["ft_std"] = ft.mean(axis=1), ft.std(axis=1)
    df["ft_std_bin"] = pd.qcut(df["ft_std"], 5, labels=["q1 (agree)", "q2", "q3", "q4", "q5 (disagree)"])
    logger.info("3a. by disagreement among 8 fine-tuned scores:\n" + segment_table(df, "ft_std_bin").round(4).to_string())
    single = {c: rmse(df.y, ft[c]) for c in FT_COLS}
    single["mean of 8"] = rmse(df.y, df.ft_mean)
    logger.info("3b. standalone rmse of each fine-tuned score and their mean:\n" + pd.Series(single).round(5).to_string())
    corr = pd.DataFrame({"y": df.y, "pred33": df.pred33, **{c: ft[c] for c in FT_COLS}}).corr()
    logger.info("3c. correlation with y:\n" + corr["y"].round(4).to_string())

    # 4. exp026 → exp033 で大きく変わった行
    df["change"] = df["pred33"] - df["pred26"]
    df["gain"] = (df["y"] - df["pred26"]) ** 2 - (df["y"] - df["pred33"]) ** 2
    logger.info(f"4. prediction change exp026 -> exp033: std {df.change.std():.4f}, "
                f"rows improved {(df.gain > 0).mean():.1%}, worsened {(df.gain < 0).mean():.1%}")
    cols = ["search_term", "title", "y", "pred26", "pred33"]
    logger.info("4a. most improved rows:\n" + df.nlargest(10, "gain")[cols].round(3).to_string())
    logger.info("4b. most worsened rows:\n" + df.nsmallest(10, "gain")[cols].round(3).to_string())

    # 5. 大きく外した行
    df["abs_res"] = df["res"].abs()
    logger.info(f"5. rows with |residual| > 1: {(df.abs_res > 1).sum()} ({(df.abs_res > 1).mean():.2%}), "
                f"their share of squared error {((df.res ** 2)[df.abs_res > 1]).sum() / (df.res ** 2).sum():.1%}")
    logger.info("5a. worst over-predictions:\n" + df.nsmallest(12, "res")[cols + ["ft_mean"]].round(3).to_string())
    logger.info("5b. worst under-predictions:\n" + df.nlargest(12, "res")[cols + ["ft_mean"]].round(3).to_string())

    # 6. ラベルのノイズの目安（同じ検索語で、数字を除いたタイトルが同じ商品の組）
    key = df["search_term"] + "\x1f" + df["title"].map(lambda t: re.sub(r"[\d./-]+", "", t).lower().strip())
    size = key.map(key.value_counts())
    dup = df[size >= 2].copy()
    dup["grp_mean"] = dup.groupby(key[size >= 2])["y"].transform("mean")
    logger.info(f"6. near-duplicate groups: {key[size >= 2].nunique()} groups, {len(dup)} rows. "
                f"rmse to group mean {rmse(dup.y, dup.grp_mean):.4f}, exp033 {rmse(dup.y, dup.pred33):.4f}, "
                f"exp026 {rmse(dup.y, dup.pred26):.4f}")

    # 7. 特徴量の重要度（fold 0 で LightGBM を 1 本学習し、gain をグループごとに集計）
    import lightgbm as lgb
    folds = fold_ids(pd.Series(meta["groups"]))
    tr, va = folds != 0, folds == 0
    model = lgb.train({**m23.LGB_BASE, "seed": 42}, lgb.Dataset(X[tr], y[tr]), num_boost_round=10000,
                      valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    gain = pd.Series(model.feature_importance("gain"), index=names)
    gain = gain / gain.sum()

    def group(c):
        if names.index(c) >= n:
            return {"eq": "emb vector (query)", "et": "emb vector (title)", "q": "tfidf vector (query)",
                    "t": "tfidf vector (title)"}.get(c.split("_")[0], "other vector")
        if c.startswith("q_ft_ce") or c.startswith("q_ce") or (c.startswith("q_") and c.endswith(("_diff_mean", "_diff_max", "_rank"))):
            return "query-relative"
        if c.startswith("ft_ce"):
            return "fine-tuned score"
        if c.startswith(("ce_", "emb_cos")):
            return "pretrained ce / emb cos"
        return "classic"
    logger.info(f"7. fold 0 model: best_iteration {model.best_iteration}")
    logger.info("7a. gain share by group:\n" + gain.groupby(gain.index.map(group)).sum().sort_values(ascending=False).round(4).to_string())
    logger.info("7b. top 25 features by gain:\n" + gain.nlargest(25).round(4).to_string())

    # 8. test の予測（商品の新しさの帯ごと、exp026 との比較）
    t = pd.DataFrame({"uid": test["product_uid"], "pred33": z33["test"], "pred26": z26["test"]})
    t["uid_bin"] = pd.cut(t["uid"], [100000, 150000, 180000, 195000, 206650, 300000])
    seen = set(train["search_term"])
    t["seen_query"] = test["search_term"].isin(seen)
    g = t.groupby("uid_bin", observed=True)
    logger.info("8. test predictions by product_uid:\n" + pd.DataFrame({
        "n": g.size(), "seen_query": g["seen_query"].mean(), "pred33": g["pred33"].mean(), "pred26": g["pred26"].mean(),
        "change": g.apply(lambda d: (d.pred33 - d.pred26).mean())}).round(4).to_string())
    logger.info(f"   train: y mean {df.y.mean():.4f}, OOF pred mean {df.pred33.mean():.4f}; "
                f"test pred mean {t.pred33.mean():.4f} (exp026 {t.pred26.mean():.4f})")


if __name__ == "__main__":
    main()
