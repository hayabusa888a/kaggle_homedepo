"""exp036（準備）: exp035 の特徴量の数値の末尾に、商品単位の特徴量を 6 列加える。

exp034・exp035 で、product_uid による「新しい商品ほど予測を下げる」補正が LB に効いた。ただ product_uid は、
train より新しい商品には、train で最も新しい帯の値を当てはめるしかない。新しい商品の性質により近い情報を与える。
どれもラベルを使わず、train + test を合わせて商品ごとに集計する。
- prod_n_rows: その商品が train + test に出てくる行数
- prod_n_attr / prod_n_bullets: 属性データの行数 / そのうち箇条書き（Bullet）の数
- prod_n_desc_chars: 説明文の文字数
- prod_ft_mean: その商品の全部の行での、ファインチューニングした 8 本のスコアの平均の平均
- prod_ft_diff: この行の 8 本の平均 − prod_ft_mean
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import load_data
from src.utils import ROOT, get_logger

SRC_DIR, DST_DIR = ROOT / "data" / "features" / "exp035", ROOT / "data" / "features" / "exp036"
FT_COLS = ["ft_ce_title", "ft_ce_title_desc", "ft_ce_large_title", "ft_ce_large_title_desc",
           "ft_ce_deberta_title_desc", "ft_ce_deberta_title", "ft_ce_modernbert_title_desc", "ft_ce_qwen3_title"]
NEW_COLS = ["prod_n_rows", "prod_n_attr", "prod_n_bullets", "prod_n_desc_chars", "prod_ft_mean", "prod_ft_diff"]


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(ROOT / "output" / "exp036_features_log.txt")
    meta = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n = meta["feature_names"], meta["n_dense"]
    ft_idx = [names.index(c) for c in FT_COLS]
    train, test = load_data()
    assert train["id"].tolist() == meta["train_id"] and test["id"].tolist() == meta["test_id"]
    df = pd.concat([train, test], ignore_index=True)
    uid = df["product_uid"]
    attr = pd.read_csv(ROOT / "input" / "attributes.csv", encoding="ISO-8859-1").dropna(subset=["product_uid"])
    attr["product_uid"] = attr["product_uid"].astype(int)
    static = pd.DataFrame({
        "prod_n_rows": uid.map(uid.value_counts()),
        "prod_n_attr": uid.map(attr.groupby("product_uid").size()).fillna(0),
        "prod_n_bullets": uid.map(attr[attr["name"].str.startswith("Bullet", na=False)].groupby("product_uid").size()).fillna(0),
        "prod_n_desc_chars": df["product_description"].fillna("").str.len(),
    }).to_numpy(np.float32)

    X = sparse.load_npz(SRC_DIR / "X_train.npz").tocsr()
    ft_tr = X[:, ft_idx].toarray().mean(axis=1)

    def new_features(ft_te):
        """train（OOF）と test のスコアを合わせて、商品ごとの平均とこの行との差を作る"""
        ft = pd.Series(np.concatenate([ft_tr, ft_te]))
        mean = ft.groupby(uid).transform("mean").to_numpy()
        f = np.hstack([static, mean[:, None], (ft.to_numpy() - mean)[:, None]]).astype(np.float32)
        return f[: len(train)], f[len(train):]

    def append(Xs, f):
        return sparse.hstack([Xs[:, :n], sparse.csr_matrix(f), Xs[:, n:]], format="csr")

    X_test = sparse.load_npz(SRC_DIR / "X_test.npz").tocsr()
    f_tr, f_te = new_features(X_test[:, ft_idx].toarray().mean(axis=1))
    sparse.save_npz(DST_DIR / "X_train.npz", append(X, f_tr), compressed=False)
    sparse.save_npz(DST_DIR / "X_test.npz", append(X_test, f_te), compressed=False)
    for k in range(5):
        te_k = np.load(SRC_DIR / f"X_test_dense_fold{k}.npy")
        np.save(DST_DIR / f"X_test_dense_fold{k}.npy", np.hstack([te_k, new_features(te_k[:, ft_idx].mean(axis=1))[1]]))
    np.save(DST_DIR / "y.npy", np.load(SRC_DIR / "y.npy"))
    meta.update({"feature_names": names[:n] + NEW_COLS + names[n:], "n_dense": n + len(NEW_COLS)})
    (DST_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    y = np.load(SRC_DIR / "y.npy")
    summary = pd.DataFrame({"train_mean": f_tr.mean(axis=0), "test_mean": f_te.mean(axis=0),
                            "corr_y": [np.corrcoef(f_tr[:, j], y)[0, 1] for j in range(len(NEW_COLS))]}, index=NEW_COLS)
    new_te = test["product_uid"].to_numpy() > train["product_uid"].max()
    summary["test_new_mean"] = f_te[new_te].mean(axis=0)
    logger.info("new features:\n" + summary.round(4).to_string())
    logger.info(f"saved: {DST_DIR} (n_dense {n + len(NEW_COLS)})")


if __name__ == "__main__":
    main()
