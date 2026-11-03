"""exp035（準備）: exp033 の特徴量の数値の末尾に product_uid（商品の新しさ）を 1 列加える。

exp034 で、2 段目の LightGBM が product_uid を使って「新しい商品ほど予測を下げる」補正を学び、LB が改善した。
1 段目の LightGBM にも入れて、ほかの特徴量と組み合わせた補正を学べるようにする。
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse

from src.data import load_data
from src.utils import ROOT, get_logger

SRC_DIR, DST_DIR = ROOT / "data" / "features" / "exp033", ROOT / "data" / "features" / "exp035"


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger(ROOT / "output" / "exp035_features_log.txt")
    meta = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))
    names, n = meta["feature_names"], meta["n_dense"]
    train, test = load_data()
    assert train["id"].tolist() == meta["train_id"] and test["id"].tolist() == meta["test_id"]
    uid_tr = train["product_uid"].to_numpy(np.float32)[:, None]
    uid_te = test["product_uid"].to_numpy(np.float32)[:, None]

    def insert(X, uid):
        return sparse.hstack([X[:, :n], sparse.csr_matrix(uid), X[:, n:]], format="csr")

    for split, uid in [("train", uid_tr), ("test", uid_te)]:
        X = sparse.load_npz(SRC_DIR / f"X_{split}.npz").tocsr()
        sparse.save_npz(DST_DIR / f"X_{split}.npz", insert(X, uid), compressed=False)
    for k in range(5):
        np.save(DST_DIR / f"X_test_dense_fold{k}.npy", np.hstack([np.load(SRC_DIR / f"X_test_dense_fold{k}.npy"), uid_te]))
    np.save(DST_DIR / "y.npy", np.load(SRC_DIR / "y.npy"))
    meta.update({"feature_names": names[:n] + ["product_uid"] + names[n:], "n_dense": n + 1})
    (DST_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    logger.info(f"saved: {DST_DIR} (n_dense {n + 1})")


if __name__ == "__main__":
    main()
