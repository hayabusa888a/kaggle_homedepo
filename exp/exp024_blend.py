"""exp024: fold ごとにそろえた test の特徴量で学習し直したメンバーを混ぜる（混ぜ方は exp023 と同じ）。

提出ファイルを 2 つ作る:
- submission.csv: アンサンブル（exp023 と比べる用）
- submission_lgb.csv: LightGBM 単体（exp022 と同じ設定。対策そのものの効果を exp022 と比べる用）
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from exp023_blend import fit_weights
from exp023_members import Data
from exp024_members import MEMBER_DIR, OUT_DIR
from src.utils import get_logger, rmse


def main():
    logger = get_logger(OUT_DIR / "blend_log.txt")
    data = Data()
    y, folds = data.y, data.folds
    oof, test = {}, {}
    for path in sorted(MEMBER_DIR.glob("*.npz")):
        d = np.load(path)
        oof[path.stem], test[path.stem] = d["oof"], d["test"]
    oof, test = pd.DataFrame(oof), pd.DataFrame(test)
    logger.info(f"members: {oof.shape[1]}")
    logger.info("single CV rmse:\n" + pd.Series({c: rmse(y, oof[c]) for c in oof}).sort_values().round(5).to_string())

    P = oof.to_numpy()
    nested = np.zeros(len(y))
    for k in range(5):
        tr, va = folds != k, folds == k
        nested[va] = P[va] @ fit_weights(P[tr], y[tr])
        logger.info(f"nested fold {k}: rmse={rmse(y[va], nested[va]):.5f} (lgb_base {rmse(y[va], oof['lgb_base'][va]):.5f})")
    logger.info(f"optimized blend, nested CV rmse: {rmse(y, nested):.5f}")

    w = pd.Series(fit_weights(P, y), index=oof.columns)
    blend_oof, blend_test = np.clip(P @ w.to_numpy(), 1, 3), np.clip(test.to_numpy() @ w.to_numpy(), 1, 3)
    logger.info(f"optimized blend, in-sample CV rmse: {rmse(y, blend_oof):.5f}")
    logger.info("weights:\n" + w[w > 1e-4].sort_values(ascending=False).round(4).to_string())

    ids, test_ids = data.meta["train_id"], data.meta["test_id"]
    pd.DataFrame({"id": ids, "relevance": y, "pred": blend_oof}).to_csv(OUT_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "relevance": blend_test}).to_csv(OUT_DIR / "submission.csv", index=False)
    pd.DataFrame({"id": test_ids, "relevance": test["lgb_base"]}).to_csv(OUT_DIR / "submission_lgb.csv", index=False)
    w.to_csv(OUT_DIR / "weights.csv", header=["weight"])

    # 対策の効果の確認: exp022 / exp023 の test の予測との違い
    sub22 = pd.read_csv(OUT_DIR.parent / "exp022" / "submission.csv").set_index("id").loc[test_ids, "relevance"].to_numpy()
    sub23 = pd.read_csv(OUT_DIR.parent / "exp023" / "submission.csv").set_index("id").loc[test_ids, "relevance"].to_numpy()
    logger.info(f"test pred mean/std: lgb_base {test['lgb_base'].mean():.4f}/{test['lgb_base'].std():.4f}, "
                f"exp022 {sub22.mean():.4f}/{sub22.std():.4f}, blend {blend_test.mean():.4f}/{blend_test.std():.4f}, "
                f"exp023 {sub23.mean():.4f}/{sub23.std():.4f}")
    logger.info(f"rmse between test preds: lgb_base vs exp022 {rmse(sub22, test['lgb_base']):.5f}, "
                f"blend vs exp023 {rmse(sub23, blend_test):.5f}")
    logger.info(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
