"""exp028: test のファインチューニングのスコアを「その検索語を見ていないモデル」の予測にそろえた特徴量
（data/features/exp028/）で、LightGBM 単体（全部の列 / 数値の特徴量だけ）を学習する。exp027 のスクリプトを使う。"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import exp027_feature_groups as e27
from src.utils import OUTPUT_DIR, ROOT

if __name__ == "__main__":
    e27.main(feat_dir=ROOT / "data" / "features" / "exp028", out_dir=OUTPUT_DIR / "exp028", group_names=["all", "num"])
