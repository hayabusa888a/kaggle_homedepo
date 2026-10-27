"""exp031〜033: 別系統のリランカーのスコアを加えた特徴量（data/features/<exp>/）で、LightGBM 単体（全部の列 / 参考に数値だけ）を学習する。
    python exp/exp031_run.py exp031
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import exp027_feature_groups as e27
from src.utils import OUTPUT_DIR, ROOT

if __name__ == "__main__":
    exp = sys.argv[1]
    e27.main(feat_dir=ROOT / "data" / "features" / exp, out_dir=OUTPUT_DIR / exp, group_names=["all", "num"])
