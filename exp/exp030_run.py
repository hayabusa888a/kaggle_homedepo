"""exp030: DeBERTa（タイトル）のスコアを加えた特徴量（data/features/exp030/）で、LightGBM 単体（全部の列 / 参考に数値だけ）を学習する。"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import exp027_feature_groups as e27
from src.utils import OUTPUT_DIR, ROOT

if __name__ == "__main__":
    e27.main(feat_dir=ROOT / "data" / "features" / "exp030", out_dir=OUTPUT_DIR / "exp030", group_names=["all", "num"])
