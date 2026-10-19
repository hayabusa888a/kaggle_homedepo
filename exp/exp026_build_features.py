"""exp026（準備）: exp025 の特徴量の ft_ce_large_title（v2-m3、タイトルのみ）を、2 つの種のモデルの平均に置き換える。"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import exp025_build_features as b25
from src.utils import ROOT

if __name__ == "__main__":
    b25.main(src_dir=ROOT / "data" / "features" / "exp025", dst_dir=ROOT / "data" / "features" / "exp026",
             seed_dirs=(b25.PROC / "ft_ce_bge_v2m3_title", b25.PROC / "ft_ce_bge_v2m3_title_seed2"),
             col="ft_ce_large_title", doc="title", log_name="exp026_features_log.txt")
