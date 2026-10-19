"""exp026: exp025 に加えて v2-m3（タイトルのみ）も 2 つの種の平均に置き換えた特徴量（data/features/exp026/）で、exp024 と同じメンバーの学習と混ぜ合わせを行う。

exp024 のスクリプトを、読み込む特徴量と出力先だけ変えて使う。
    python exp/exp026_run.py members
    .venv-gpu/Scripts/python.exe exp/exp026_run.py mlp
    python exp/exp026_run.py blend
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import exp023_members as m23
from src.utils import OUTPUT_DIR, ROOT

m23.FEAT_DIR = ROOT / "data" / "features" / "exp026"
OUT_DIR = OUTPUT_DIR / "exp026"

import exp024_members as m24  # noqa: E402

m24.OUT_DIR, m24.MEMBER_DIR = OUT_DIR, OUT_DIR / "members"

if __name__ == "__main__":
    step = sys.argv[1]
    if step == "members":
        m24.main()
    elif step == "mlp":
        sys.argv.append("--mlp")
        m24.main()
    elif step == "blend":
        import exp024_blend as b24
        b24.OUT_DIR, b24.MEMBER_DIR = OUT_DIR, OUT_DIR / "members"
        b24.main()
