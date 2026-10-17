"""cross-encoder（リランカー）による（検索語, 文書）ペアの関連度スコア"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.embedding import GPU_PYTHON

RERANK_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rerank_st.py"
SEP = "\x1f"


def rerank_pairs_gpu(queries: pd.Series, docs: pd.Series, cache_path: Path, model: str,
                     batch_size: int = 64, logger=None) -> np.ndarray:
    """各行の（検索語, 文書）ペアを cross-encoder で採点し、スコア（ロジット）を行順で返す。
    ユニークなペアのうちキャッシュにないものだけを、GPU 用の仮想環境の scripts/rerank_st.py で処理する"""
    keys = queries + SEP + docs
    uniq = keys.drop_duplicates()
    known = set()
    if cache_path.exists():
        known = set(np.load(cache_path, allow_pickle=False)["keys"].tolist())
    todo = [k for k in uniq if k not in known]
    if logger:
        logger.info(f"rerank {cache_path.stem}: {len(uniq)} unique pairs, {len(uniq) - len(todo)} cached, {len(todo)} to process")
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        todo_path = cache_path.with_suffix(".todo.json")
        todo_path.write_text(json.dumps([k.split(SEP, 1) for k in todo], ensure_ascii=False), encoding="utf-8")
        cmd = [str(GPU_PYTHON), str(RERANK_SCRIPT), "--model", model, "--input", str(todo_path),
               "--cache", str(cache_path), "--batch-size", str(batch_size)]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace") as proc:
            for line in proc.stdout:
                line = line.rstrip()
                # 進捗の行だけを残し、ライブラリの警告などは捨てる
                if logger and (line[:1].isdigit() or line.startswith("device=")):
                    logger.info(f"rerank {cache_path.stem}: {line}")
        if proc.returncode != 0:
            raise RuntimeError(f"rerank_st.py failed (exit {proc.returncode})")
        todo_path.unlink()

    data = np.load(cache_path, allow_pickle=False)
    index = {k: i for i, k in enumerate(data["keys"].tolist())}
    return data["scores"][keys.map(index).to_numpy()].astype(np.float32)
