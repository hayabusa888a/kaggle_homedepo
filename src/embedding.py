"""ローカルの埋め込みモデル（Ollama）によるテキストのベクトル化"""
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"


def _embed(model: str, texts: list[str]) -> np.ndarray:
    """1リクエスト分をベクトル化する。エラーが返ったらバッチを半分に分けて送り直す（大きいバッチを並列に送ると 400 が返ることがある）"""
    payload = {"model": model, "input": texts}
    req = urllib.request.Request(OLLAMA_EMBED_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            return np.asarray(json.load(res)["embeddings"], dtype=np.float32)
    except urllib.error.HTTPError:
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        return np.vstack([_embed(model, texts[:mid]), _embed(model, texts[mid:])])


def _save(cache_path: Path, texts: list[str], vectors: list[np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, texts=np.array(texts, dtype=str), vectors=np.vstack(vectors))


def embed_texts(texts: pd.Series, prefix: str, cache_path: Path, model: str = "nomic-embed-text",
                batch_size: int = 64, workers: int = 8, logger=None, save_every: int = 10000) -> np.ndarray:
    """texts の各行を埋め込みベクトル（L2 正規化済み）にして返す。ユニークなテキストだけを、
    batch_size 件ずつ workers 本の並列リクエストで処理する。結果は cache_path（.npz: texts と vectors）に
    save_every 件ごとに保存し、保存済みのテキストは再計算しない。
    prefix は nomic-embed-text の用途の目印（"search_query: " / "search_document: "）"""
    done_texts: list[str] = []
    done_vecs: list[np.ndarray] = []
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        done_texts, done_vecs = data["texts"].tolist(), [data["vectors"]]

    known = set(done_texts)
    uniq = texts.drop_duplicates()
    todo = [t for t in uniq if t not in known]
    if logger:
        logger.info(f"embed {cache_path.stem}: {len(uniq)} unique, {len(uniq) - len(todo)} cached, {len(todo)} to process")

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for start in range(0, len(todo), save_every):
            chunk = todo[start : start + save_every]
            batches = [[prefix + t for t in chunk[i : i + batch_size]] for i in range(0, len(chunk), batch_size)]
            done_vecs.append(np.vstack(list(ex.map(lambda b: _embed(model, b), batches))))
            done_texts.extend(chunk)
            _save(cache_path, done_texts, done_vecs)
            if logger:
                done, elapsed = start + len(chunk), time.time() - t0
                logger.info(f"embed {cache_path.stem}: {done}/{len(todo)} ({done / len(todo):.0%}) "
                            f"elapsed {elapsed / 60:.1f}min eta {elapsed / done * (len(todo) - done) / 60:.1f}min")

    vectors = np.vstack(done_vecs)
    index = {t: i for i, t in enumerate(done_texts)}
    # コサイン類似度を内積で計算できるように L2 正規化する
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors[texts.map(index).to_numpy()]


GPU_PYTHON = Path(__file__).resolve().parents[1] / ".venv-gpu" / "Scripts" / "python.exe"
EMBED_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "embed_st.py"


def embed_texts_gpu(texts: pd.Series, prefix: str, cache_path: Path, model: str,
                    batch_size: int = 128, logger=None) -> np.ndarray:
    """embed_texts と同じことを、GPU 用の仮想環境の sentence-transformers（PyTorch）で行う。
    キャッシュにないテキストだけを scripts/embed_st.py に渡し、結果（L2 正規化済み）を texts の行順で返す"""
    import subprocess

    uniq = texts.drop_duplicates().tolist()
    known = set()
    if cache_path.exists():
        known = set(np.load(cache_path, allow_pickle=False)["texts"].tolist())
    todo = [t for t in uniq if t not in known]
    if logger:
        logger.info(f"embed {cache_path.stem}: {len(uniq)} unique, {len(uniq) - len(todo)} cached, {len(todo)} to process")
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        todo_path = cache_path.with_suffix(".todo.json")
        todo_path.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
        cmd = [str(GPU_PYTHON), str(EMBED_SCRIPT), "--model", model, "--input", str(todo_path),
               "--cache", str(cache_path), "--prefix", prefix, "--batch-size", str(batch_size)]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace") as proc:
            for line in proc.stdout:
                line = line.rstrip()
                # 進捗の行（"12000/110128 ..." と "device=..."）だけを残し、ライブラリの警告などは捨てる
                if logger and (line[:1].isdigit() or line.startswith("device=")):
                    logger.info(f"embed {cache_path.stem}: {line}")
        if proc.returncode != 0:
            raise RuntimeError(f"embed_st.py failed (exit {proc.returncode})")
        todo_path.unlink()

    data = np.load(cache_path, allow_pickle=False)
    index = {t: i for i, t in enumerate(data["texts"].tolist())}
    vectors = data["vectors"]
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors[texts.map(index).to_numpy()]
