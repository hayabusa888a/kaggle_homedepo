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
