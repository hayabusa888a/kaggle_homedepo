"""sentence-transformers（PyTorch, GPU）でテキストを埋め込みベクトルにし、キャッシュ（.npz）に追記する。

GPU 用の仮想環境（.venv-gpu、Python 3.12 + PyTorch CUDA）で実行する。src/embedding.py の embed_texts_gpu から呼ばれる。
    .venv-gpu/Scripts/python.exe scripts/embed_st.py --model <name> --input todo.json --cache cache.npz --prefix "..."
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="ベクトル化するテキストの JSON 配列")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=20000)
    args = parser.parse_args()

    texts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cache = Path(args.cache)
    done_texts, done_vecs = [], []
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        done_texts, done_vecs = data["texts"].tolist(), [data["vectors"]]
    known = set(done_texts)
    todo = [t for t in texts if t not in known]

    model = SentenceTransformer(args.model, device="cuda")
    model.half()  # fp16 で GPU のメモリと計算量を減らす
    model.max_seq_length = args.max_seq_length
    print(f"device={torch.cuda.get_device_name(0)} todo={len(todo)} cached={len(done_texts)}", flush=True)

    t0 = time.time()
    for start in range(0, len(todo), args.save_every):
        chunk = todo[start : start + args.save_every]
        vecs = model.encode([args.prefix + t for t in chunk], batch_size=args.batch_size,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        done_vecs.append(vecs.astype(np.float32))
        done_texts.extend(chunk)
        np.savez(cache, texts=np.array(done_texts, dtype=str), vectors=np.vstack(done_vecs))
        done, elapsed = start + len(chunk), time.time() - t0
        print(f"{done}/{len(todo)} ({done / len(todo):.0%}) elapsed {elapsed / 60:.1f}min "
              f"eta {elapsed / done * (len(todo) - done) / 60:.1f}min "
              f"gpu_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
