"""sentence-transformers の CrossEncoder（PyTorch, GPU）で（検索語, 文書）のペアを採点し、キャッシュ（.npz）に追記する。

GPU 用の仮想環境（.venv-gpu、Python 3.12 + PyTorch CUDA）で実行する。src/rerank.py の rerank_pairs_gpu から呼ばれる。
    .venv-gpu/Scripts/python.exe scripts/rerank_st.py --model <name> --input todo.json --cache cache.npz
入力の JSON は [[query, doc], ...] の配列。キャッシュは keys（query と doc を区切り文字でつないだ文字列）と scores を持つ
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

SEP = "\x1f"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=20000)
    args = parser.parse_args()

    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cache = Path(args.cache)
    done_keys, done_scores = [], []
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        done_keys, done_scores = data["keys"].tolist(), [data["scores"]]
    known = set(done_keys)
    todo = [p for p in pairs if p[0] + SEP + p[1] not in known]

    model = CrossEncoder(args.model, device="cuda", max_length=args.max_length,
                         model_kwargs={"torch_dtype": torch.float16})
    print(f"device={torch.cuda.get_device_name(0)} todo={len(todo)} cached={len(done_keys)}", flush=True)

    t0 = time.time()
    for start in range(0, len(todo), args.save_every):
        chunk = todo[start : start + args.save_every]
        # 長さの近いペアを同じバッチにまとめると、パディングが減って速くなる
        order = np.argsort([len(q) + len(d) for q, d in chunk])
        scores = np.empty(len(chunk), dtype=np.float32)
        scores[order] = model.predict([chunk[i] for i in order], batch_size=args.batch_size,
                                      show_progress_bar=False, convert_to_numpy=True)
        done_scores.append(scores)
        done_keys.extend(q + SEP + d for q, d in chunk)
        np.savez(cache, keys=np.array(done_keys, dtype=str), scores=np.concatenate(done_scores))
        done, elapsed = start + len(chunk), time.time() - t0
        print(f"{done}/{len(todo)} ({done / len(todo):.0%}) elapsed {elapsed / 60:.1f}min "
              f"eta {elapsed / done * (len(todo) - done) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
