"""cross-encoder を 1 つの fold についてファインチューニングし、その fold（学習に使っていない行）と test を予測する。

GPU 用の仮想環境（.venv-gpu）で実行する。src/finetune.py の finetune_ce_oof から fold ごとに呼ばれる。
    .venv-gpu/Scripts/python.exe scripts/finetune_ce.py --data input.json --fold 0 --out-dir <dir> --model <name>
入力の JSON: {"train": [[query, doc, label(0〜1), fold], ...], "test": [[query, doc], ...]}
出力: <out-dir>/oof_fold{k}.npy（fold k の行のスコア、入力の順）、<out-dir>/test_fold{k}.npy
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from transformers import TrainerCallback


# Qwen3-Reranker の入力の形（モデルカードのもの）。この形で "yes" と答える確からしさをスコアとして学習済み
QWEN3_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
                'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
                '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: ')
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def apply_qwen3_template(rows, tokenizer, max_length):
    """検索語と文書を Qwen3-Reranker の形に包む。長い文書は、末尾の形の部分が切れないように、
    文書の本文だけを先にトークン数で切り詰めてから包む"""
    queries = [QWEN3_PREFIX + r[0] + "\n" for r in rows]
    q_len = [len(x) for x in tokenizer(queries, add_special_tokens=False)["input_ids"]]
    doc_ids = tokenizer([r[1] for r in rows], add_special_tokens=False)["input_ids"]
    fixed = len(tokenizer("<Document>: " + QWEN3_SUFFIX, add_special_tokens=False)["input_ids"]) + 2
    docs = [tokenizer.decode(ids[: max(max_length - n - fixed, 8)]) for ids, n in zip(doc_ids, q_len)]
    return [[q, "<Document>: " + d + QWEN3_SUFFIX, *r[2:]] for q, d, r in zip(queries, docs, rows)]


class ProgressCallback(TrainerCallback):
    """一定のステップごとに、学習の損失と経過時間を出す"""

    def __init__(self):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            elapsed = time.time() - self.t0
            eta = elapsed / max(state.global_step, 1) * (state.max_steps - state.global_step)
            print(f"step {state.global_step}/{state.max_steps} epoch {state.epoch:.2f} loss {logs['loss']:.4f} "
                  f"elapsed {elapsed / 60:.1f}min eta {eta / 60:.1f}min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--grad-accum", type=int, default=1, help="この回数分の勾配をためてから更新する（大きいモデルでメモリを節約）")
    parser.add_argument("--freeze-embeddings", action="store_true", help="単語の埋め込み層を学習しない（多言語モデルでは全体の約半分を占める）")
    parser.add_argument("--grad-checkpointing", action="store_true", help="中間結果を保存せず計算し直して、メモリを節約する")
    parser.add_argument("--optim", default="adamw_torch", help="最適化手法（例: adamw_torch, adamw_bnb_8bit, adafactor）")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--attn", default=None, help="アテンションの実装（例: eager）。None ならライブラリの既定")
    parser.add_argument("--template", choices=["qwen3"], default=None, help="入力をモデル固有の形に包む（Qwen3-Reranker 用）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    oof_path, test_path = out_dir / f"oof_fold{args.fold}.npy", out_dir / f"test_fold{args.fold}.npy"
    if oof_path.exists() and test_path.exists():
        print(f"fold {args.fold}: cached", flush=True)
        return

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tr = [r for r in data["train"] if r[3] != args.fold]
    va = [r for r in data["train"] if r[3] == args.fold]
    print(f"device={torch.cuda.get_device_name(0)} fold={args.fold} train={len(tr)} valid={len(va)} "
          f"test={len(data['test'])}", flush=True)

    model_kwargs = {"attn_implementation": args.attn} if args.attn else {}
    model = CrossEncoder(args.model, num_labels=1, max_length=args.max_length, device="cuda", model_kwargs=model_kwargs)
    hf_model = getattr(model, "model", None) or model[0].auto_model
    if args.template == "qwen3":
        tr, va, data["test"] = (apply_qwen3_template(rows, model.tokenizer, args.max_length) for rows in (tr, va, data["test"]))
    train_ds = Dataset.from_dict({"query": [r[0] for r in tr], "doc": [r[1] for r in tr],
                                  "label": [float(r[2]) for r in tr]})
    if args.freeze_embeddings:
        # エンコーダ（BERT 系）は embeddings、デコーダ（Qwen など）は embed_tokens が単語の埋め込み層
        emb = getattr(hf_model.base_model, "embeddings", None) or hf_model.base_model.embed_tokens
        for param in emb.parameters():
            param.requires_grad = False
    n_train = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in hf_model.parameters())
    print(f"trainable params: {n_train / 1e6:.0f}M / {n_all / 1e6:.0f}M", flush=True)
    if args.grad_checkpointing:
        # Trainer 経由の設定は sentence-transformers 側が引数を受け付けないので、中のモデルに直接設定する
        hf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / f"tmp_fold{args.fold}"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim=args.optim,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=args.precision == "bf16",
        logging_steps=200,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )
    # relevance を 0〜1 に変換した値を、BCE（ロジットに対する二値交差エントロピー）で学習する
    trainer = CrossEncoderTrainer(model=model, args=train_args, train_dataset=train_ds,
                                  loss=BinaryCrossEntropyLoss(model), callbacks=[ProgressCallback()])
    trainer.train()
    # 予測の前に、学習用のデータ（最適化手法の内部状態など）を GPU から解放する
    del trainer
    torch.cuda.empty_cache()
    model.eval()

    def predict(pairs, chunk=20000):
        # モデルによって出力がロジットのままのもの（MiniLM など）と 0〜1 のものがあるので、常にシグモイドで 0〜1 にそろえる。
        # 大きいモデルで長い入力を通常の精度でまとめて予測するとメモリがあふれるので、bf16 で行う
        out = []
        for start in range(0, len(pairs), chunk):
            part = [[q, d] for q, d, *_ in pairs[start : start + chunk]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(np.asarray(model.predict(part, batch_size=args.predict_batch_size,
                                                    activation_fn=torch.nn.Sigmoid(), show_progress_bar=False,
                                                    convert_to_numpy=True), dtype=np.float32))
            if len(pairs) > chunk:
                print(f"predict {min(start + chunk, len(pairs))}/{len(pairs)}", flush=True)
        return np.concatenate(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    oof = predict(va)
    np.save(oof_path, oof)
    np.save(test_path, predict(data["test"]))
    labels = np.array([r[2] for r in va])
    rmse = float(np.sqrt(np.mean((oof * 2 + 1 - (labels * 2 + 1)) ** 2)))
    corr = float(np.corrcoef(oof, labels)[0, 1]) if oof.std() > 0 else float("nan")
    print(f"fold {args.fold}: done, standalone rmse={rmse:.5f} pred_std={oof.std():.4f} pearson={corr:.4f} "
          f"gpu_mem_peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
