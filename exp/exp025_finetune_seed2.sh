#!/usr/bin/env bash
# exp025（準備）: exp022 の v2-m3（検索語 × タイトル + 説明文）のファインチューニングを、乱数の種だけ変えてもう一度行う。
# 入力データ（input.json、fold の分け方）は exp022 と同じものを使う。保存済みの fold は飛ばす。
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=data/processed/ft_ce_bge_v2m3_title_desc
OUT=data/processed/ft_ce_bge_v2m3_title_desc_seed2
mkdir -p "$OUT"
for k in 0 1 2 3 4; do
  HF_HUB_DISABLE_SYMLINKS_WARNING=1 .venv-gpu/Scripts/python.exe scripts/finetune_ce.py \
    --data "$SRC/input.json" --fold "$k" --out-dir "$OUT" --model BAAI/bge-reranker-v2-m3 \
    --epochs 2 --lr 2e-5 --max-length 256 --batch-size 8 --grad-accum 4 --predict-batch-size 64 \
    --freeze-embeddings --optim adamw_bnb_8bit --seed 2025 2>&1 \
    | grep --line-buffered -E "^(step|fold|device|trainable|predict)"
done
echo "all folds done"
