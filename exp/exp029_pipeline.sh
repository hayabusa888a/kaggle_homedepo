#!/usr/bin/env bash
# exp029 を最後まで通して実行する（独立したプロセスとして起動する）。
#   1. DeBERTa（mxbai-rerank-base-v1）を、検索語 × タイトル + 説明文で 5 fold ファインチューニングする。
#      transformers 5.x では DeBERTa の学習が NaN になるため、transformers 4.49 の仮想環境（.venv-deberta）を使う。
#      入力データ（fold の分け方を含む）は exp020 と同じものを使う。保存済みの fold は飛ばす
#   2. exp026 の特徴量にスコアと相対値を加える
#   3. LightGBM 単体を学習する
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=output/exp029_pipeline_log.txt
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
SRC=data/processed/ft_ce_bge_base_title_desc
OUT=data/processed/ft_ce_mxbai_base_title_desc
mkdir -p "$OUT" && cp "$SRC/input.json" "$OUT/input.json"

log "start"
for k in 0 1 2 3 4; do
  .venv-deberta/Scripts/python.exe scripts/finetune_ce.py --data "$OUT/input.json" --fold "$k" --out-dir "$OUT" \
    --model mixedbread-ai/mxbai-rerank-base-v1 --epochs 2 --lr 2e-5 --max-length 256 --batch-size 16 --grad-accum 2 \
    --predict-batch-size 64 --precision bf16 2>&1 \
    | grep --line-buffered -E "^(fold|device|trainable|predict)" >> output/exp029_finetune_log.txt || { log "finetune failed"; exit 1; }
done
log "finetune done"
python -B exp/exp029_build_features.py >> "$LOG" 2>&1 || { log "build features failed"; exit 1; }
log "features done"
python -B exp/exp029_run.py >> "$LOG" 2>&1 || { log "lgb failed"; exit 1; }
log "pipeline done"
