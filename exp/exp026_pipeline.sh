#!/usr/bin/env bash
# exp026 を最後まで通して実行する（独立したプロセスとして起動する）。
#   1. v2-m3（検索語 × タイトル）のファインチューニングを、乱数の種だけ変えてもう一度行う（exp021 と同じ設定。保存済みの fold は飛ばす）
#   2. exp025 の特徴量の ft_ce_large_title を 2 つの種の平均に置き換える
#   3. exp024 と同じメンバーを学習し、混ぜる
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=output/exp026_pipeline_log.txt
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
SRC=data/processed/ft_ce_bge_v2m3_title
OUT=data/processed/ft_ce_bge_v2m3_title_seed2
mkdir -p "$OUT" && cp "$SRC/input.json" "$OUT/input.json"

log "start"
for k in 0 1 2 3 4; do
  .venv-gpu/Scripts/python.exe scripts/finetune_ce.py --data "$SRC/input.json" --fold "$k" --out-dir "$OUT" \
    --model BAAI/bge-reranker-v2-m3 --epochs 2 --lr 2e-5 --max-length 128 --batch-size 32 \
    --freeze-embeddings --optim adamw_bnb_8bit --seed 2025 2>&1 \
    | grep --line-buffered -E "^(fold|device|trainable)" >> output/exp026_finetune_log.txt || { log "finetune failed"; exit 1; }
done
log "finetune done"
python -B exp/exp026_build_features.py >> "$LOG" 2>&1 || { log "build features failed"; exit 1; }
log "features done"
python -B exp/exp026_run.py members >> "$LOG" 2>&1 || { log "members failed"; exit 1; }
.venv-gpu/Scripts/python.exe -B exp/exp026_run.py mlp >> "$LOG" 2>&1 || { log "mlp failed"; exit 1; }
python -B exp/exp026_run.py blend >> "$LOG" 2>&1 || { log "blend failed"; exit 1; }
log "pipeline done"
