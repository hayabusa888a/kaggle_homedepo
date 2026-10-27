#!/usr/bin/env bash
# exp031〜033 を最後まで通して実行する（独立したプロセスとして起動する）。exp030 のパイプラインが終わるのを待ってから始める。
#   exp031: ModernBERT 系のリランカー（gte-reranker-modernbert-base）を、検索語 × タイトル + 説明文で 5 fold ファインチューニング
#   exp032: LLM 系のリランカー（Qwen3-Reranker-0.6B、分類用に変換したもの）を、検索語 × タイトルで 5 fold ファインチューニング
#           （モデル固有の入力の形で包む。埋め込み層の固定 + 8bit AdamW で 8GB に収める）
#   exp033: exp029 に DeBERTa（タイトル）・ModernBERT・Qwen3 の 3 つを全部加える
# それぞれ exp029 の特徴量にスコアと相対値を加えて、LightGBM 単体を学習する。どれかが失敗しても次へ進む。
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=output/exp031_pipeline_log.txt
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
PROC=data/processed

log "waiting for exp030"
until grep -qE "pipeline done|failed" output/exp030_pipeline_log.txt 2>/dev/null; do sleep 60; done
log "start"

# finetune <exp> <出力フォルダ> <入力の元フォルダ> <モデル> <その他の引数...>
finetune() {
  local exp=$1 out=$PROC/$2 src=$PROC/$3 model=$4; shift 4
  mkdir -p "$out" && cp "$src/input.json" "$out/input.json"
  for k in 0 1 2 3 4; do
    .venv-gpu/Scripts/python.exe scripts/finetune_ce.py --data "$out/input.json" --fold "$k" --out-dir "$out" \
      --model "$model" --epochs 2 --lr 2e-5 --batch-size 16 --grad-accum 2 --predict-batch-size 64 --precision bf16 "$@" 2>&1 \
      | grep --line-buffered -E "^(fold|device|trainable|predict)" >> "output/${exp}_finetune_log.txt"
    [ -f "$out/oof_fold$k.npy" ] || { log "$exp finetune failed (fold $k)"; return 1; }
  done
  log "$exp finetune done"
}

# features_lgb <exp>
features_lgb() {
  python -B exp/exp031_build_features.py "$1" >> "$LOG" 2>&1 || { log "$1 build features failed"; return 1; }
  python -B exp/exp031_run.py "$1" >> "$LOG" 2>&1 || { log "$1 lgb failed"; return 1; }
  log "$1 done"
}

finetune exp031 ft_ce_gte_modernbert_title_desc ft_ce_bge_base_title_desc Alibaba-NLP/gte-reranker-modernbert-base \
  --max-length 256 && features_lgb exp031
finetune exp032 ft_ce_qwen3_06b_title ft_ce_bge_base_title tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
  --max-length 160 --template qwen3 --freeze-embeddings --optim adamw_bnb_8bit && features_lgb exp032
if grep -q "exp031 finetune done" "$LOG" && grep -q "exp032 finetune done" "$LOG" && grep -q "pipeline done" output/exp030_pipeline_log.txt; then
  features_lgb exp033
fi
log "pipeline done"
