#!/usr/bin/env bash
# exp025 を最後まで通して実行する（長時間かかるので、独立したプロセスとして起動する）。
#   1. v2-m3（タイトル + 説明文）のファインチューニングを、乱数の種だけ変えてもう一度行う（保存済みの fold は飛ばす）
#   2. 2 つの種の平均に置き換えた特徴量を作る
#   3. exp024 と同じメンバーを学習する（MLP は GPU 用の仮想環境）
#   4. 混ぜて提出ファイルを作る
# 進み具合は output/exp025_pipeline_log.txt に出す。
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=output/exp025_pipeline_log.txt
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

log "start"
bash exp/exp025_finetune_seed2.sh >> output/exp025_finetune_log.txt 2>&1 || { log "finetune failed"; exit 1; }
log "finetune done"
python -B exp/exp025_build_features.py >> "$LOG" 2>&1 || { log "build features failed"; exit 1; }
log "features done"
python -B exp/exp025_run.py members >> "$LOG" 2>&1 || { log "members failed"; exit 1; }
.venv-gpu/Scripts/python.exe -B exp/exp025_run.py mlp >> "$LOG" 2>&1 || { log "mlp failed"; exit 1; }
python -B exp/exp025_run.py blend >> "$LOG" 2>&1 || { log "blend failed"; exit 1; }
log "pipeline done"
