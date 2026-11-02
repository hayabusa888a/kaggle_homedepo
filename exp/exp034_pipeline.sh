#!/usr/bin/env bash
# exp034 のメンバーを全部学習する（独立したプロセスとして起動する）。MLP は GPU 用の仮想環境で実行する。
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=output/exp034_pipeline_log.txt
export PYTHONIOENCODING=utf-8
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
mkdir -p output/exp034
log "start"
python -B exp/exp034_members.py >> "$LOG" 2>&1 || log "members failed"
log "members done"
.venv-gpu/Scripts/python.exe -B exp/exp034_members.py --mlp >> "$LOG" 2>&1 || log "mlp failed"
log "pipeline done"
