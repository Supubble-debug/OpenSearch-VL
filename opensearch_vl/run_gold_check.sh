#!/usr/bin/env bash
# RL gold-answer audit on both GPUs: one 27B judge server per GPU (image + thinking),
# check_rl_gold.py sharded over them.
#   bash opensearch_vl/run_gold_check.sh <out_dir> <cands.parquet>...
set -euo pipefail
OUT=${1:?out dir}; shift
PY=/root/miniconda3/envs/sft/bin/python
: "${OUTPUT_ROOT:?}"
mkdir -p "$OUT"
PIDS=()
trap 'for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; sleep 5; pkill -f "[v]llm serve" 2>/dev/null || true' EXIT
for g in 0 1; do
  SUMMARY_GPU=$g SUMMARY_PORT=$((8000 + 2 * g)) SUMMARY_MAX_IMAGES=1 SUMMARY_MAX_LEN=32768 \
    bash opensearch_vl/start_summary_service.sh > $OUTPUT_ROOT/judge_gpu$g.log 2>&1 &
  PIDS+=($!)
done
for port in 8000 8002; do
  for i in $(seq 1 180); do curl -sf -m 5 -o /dev/null http://127.0.0.1:$port/v1/models && break; sleep 10; done
  curl -sf -m 5 -o /dev/null http://127.0.0.1:$port/v1/models || { echo "ERROR: judge on $port not up"; tail -30 $OUTPUT_ROOT/judge_gpu*.log; exit 1; }
done
echo "[$(date +%T)] judges up"
$PY opensearch_vl/check_rl_gold.py --cands "$@" --out "$OUT" --api http://127.0.0.1:8000/v1 --shard 0/2 > $OUTPUT_ROOT/gold_0.log 2>&1 &
p0=$!
$PY opensearch_vl/check_rl_gold.py --cands "$@" --out "$OUT" --api http://127.0.0.1:8002/v1 --shard 1/2 > $OUTPUT_ROOT/gold_1.log 2>&1 &
p1=$!
wait $p0 && wait $p1 || { tail -20 $OUTPUT_ROOT/gold_*.log; exit 1; }
cat "$OUT"/verdicts_*.jsonl > $OUTPUT_ROOT/gold_verdicts.jsonl
echo "[$(date +%T)] judged $(wc -l < $OUTPUT_ROOT/gold_verdicts.jsonl)"
echo GOLD_CHECK_OK
