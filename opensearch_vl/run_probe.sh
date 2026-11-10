#!/usr/bin/env bash
# Probe rollouts of the SFT model on RL candidate questions: K samples per question,
# vLLM token server (GPU 0) + local summariser (GPU 1) + paid search with a shared cache,
# then strict and lenient judging and probe_summary.py.
#
#   bash opensearch_vl/run_probe.sh <candidates.parquet> <work_dir> [K=4] [workers_per_sample=8]
#
# Env: PROBE_MODEL (merged SFT checkpoint), PROBE_TEMPERATURE (1.0), AGENT_MAX_TURNS (15),
#      MAX_NEW_TOKENS (4096), MIN_DEEPSEEK_CNY (balance needed to start).
# Re-running the same command resumes: finished trajectories are skipped, the cache is reused.
set -uo pipefail
CANDS=${1:?candidates parquet}; WORK=${2:?work dir}; K=${3:-4}; W=${4:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
SFT_ENV=/root/miniconda3/envs/sft
VLLM_PY=/root/venvs/vllm/bin/python
MODEL=${PROBE_MODEL:-/root/models/qwen3vl4b-sft-merged}
TEMP=${PROBE_TEMPERATURE:-1.0}
export PATH="$SFT_ENV/bin:/root/miniconda3/bin:$PATH"
export TOKENIZERS_PARALLELISM=false AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-15}
export QWEN3VL_4B_PATH=$MODEL VLLM_TOKEN_URL=http://127.0.0.1:8001
export SAMPLING_TOP_P=${SAMPLING_TOP_P:-1.0} SAMPLING_TOP_K=${SAMPLING_TOP_K:--1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
SECRETS=/root/.imagentrl-secrets.env
[ -f "$SECRETS" ] || { echo "ERROR: $SECRETS missing"; exit 1; }
set -a; . "$SECRETS"; set +a
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
mkdir -p "$WORK"
export TOOL_CACHE_DB=$WORK/tool_cache.sqlite

python - <<'PY' || exit 1
import os, requests
need = float(os.environ.get("MIN_DEEPSEEK_CNY", "0"))
r = requests.get("https://api.deepseek.com/user/balance", timeout=30,
                 headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
r.raise_for_status()
total = sum(float(b["total_balance"]) for b in r.json()["balance_infos"])
print(f"DeepSeek balance: {total:.2f} (required {need:.2f})")
if total < need:
    raise SystemExit("ERROR: top up DeepSeek before running the probe")
PY

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
up() { curl -s -m 5 -o /dev/null -f "$1"; }
start() {  # name url cmd...
  local name=$1 url=$2; shift 2
  if ! up "$url"; then
    "$@" > "$OUTPUT_ROOT/$name.log" 2>&1 &
    PIDS+=($!)
  fi
}
start ocr_service http://127.0.0.1:8080/docs bash "$HERE/start_ocr_service.sh"
start summary_service http://127.0.0.1:8000/v1/models bash "$HERE/start_summary_service.sh"
start vllm_policy http://127.0.0.1:8001/health env CUDA_VISIBLE_DEVICES=0 "$VLLM_PY" "$HERE/vllm_token_server.py" \
  --model "$MODEL" --port 8001 --gpu-memory-utilization 0.75 --max-num-seqs 64
for svc in http://127.0.0.1:8080/docs http://127.0.0.1:8000/v1/models http://127.0.0.1:8001/health; do
  for _ in $(seq 1 180); do up "$svc" && break; sleep 10; done
  up "$svc" || { echo "ERROR: $svc did not come up"; tail -20 "$OUTPUT_ROOT"/*.log; exit 1; }
done
export QWEN_API_BASE=http://127.0.0.1:8000/v1 QWEN_MODEL=summariser QWEN_MODEL_NAME=summariser QWEN_API_KEY=
export SUMMARY_EXTRA_BODY_JSON='{"top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}'
echo "[$(date +%T)] services ready"

pids=()
for k in $(seq 0 $((K - 1))); do
  mkdir -p "$WORK/s$k"
  python "$HERE/run_infer.py" --model qwen3-vl-4b-vllm --dataset train --data-path "$CANDS" \
    --output-dir "$WORK/s$k" --skip-existing --workers "$W" --temperature "$TEMP" \
    --max-tokens "$MAX_NEW_TOKENS" --log-level WARNING > "$OUTPUT_ROOT/probe_s$k.log" 2>&1 &
  pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "[$(date +%T)] rollouts done ($failed processes failed)"
if [ "$failed" -ne 0 ]; then
  grep -h "Stopping:" "$OUTPUT_ROOT"/probe_s*.log | sort -u | head -5
  echo "ERROR: rollouts incomplete; re-run the same command to resume"
  exit 1
fi

for k in $(seq 0 $((K - 1))); do
  python "$HERE/eval_with_gpt4o.py" --traj_dir "$WORK/s$k" --benchmark probe_strict --strict --max_workers 16 \
    2>&1 | grep -vE "it/s|s/it" | tail -2
  python "$HERE/eval_with_gpt4o.py" --traj_dir "$WORK/s$k" --benchmark probe_lenient --max_workers 16 \
    2>&1 | grep -vE "it/s|s/it" | tail -2
done
python "$HERE/probe_summary.py" --cands "$CANDS" --work "$WORK" --k "$K" --out "$OUTPUT_ROOT/probe" || exit 1
mkdir -p "$OUTPUT_ROOT/trajectories"
for k in $(seq 0 $((K - 1))); do
  mkdir -p "$OUTPUT_ROOT/trajectories/s$k"
  cp "$WORK/s$k"/*_trajectory.json "$WORK/s$k"/eval_report_* "$OUTPUT_ROOT/trajectories/s$k/" 2>/dev/null || true
done
echo PROBE_OK
