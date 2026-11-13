#!/usr/bin/env bash
# Agentic GRPO/RLOO with LoRA (rl/train_grpo.py) on 2 GPUs:
#   GPU 0: vLLM token server (policy, per-request LoRA) + the trainer (HF + PEFT)
#   GPU 1: local 27B summariser for search results
# DeepSeek: text search + strict judge; Doubao: image search.
#   bash opensearch_vl/run_rl.sh <run_name> [extra train_grpo.py args...]
# Env: RL_MODEL, RL_TRAIN, RL_DEV, VLLM_MEM (0.40), MIN_DEEPSEEK_CNY (30). Re-running resumes.
# RL_ENTRY: another script taking the same --model/--train/--dev/--run-dir/--metrics/--max-turns
# (e.g. rl/seen_unseen_check.py) run against the same services.
set -uo pipefail
NAME=${1:?run name}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
SFT_ENV=/root/miniconda3/envs/sft; VLLM_PY=/root/venvs/vllm/bin/python
MODEL=${RL_MODEL:-/root/models/qwen3vl4b-sft-v2-merged}
TRAIN=${RL_TRAIN:-/root/rl_data/rl_train.parquet}; DEV=${RL_DEV:-/root/rl_data/rl_dev.parquet}
RUN=/root/rl_runs/$NAME
export PATH="$SFT_ENV/bin:/root/miniconda3/bin:$PATH"
export TOKENIZERS_PARALLELISM=false AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-10}
export QWEN3VL_4B_PATH=$MODEL VLLM_TOKEN_URL=http://127.0.0.1:8001 SAMPLING_TOP_P=1.0 SAMPLING_TOP_K=-1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SECRETS=/root/.imagentrl-secrets.env
[ -f "$SECRETS" ] || { echo "ERROR: $SECRETS missing"; exit 1; }
set -a; . "$SECRETS"; set +a
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
mkdir -p "$RUN"
export TOOL_CACHE_DB=$RUN/tool_cache.sqlite

python - <<'PY' || exit 1
import os, requests
need = float(os.environ.get("MIN_DEEPSEEK_CNY", "30"))
r = requests.get("https://api.deepseek.com/user/balance", timeout=30,
                 headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
r.raise_for_status()
total = sum(float(b["total_balance"]) for b in r.json()["balance_infos"])
print(f"DeepSeek balance: {total:.2f} (required {need:.2f})")
if total < need:
    raise SystemExit("ERROR: top up DeepSeek before RL")
PY

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 5; pkill -f "[v]llm_token_server" 2>/dev/null; pkill -f "[v]llm serve" 2>/dev/null; }
trap cleanup EXIT
up() { curl -s -m 5 -o /dev/null -f "$1"; }
start() { local name=$1 url=$2; shift 2; up "$url" || { "$@" > "$OUTPUT_ROOT/$name.log" 2>&1 & PIDS+=($!); }; }
start ocr_service http://127.0.0.1:8080/docs bash "$HERE/start_ocr_service.sh"
start summary_service http://127.0.0.1:8000/v1/models env SUMMARY_GPU=1 bash "$HERE/start_summary_service.sh"
start vllm_policy http://127.0.0.1:8001/health env CUDA_VISIBLE_DEVICES=0 "$VLLM_PY" "$HERE/vllm_token_server.py" \
  --model "$MODEL" --port 8001 --gpu-memory-utilization ${VLLM_MEM:-0.40} --max-num-seqs 64 --enable-lora --max-lora-rank 32
for svc in http://127.0.0.1:8080/docs http://127.0.0.1:8000/v1/models http://127.0.0.1:8001/health; do
  for _ in $(seq 1 180); do up "$svc" && break; sleep 10; done
  up "$svc" || { echo "ERROR: $svc did not come up"; tail -20 "$OUTPUT_ROOT"/*.log; exit 1; }
done
# Summariser = local 27B (never DeepSeek: one call per fetched page would multiply the bill).
export QWEN_API_BASE=http://127.0.0.1:8000/v1 QWEN_MODEL=summariser QWEN_MODEL_NAME=summariser QWEN_API_KEY=
export SUMMARY_EXTRA_BODY_JSON='{"top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}'
case "$QWEN_API_BASE" in http://127.0.0.1*) ;; *) echo "ERROR: summariser must be local"; exit 1;; esac
echo "[$(date +%T)] services ready"

cd "$HERE"
CUDA_VISIBLE_DEVICES=0 python "${RL_ENTRY:-rl/train_grpo.py}" --model "$MODEL" --train "$TRAIN" --dev "$DEV" \
  --run-dir "$RUN" --metrics "$OUTPUT_ROOT/metrics.jsonl" --max-turns "$AGENT_MAX_TURNS" "$@"
rc=$?
mkdir -p "$OUTPUT_ROOT/adapter"
cp -r "$RUN/adapters/final/." "$OUTPUT_ROOT/adapter/" 2>/dev/null || true
cp "$RUN/state.json" "$OUTPUT_ROOT/" 2>/dev/null || true
exit $rc
