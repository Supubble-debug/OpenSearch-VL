#!/usr/bin/env bash
# Benchmark eval with the vLLM token backend (prompt = LLaMA-Factory template, token for token):
# one policy server per model on GPU 0, local 27B summariser on GPU 1 (reused if already up on
# :8000), DeepSeek judge in lenient ("vdr") and strict ("vdr_strict") mode. Greedy decoding.
#   bash opensearch_vl/run_eval_vllm.sh <cases.parquet> <work_dir> tag1=/merged/model1 [tag2=/merged/model2 ...]
# A spec tag=/model+/lora_dir serves the RL LoRA unmerged (vLLM --enable-lora, per-request adapter):
# the policy exactly as trained, without bf16 merge rounding.
# Re-running resumes (finished trajectories are skipped; search results come from the shared cache).
set -uo pipefail
CASES=${1:?cases parquet}; WORK=${2:?work dir}; shift 2
[ $# -ge 1 ] || { echo "usage: $0 cases work tag=model ..."; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
SFT_ENV=/root/miniconda3/envs/sft; VLLM_PY=/root/venvs/vllm/bin/python
export PATH="$SFT_ENV/bin:/root/miniconda3/bin:$PATH"
export TOKENIZERS_PARALLELISM=false AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-15}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}; WORKERS=${WORKERS:-8}
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
    raise SystemExit("ERROR: top up DeepSeek before running this evaluation")
PY

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
up() { curl -s -m 5 -o /dev/null -f "$1"; }
if ! curl -s -m 5 -o /dev/null http://127.0.0.1:8080/docs; then
  bash "$HERE/start_ocr_service.sh" > "$OUTPUT_ROOT/ocr_service.log" 2>&1 & PIDS+=($!)
fi
up http://127.0.0.1:8000/v1/models || { bash "$HERE/start_summary_service.sh" > "$OUTPUT_ROOT/summary_service.log" 2>&1 & PIDS+=($!); }

TAGS=(); MODELS=(); PORTS=(); LORAS=()
n=$#; MEM=$(python -c "print(round(0.84 / $n, 2))")
i=0
for spec in "$@"; do
  tag=${spec%%=*}; model=${spec#*=}; port=$((8001 + 2 * i)); lora=""
  case "$model" in *+*) lora=${model#*+}; model=${model%%+*};; esac
  [ -f "$model/config.json" ] || { echo "ERROR: $model is not a model dir"; exit 1; }
  [ -z "$lora" ] || [ -f "$lora/adapter_config.json" ] || { echo "ERROR: $lora is not a LoRA dir"; exit 1; }
  TAGS+=("$tag"); MODELS+=("$model"); PORTS+=("$port"); LORAS+=("$lora")
  CUDA_VISIBLE_DEVICES=0 "$VLLM_PY" "$HERE/vllm_token_server.py" --model "$model" --port "$port" \
    --gpu-memory-utilization "$MEM" --max-num-seqs 32 ${lora:+--enable-lora --max-lora-rank 32} \
    > "$OUTPUT_ROOT/vllm_policy_$tag.log" 2>&1 &
  PIDS+=($!); i=$((i + 1))
  sleep 60  # stagger engine start-up (memory profiling) on the shared GPU
done
SVCS=(http://127.0.0.1:8080/docs http://127.0.0.1:8000/v1/models)
for p in "${PORTS[@]}"; do SVCS+=("http://127.0.0.1:$p/health"); done
for svc in "${SVCS[@]}"; do
  for _ in $(seq 1 180); do curl -s -m 5 -o /dev/null "$svc" && break; sleep 10; done
  curl -s -m 5 -o /dev/null "$svc" || { echo "ERROR: $svc did not come up"; tail -20 "$OUTPUT_ROOT"/*.log; exit 1; }
done
export QWEN_API_BASE=http://127.0.0.1:8000/v1 QWEN_MODEL=summariser QWEN_MODEL_NAME=summariser QWEN_API_KEY=
export SUMMARY_EXTRA_BODY_JSON='{"top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}'
echo "[$(date +%T)] services ready: ${TAGS[*]}"

pids=()
for j in "${!TAGS[@]}"; do
  tag=${TAGS[$j]}; mkdir -p "$WORK/$tag"
  VLLM_TOKEN_URL=http://127.0.0.1:${PORTS[$j]} VLLM_LORA_PATH=${LORAS[$j]} python "$HERE/run_infer.py" --model qwen3-vl-4b-vllm \
    --checkpoint "${MODELS[$j]}" --dataset test --data-path "$CASES" --output-dir "$WORK/$tag" --skip-existing \
    --workers "$WORKERS" --temperature 0 --max-tokens "$MAX_NEW_TOKENS" --log-level WARNING \
    > "$OUTPUT_ROOT/infer_$tag.log" 2>&1 &
  pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "[$(date +%T)] inference done ($failed processes failed)"
if [ "$failed" -ne 0 ]; then
  grep -h "Stopping:" "$OUTPUT_ROOT"/infer_*.log | sort -u | head -5
  echo "ERROR: inference incomplete, not judging. Re-run the same command to resume."
  exit 1
fi

python - "$WORK" "${TAGS[@]}" <<'PY' || exit 1
import collections, glob, json, os, sys
work, tags = sys.argv[1], sys.argv[2:]
bad = False
for tag in tags:
    stops = collections.Counter(json.load(open(p, encoding="utf-8")).get("stop_reason", "?")
                                for p in glob.glob(os.path.join(work, tag, "*_trajectory.json")))
    total, broken = sum(stops.values()), stops["runner_error"] + stops["runner_exception"]
    print(f"{tag}: {total} trajectories, stop reasons {dict(stops)}")
    if total == 0 or broken / total > 0.2:
        bad = True
        print(f"ERROR: {tag}: {broken}/{total} cases hit runner failures")
sys.exit(1 if bad else 0)
PY

for tag in "${TAGS[@]}"; do
  python "$HERE/eval_with_gpt4o.py" --traj_dir "$WORK/$tag" --benchmark vdr --max_workers 8 2>&1 | grep -vE "it/s|s/it" | tail -3
  python "$HERE/eval_with_gpt4o.py" --traj_dir "$WORK/$tag" --benchmark vdr_strict --strict --max_workers 8 2>&1 | grep -vE "it/s|s/it" | tail -3
  mkdir -p "$OUTPUT_ROOT/trajectories/$tag"
  cp "$WORK/$tag"/*_trajectory.json "$WORK/$tag"/eval_report_* "$OUTPUT_ROOT/trajectories/$tag/" 2>/dev/null || true
done
echo EVAL_VLLM_OK
