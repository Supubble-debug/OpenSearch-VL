#!/usr/bin/env bash
# Teacher trajectories (SFT data / teacher trials) with an OpenAI-compatible vision API
# (default: official DeepSeek, model deepseek-teacher in the registry), in the same tool
# environment as the student: local OCR (:8080) + local 27B summariser (:8000), DeepSeek
# text search, Doubao image search. Inference only; judge separately.
#   bash opensearch_vl/run_teacher.sh <cases.parquet> <work_dir> [tag]
# Env: TEACHER_MODEL, TEACHER_THINKING, TEMPERATURE (0.6), WORKERS (8), AGENT_MAX_TURNS (15),
#      MIN_DEEPSEEK_CNY (5), TOOL_CACHE_DB (default <work_dir>/tool_cache.sqlite).
set -uo pipefail
CASES=${1:?cases parquet}; WORK=${2:?work dir}; TAG=${3:-teacher}
HERE=$(cd "$(dirname "$0")" && pwd)
SFT_ENV=/root/miniconda3/envs/sft
export PATH="$SFT_ENV/bin:/root/miniconda3/bin:$PATH" TOKENIZERS_PARALLELISM=false
export AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-15}
SECRETS=/root/.imagentrl-secrets.env
[ -f "$SECRETS" ] || { echo "ERROR: $SECRETS missing"; exit 1; }
set -a; . "$SECRETS"; set +a
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
mkdir -p "$WORK/$TAG"
export TOOL_CACHE_DB=${TOOL_CACHE_DB:-$WORK/tool_cache.sqlite}
balance() { python - <<'PY'
import os, requests
r = requests.get("https://api.deepseek.com/user/balance", timeout=30,
                 headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
r.raise_for_status()
print(f"{sum(float(b['total_balance']) for b in r.json()['balance_infos']):.2f}")
PY
}
B0=$(balance) || exit 1
echo "DeepSeek balance: $B0 (required ${MIN_DEEPSEEK_CNY:-5})"
python -c "import sys; sys.exit(0 if $B0 >= ${MIN_DEEPSEEK_CNY:-5} else 1)" || { echo "ERROR: top up DeepSeek"; exit 1; }
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 5; pkill -f "[v]llm serve" 2>/dev/null; }
trap cleanup EXIT
up() { curl -s -m 5 -o /dev/null -f "$1"; }
up http://127.0.0.1:8080/docs || { bash "$HERE/start_ocr_service.sh" > "$OUTPUT_ROOT/ocr_service.log" 2>&1 & PIDS+=($!); }
up http://127.0.0.1:8000/v1/models || { SUMMARY_GPU=${SUMMARY_GPU:-1} bash "$HERE/start_summary_service.sh" > "$OUTPUT_ROOT/summary_service.log" 2>&1 & PIDS+=($!); }
for svc in http://127.0.0.1:8080/docs http://127.0.0.1:8000/v1/models; do
  for _ in $(seq 1 180); do up "$svc" && break; sleep 10; done
  up "$svc" || { echo "ERROR: $svc did not come up"; exit 1; }
done
export QWEN_API_BASE=http://127.0.0.1:8000/v1 QWEN_MODEL=summariser QWEN_MODEL_NAME=summariser QWEN_API_KEY=
export SUMMARY_EXTRA_BODY_JSON='{"top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}'
echo "[$(date +%T)] services ready; teacher ${TEACHER_MODEL:-deepseek-flash} thinking=${TEACHER_THINKING:-0}"
python "$HERE/run_infer.py" --model deepseek-teacher --dataset test --data-path "$CASES" --output-dir "$WORK/$TAG" \
  --skip-existing --workers "${WORKERS:-8}" --temperature "${TEMPERATURE:-0.6}" --max-tokens 8192 --log-level WARNING \
  > "$OUTPUT_ROOT/infer_$TAG.log" 2>&1
rc=$?
B1=$(balance)
python - "$WORK/$TAG" "$B0" "$B1" <<'PY'
import collections, glob, json, sys
d, b0, b1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
stops, tools, tok = collections.Counter(), collections.Counter(), collections.Counter()
n = 0
for p in glob.glob(f"{d}/*_trajectory.json"):
    t = json.load(open(p)); n += 1; stops[t.get("stop_reason")] += 1
    for x in t.get("turns", []):
        u = (x.get("response") or {}).get("usageMetadata") or {}
        for k in ("promptTokenCount", "candidatesTokenCount", "cachedTokenCount"): tok[k] += u.get(k, 0) or 0
        if x.get("tool_call"):
            try: tools[json.loads(x["tool_call"]).get("name")] += 1
            except Exception: tools["?"] += 1
print(f"{n} trajectories, stops {dict(stops)}")
print(f"tool calls {dict(tools)} ({sum(tools.values())/max(n,1):.2f}/traj)")
print(f"teacher tokens {dict(tok)}")
print(f"DeepSeek spent {b0-b1:.2f} CNY ({(b0-b1)/max(n,1):.4f}/traj), balance {b1:.2f}")
PY
mkdir -p "$OUTPUT_ROOT/trajectories/$TAG"; cp "$WORK/$TAG"/*_trajectory.json "$OUTPUT_ROOT/trajectories/$TAG/" 2>/dev/null
echo "TEACHER_DONE rc=$rc"
exit $rc
