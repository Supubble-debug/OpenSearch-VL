#!/usr/bin/env bash
# SFT v3b data (2026-11-17): ~1000 SFT-36K trajectories (Claude teacher, released data) chosen for the
# student's gap behaviours, no new API spend. Comparison arm for v3a (our DeepSeek teacher).
#   1. never-audited trajectories with query rewriting (>=3 distinct searches) or >=5 calls (cap 400/source)
#   2. same structure/length/eval-overlap checks as v2 (select_sft_subset.py), held out from the v2 subset
#   3. two-pass 27B vision answer filter, as for v2 (check_sft_answers.py, no thinking, then thinking)
#   4. fill to 1000 with unused v2-audited trajectories ranked by behaviour score (select_gap_behaviors.py)
#   5. mix with the same 1000 replayed v2 samples as v3a (build_sft_mix.py, same seed)
set -euo pipefail
PY=/root/miniconda3/envs/sft/bin/python; S=SFT/scripts; M=/root/models/Qwen3-VL-4B-Instruct
D=/root/datasets/Search-VL-SFT-36K; E=/root/datasets/vdr-eval/data; A=/root/sft-audit/v3b
V2M=/root/datasets/sft_subset_q3vl_v2/manifest.jsonl
: "${OUTPUT_ROOT:?}"
mkdir -p $A
EVAL="$E/MMSearch-Plus_221.parquet $E/bcvl_level1_199.parquet $E/bcvl_level2_200.parquet $E/fvqa_300.parquet $E/infoseek_300.parquet $E/livevqa_webwatcher_subset_300.parquet $E/mmsearch_vision_171.parquet $E/simplevqa_300.parquet $E/testmini500_v1.parquet $E/worldvqa_sampled_300.parquet /root/datasets/fvqa/fvqa_test.parquet"
COMMON="--data-root $D --model $M --eval $EVAL --exclude-manifest $V2M --ratios retrieval=0.6,crop_ocr=0.3,enhancement=0.1 --cutoff-len 32000"

PIDS=()
trap 'for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; sleep 5; pkill -f "vllm serve" 2>/dev/null || true' EXIT
pkill -f "[v]llm serve" || true; pkill -f "[v]llm_token_server" || true; sleep 10
for g in 0 1; do
  SUMMARY_GPU=$g SUMMARY_PORT=$((8000 + 2 * (1 - g))) SUMMARY_MAX_IMAGES=4 SUMMARY_MAX_LEN=32768 \
    bash opensearch_vl/start_summary_service.sh > $OUTPUT_ROOT/vllm_gpu$g.log 2>&1 &
  PIDS+=($!)
done

echo "[$(date +%T)] 1/5 behaviour candidates (never audited)"
$PY $S/select_gap_behaviors.py --data-root $D --unaudited-from /root/sft-audit/v2/cand10000/manifest.jsonl \
  --exclude $V2M /root/datasets/sft_val_200/manifest.jsonl --cap-per-source 400 --out $A
echo "[$(date +%T)] 2/5 structure / length / eval-overlap checks"
$PY $S/select_sft_subset.py $COMMON --n 100000 --keep-ids $A/cand_ids.txt --out $A/cand --dataset-name imagentrl_v3b_cand
for port in 8000 8002; do
  for i in $(seq 1 180); do curl -sf -m 5 -o /dev/null http://127.0.0.1:$port/v1/models && break; sleep 10; done
  curl -sf -m 5 -o /dev/null http://127.0.0.1:$port/v1/models || { echo "ERROR: judge on $port not up"; exit 1; }
done
judge() {
  local man=$1 out=$2; shift 2
  awk 'NR%2==1' $man > ${out}_h0.jsonl; awk 'NR%2==0' $man > ${out}_h1.jsonl
  $PY $S/check_sft_answers.py --data-root $D --tokenizer $M --manifest ${out}_h0.jsonl --images 4 --workers 48 \
      --api http://127.0.0.1:8000/v1 --out ${out}_h0 "$@" > ${out}_h0.log 2>&1 &
  local p0=$!
  $PY $S/check_sft_answers.py --data-root $D --tokenizer $M --manifest ${out}_h1.jsonl --images 4 --workers 48 \
      --api http://127.0.0.1:8002/v1 --out ${out}_h1 "$@" > ${out}_h1.log 2>&1 &
  local p1=$!
  wait $p0 && wait $p1 || { tail -20 ${out}_h0.log ${out}_h1.log; return 1; }
  cat ${out}_h0/bad_ids.txt ${out}_h1/bad_ids.txt > ${out}_bad.txt
}
keep() { $PY - "$1" "$2" "$3" <<'PYX'
import json, sys
man, bad, out = sys.argv[1:]
bad = set(open(bad).read().split())
rows = [l for l in open(man) if json.loads(l)["id"] not in bad]
open(out, "w").writelines(rows)
print(f"flagged {len(bad)}, kept {len(rows)}")
PYX
}
echo "[$(date +%T)] 3/5 vision answer filter, pass 1 (no thinking) and pass 2 (thinking)"
judge $A/cand/manifest.jsonl $A/pass1
keep $A/cand/manifest.jsonl $A/pass1_bad.txt $A/pass1_keep.jsonl
judge $A/pass1_keep.jsonl $A/pass2 --thinking
keep $A/pass1_keep.jsonl $A/pass2_bad.txt $A/new_keep.jsonl
for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; pkill -f "[v]llm serve" || true
echo "[$(date +%T)] 4/5 fill to 1000 with unused v2-audited trajectories (behaviour-ranked)"
$PY $S/select_gap_behaviors.py --data-root $D --audited /root/sft-audit/v2/final_keep.jsonl \
  --exclude $V2M /root/datasets/sft_val_200/manifest.jsonl --n 1000 --out $A/leftover
$PY - <<PYX
import json
new = [json.loads(l)["id"] for l in open("$A/new_keep.jsonl")]
left = [l.strip() for l in open("$A/leftover/keep_ids.txt") if l.strip()]
ids = (new + [i for i in left if i not in set(new)])[:1000]
open("$A/final_ids.txt", "w").write("".join(i + "\n" for i in ids))
print(f"new-audited {len(new)} + leftover {len(ids) - min(len(new), 1000)} = {len(ids)}")
PYX
$PY $S/select_sft_subset.py $COMMON --n 1000 --keep-ids $A/final_ids.txt --out /root/datasets/sft_v3b_new --dataset-name imagentrl_v3b_new
echo "[$(date +%T)] 5/5 mix with replay"
$PY opensearch_vl/build_sft_mix.py --new /root/datasets/sft_v3b_new/sft_subset.json \
  --replay /root/datasets/sft_subset_q3vl_v2/sft_subset.json --n-replay 1000 --out /root/datasets/sft_v3b_mix --name imagentrl_sft_v3b
cp $A/cand_report.json /root/datasets/sft_v3b_new/report.json $OUTPUT_ROOT/ 2>/dev/null || true
echo BUILD_V3B_OK
