#!/usr/bin/env bash
# SFT v2 data: two-pass vision answer filter on both GPUs, then the final 4096 selection.
#   bash SFT/scripts/filter_sft_v2.sh [n_candidates=10000]
# Pass 1: 27B judge with images, no thinking, on all candidates.
# Pass 2: same judge WITH thinking, only on pass-1 survivors. A sample is dropped if either pass flags it.
# Measured on 100 reference-labelled samples (2026-11-10): pass1 recall 58%, pass1 OR thinking 76%,
# ~52% of candidates flagged -> 10000 candidates leave ~4800 for the 4096 selection.
# One judge server per GPU (ports 8000/8002); each pass splits its manifest in two halves.
set -euo pipefail
N=${1:-10000}
PY=/root/miniconda3/envs/sft/bin/python; S=SFT/scripts; M=/root/models/Qwen3-VL-4B-Instruct
D=/root/datasets/Search-VL-SFT-36K; E=/root/datasets/vdr-eval/data; A=/root/sft-audit/v2
: "${OUTPUT_ROOT:?}"
mkdir -p $A
EVAL="$E/MMSearch-Plus_221.parquet $E/bcvl_level1_199.parquet $E/bcvl_level2_200.parquet $E/fvqa_300.parquet $E/infoseek_300.parquet $E/livevqa_webwatcher_subset_300.parquet $E/mmsearch_vision_171.parquet $E/simplevqa_300.parquet $E/testmini500_v1.parquet $E/worldvqa_sampled_300.parquet /root/datasets/fvqa/fvqa_test.parquet"
COMMON="--data-root $D --model $M --eval $EVAL --exclude-manifest /root/datasets/sft_val_200/manifest.jsonl --ratios retrieval=0.6,crop_ocr=0.3,enhancement=0.1 --cutoff-len 32000"

PIDS=()
trap 'for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; sleep 5; pkill -f "vllm serve" 2>/dev/null || true' EXIT
for g in 0 1; do
  SUMMARY_GPU=$g SUMMARY_PORT=$((8000 + 2 * (1 - g))) SUMMARY_MAX_IMAGES=4 SUMMARY_MAX_LEN=32768 \
    bash opensearch_vl/start_summary_service.sh > $OUTPUT_ROOT/vllm_gpu$g.log 2>&1 &
  PIDS+=($!)
done

C=$A/cand$N
echo "[$(date +%T)] 0/4 candidate selection ($N)"
[ -f $C/manifest.jsonl ] || $PY $S/select_sft_subset.py $COMMON --n $N --out $C --dataset-name imagentrl_cand_v2
cp $C/report.json $OUTPUT_ROOT/cand_selection_report.json

for port in 8000 8002; do
  for i in $(seq 1 180); do curl -sf -m 5 -o /dev/null http://127.0.0.1:$port/v1/models && break; sleep 10; done
  curl -sf -m 5 -o /dev/null http://127.0.0.1:$port/v1/models || { echo "ERROR: judge on $port not up"; tail -30 $OUTPUT_ROOT/vllm_gpu*.log; exit 1; }
done
echo "[$(date +%T)] judges up on both GPUs"

# judge <manifest> <out_prefix> [extra args] -> writes <out_prefix>_bad.txt
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
  grep -h '"judge_failed"' ${out}_h0/summary.json ${out}_h1/summary.json || true
}
keep() {  # keep <manifest> <bad.txt> <out.jsonl>
  $PY - "$1" "$2" "$3" <<'PYX'
import json, sys
man, bad, out = sys.argv[1:]
bad = set(open(bad).read().split())
rows = [l for l in open(man) if json.loads(l)["id"] not in bad]
open(out, "w").writelines(rows)
print(f"flagged {len(bad)}, kept {len(rows)}")
PYX
}

echo "[$(date +%T)] 1/4 pass 1: vision judge (no thinking) on $N"
judge $C/manifest.jsonl $A/pass1
keep $C/manifest.jsonl $A/pass1_bad.txt $A/pass1_keep.jsonl
echo "[$(date +%T)] 2/4 pass 2: vision judge WITH thinking on pass-1 survivors"
judge $A/pass1_keep.jsonl $A/pass2 --thinking
keep $A/pass1_keep.jsonl $A/pass2_bad.txt $A/final_keep.jsonl
cat $A/pass1_bad.txt $A/pass2_bad.txt > $OUTPUT_ROOT/v2_bad_ids.txt
$PY -c "import json; ids=[json.loads(l)['id'] for l in open('$A/final_keep.jsonl')]; open('$A/keep_ids.txt','w').write(''.join(i+'\n' for i in ids)); print('keep',len(ids)); assert len(ids)>=4096, 'not enough clean candidates'"
echo "[$(date +%T)] 3/4 final selection (4096)"
$PY $S/select_sft_subset.py $COMMON --n 4096 --keep-ids $A/keep_ids.txt --out /root/datasets/sft_subset_q3vl_v2 --dataset-name imagentrl_sft_q3vl_v2
$PY -c "import json; r=json.load(open('/root/datasets/sft_subset_q3vl_v2/report.json')); n=r['selected_total']; print('selected',n,r['selected_by_category'],'shortfall',r['shortfall_vs_requested']); assert n==4096"
echo "[$(date +%T)] 4/4 done"
echo SELECTION_V2_OK
