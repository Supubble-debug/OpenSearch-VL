#!/usr/bin/env bash
# SFT v1 vs v2 comparison (2026-11-11):
#   1. merge the v2 LoRA (v1 is already merged)
#   2. 27B judge+summariser on GPU 1 (images on, 32k context)
#   3. clean validation subset (two-pass vision judge over sft_val_200)
#   4. benchmark eval of v1 and v2 with the vLLM backend (paper7_n50, greedy), lenient + strict judge
#   5. strict judge on the earlier HF-backend base / v1 trajectories, summaries (lenient and strict)
#   6. val loss of every v1 / v2 checkpoint on the clean subset
# Each stage records its status; stage 6 runs even if the benchmark eval fails.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/miniconda3/envs/sft/bin:$PATH
: "${OUTPUT_ROOT:?}"
M1=/root/models/qwen3vl4b-sft-merged; M2=/root/models/qwen3vl4b-sft-v2-merged
CASES=/root/autodl-fs/datasets/imagentrl/eval/paper7_n50.parquet
HF=/root/sft-eval-q3vl/paper7_n50; W=/root/sft-eval-q3vl/vllm_n50
status=0
stop_servers() { pkill -f "[v]llm serve" 2>/dev/null; pkill -f "[v]llm_token_server" 2>/dev/null; sleep 15; }
trap stop_servers EXIT

echo "[$(date +%T)] 1/6 merge v2"
[ -f $M2/config.json ] || (cd SFT && llamafactory-cli export examples/imagentrl/qwen3_vl_4b_merge_v2.yaml) || { echo "ERROR: merge failed"; exit 1; }

echo "[$(date +%T)] 2/6 27B judge+summariser on GPU 1"
SUMMARY_GPU=1 SUMMARY_MAX_IMAGES=4 SUMMARY_MAX_LEN=32768 bash opensearch_vl/start_summary_service.sh \
  > $OUTPUT_ROOT/summary_service.log 2>&1 &
for i in $(seq 1 180); do curl -sf -m 5 -o /dev/null http://127.0.0.1:8000/v1/models && break; sleep 10; done
curl -sf -m 5 -o /dev/null http://127.0.0.1:8000/v1/models || { echo "ERROR: 27B not up"; tail -30 $OUTPUT_ROOT/summary_service.log; exit 1; }

echo "[$(date +%T)] 3/6 clean validation subset"
bash SFT/scripts/clean_val.sh || { echo "ERROR: clean_val failed"; status=1; }

echo "[$(date +%T)] 4/6 benchmark eval v1 + v2 (vLLM)"
MIN_DEEPSEEK_CNY=15 bash opensearch_vl/run_eval_vllm.sh $CASES $W v1=$M1 v2=$M2 || { echo "ERROR: eval failed"; status=1; }

echo "[$(date +%T)] 5/6 strict judge on HF trajectories + summaries"
for t in base sft; do
  ls $HF/$t/eval_report_vdr_strict_details.jsonl >/dev/null 2>&1 || \
    python opensearch_vl/eval_with_gpt4o.py --traj_dir $HF/$t --benchmark vdr_strict --strict --max_workers 8 2>&1 | grep -vE "it/s|s/it" | tail -2
done
if [ -d $W/v2 ]; then
  for rep in vdr vdr_strict; do
    python opensearch_vl/summarize_sft_eval.py --report $rep --run base_hf=$HF/base --run v1_hf=$HF/sft \
      --run v1=$W/v1 --run v2=$W/v2 --baseline base_hf --baseline v1 --out $OUTPUT_ROOT/summary_$rep || status=1
  done
fi
stop_servers

echo "[$(date +%T)] 6/6 val loss on the clean subset"
if [ -f /root/datasets/sft_val_200_clean/dataset_info.json ]; then
  for v in v1 v2; do
    R=/root/sft-runs/qwen3_vl_4b_lora_sft$([ $v = v2 ] && echo _v2)
    python SFT/scripts/eval_checkpoints.py --model /root/models/Qwen3-VL-4B-Instruct --ckpt-root $R \
      --dataset-dir /root/datasets/sft_val_200_clean --dataset-name imagentrl_sft_val200_clean --template qwen3_vl \
      --lanes 2 --work /root/sft-eval-q3vl/valclean_$v --best-file /root/sft-eval-q3vl/valclean_${v}_best.txt || status=1
    cp $OUTPUT_ROOT/val_losses.json $OUTPUT_ROOT/valclean_losses_$v.json 2>/dev/null || true
  done
else
  echo "ERROR: no clean val subset"; status=1
fi
[ $status -eq 0 ] && echo COMPARE_OK
exit $status
