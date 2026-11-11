#!/usr/bin/env bash
# Clean validation subset: run the two-pass vision answer judge (no-think, then thinking on
# survivors) over sft_val_200 and keep only samples neither pass flags. The unfiltered set
# holds ~40% bad targets, which hides whether cleaner SFT data fits *good* targets better.
#   JUDGE_API=http://127.0.0.1:8000/v1 bash SFT/scripts/clean_val.sh
# Needs a judge server started with SUMMARY_MAX_IMAGES>=4 SUMMARY_MAX_LEN=32768.
set -euo pipefail
PY=/root/miniconda3/envs/sft/bin/python; S=SFT/scripts; M=/root/models/Qwen3-VL-4B-Instruct
D=/root/datasets/Search-VL-SFT-36K; V=/root/datasets/sft_val_200; O=/root/datasets/sft_val_200_clean
A=/root/sft-audit/val_clean; API=${JUDGE_API:-http://127.0.0.1:8000/v1}
mkdir -p $A
$PY $S/check_sft_answers.py --data-root $D --tokenizer $M --manifest $V/manifest.jsonl --images 4 \
  --workers 32 --api $API --out $A/pass1
$PY - $V/manifest.jsonl $A/pass1/bad_ids.txt $A/pass1_keep.jsonl <<'PY'
import json, sys
man, bad, out = sys.argv[1:]
bad = set(open(bad).read().split())
rows = [l for l in open(man) if json.loads(l)["id"] not in bad]
open(out, "w").writelines(rows)
print(f"pass1 flagged {len(bad)}, kept {len(rows)}")
PY
$PY $S/check_sft_answers.py --data-root $D --tokenizer $M --manifest $A/pass1_keep.jsonl --images 4 --thinking \
  --workers 32 --api $API --out $A/pass2
$PY - $V $O $A/pass1/bad_ids.txt $A/pass2/bad_ids.txt <<'PY'
import json, os, sys
V, O, b1, b2 = sys.argv[1:]
bad = set(open(b1).read().split()) | set(open(b2).read().split())
man = [json.loads(l) for l in open(f"{V}/manifest.jsonl")]
data = json.load(open(f"{V}/sft_subset.json", encoding="utf-8"))
assert len(man) == len(data), (len(man), len(data))
keep = [i for i, m in enumerate(man) if m["id"] not in bad]
os.makedirs(O, exist_ok=True)
json.dump([data[i] for i in keep], open(f"{O}/sft_subset.json", "w", encoding="utf-8"), ensure_ascii=False)
with open(f"{O}/manifest.jsonl", "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(man[i], ensure_ascii=False) + "\n" for i in keep)
info = json.load(open(f"{V}/dataset_info.json"))
spec = next(iter(info.values()))
json.dump({"imagentrl_sft_val200_clean": spec}, open(f"{O}/dataset_info.json", "w"), indent=1)
print(f"val clean: {len(keep)} of {len(man)} kept ({len(bad)} flagged)")
PY
echo CLEAN_VAL_OK
