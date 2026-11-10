#!/usr/bin/env bash
# ImagentRL SFT launcher for projtool `run_script` (single node; FORCE_TORCHRUN=1 for multi-GPU).
#
#   bash SFT/scripts/run_imagentrl_sft.sh SFT/examples/imagentrl/<config>.yaml
#
# - runs LLaMA-Factory in the `sft` conda env (see .proj-tool/env-setup.sh);
# - mirrors LLaMA-Factory's trainer_log.jsonl into $OUTPUT_ROOT/metrics.jsonl
#   (projtool per-step series contract) and final scalars into metrics.json;
# - copies the final LoRA adapter + run config into $OUTPUT_ROOT/adapter.
set -euo pipefail
CFG=${1:?usage: run_imagentrl_sft.sh <config.yaml>}
SFT_ENV=/root/miniconda3/envs/sft
export PATH="$SFT_ENV/bin:/root/miniconda3/bin:$PATH"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export WANDB_PROJECT=${WANDB_PROJECT:-imagentrl-sft}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export FORCE_TORCHRUN=${FORCE_TORCHRUN:-0}  # 1 = all visible GPUs via torchrun
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set (projtool run_script exports it)}"
# Credentials (WANDB_API_KEY, …) live outside the synced workspace, mode 600.
SECRETS=/root/.imagentrl-secrets.env
if [ -f "$SECRETS" ]; then set -a; . "$SECRETS"; set +a; else echo "WARN: $SECRETS missing"; fi
# api.wandb.ai answers 403 from AutoDL westd and wandb.init then kills training: log offline,
# `wandb sync` the run dir later from a machine that can reach wandb.
export WANDB_MODE=${WANDB_MODE:-offline}
# wandb's default dir is the cwd (/workspace/SFT) — a mutagen mirror that deletes
# remote-only files. Keep wandb runs (incl. offline ones for later `wandb sync`) in the run dir.
export WANDB_DIR=${WANDB_DIR:-$OUTPUT_ROOT}

REPO=$(cd "$(dirname "$0")/../.." && pwd)
CFG_ABS=$(readlink -f "$CFG")
OUT_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG_ABS")
cp "$CFG_ABS" "$OUTPUT_ROOT/config.yaml"

# Mirror trainer_log.jsonl → metrics.jsonl while training runs.
export_metrics() {
  python - "$OUT_DIR/trainer_log.jsonl" "$OUTPUT_ROOT" <<'PY'
import json, math, os, sys
src, root = sys.argv[1], sys.argv[2]
rid = os.environ.get("PROJTOOL_RUN_ID", "")
rows = [json.dumps({"__meta__": {"schema": 1, "run_id": rid}})]
last = {}
if os.path.exists(src):
    for line in open(src):
        try:
            r = json.loads(line)
        except Exception:
            continue
        step = r.get("current_steps")
        for k in ("loss", "lr", "grad_norm", "epoch"):
            v = r.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v) and isinstance(step, int):
                rows.append(json.dumps({"step": step, "name": f"train/{k}", "value": v}))
                last[k] = v
tmp = os.path.join(root, "metrics.jsonl.tmp")
open(tmp, "w").write("\n".join(rows) + "\n")
os.replace(tmp, os.path.join(root, "metrics.jsonl"))
PY
}
( while sleep 120; do export_metrics || true; done ) &
EXPORTER=$!
trap 'kill $EXPORTER 2>/dev/null || true' EXIT

cd "$REPO/SFT"
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name())"
# LLaMA-Factory only *warns* when Liger has no patch for the model; without the fused
# linear+CE the fp32 logits OOM on long samples. Fail before training instead.
python - "$CFG_ABS" <<'PY2'
import json, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
if cfg.get("enable_liger_kernel"):
    mt = json.load(open(os.path.join(cfg["model_name_or_path"], "config.json")))["model_type"]
    import liger_kernel.transformers as lk
    fn = f"apply_liger_kernel_to_{mt}"
    if not hasattr(lk, fn):
        sys.exit(f"ERROR: enable_liger_kernel but liger-kernel has no {fn}; upgrade liger-kernel")
    print(f"liger: {fn} available")
PY2
llamafactory-cli train "$CFG_ABS"

export_metrics
python - "$OUT_DIR" "$OUTPUT_ROOT" <<'PY'
import json, os, sys
out, root = sys.argv[1], sys.argv[2]
res = {}
for name in ("train_results.json", "all_results.json"):
    p = os.path.join(out, name)
    if os.path.exists(p):
        res.update(json.load(open(p)))
keep = {k: v for k, v in res.items() if isinstance(v, (int, float))}
json.dump(keep, open(os.path.join(root, "metrics.json"), "w"), indent=1)
print("metrics.json:", keep)
PY
mkdir -p "$OUTPUT_ROOT/adapter"
cp "$OUT_DIR"/adapter_config.json "$OUT_DIR"/adapter_model.safetensors "$OUTPUT_ROOT/adapter/" 2>/dev/null || true
cp "$OUT_DIR"/trainer_log.jsonl "$OUTPUT_ROOT/" 2>/dev/null || true
echo "SFT_DONE output_dir=$OUT_DIR"
