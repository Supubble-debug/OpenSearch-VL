#!/usr/bin/env python3
"""How far did an RL LoRA move the policy, and does vLLM apply that LoRA? (needs the instance)

1. Adapter size: per LoRA pair, ||scale * B @ A||_F / ||W||_F (relative weight change).
2. Policy shift on the *sampling* distribution: reconstruct trajectories that the base policy
   sampled (RL step-1 rollouts are pure SFT-v2 samples), render them exactly as the trainer does,
   and compare base vs base+adapter on every assistant token: mean KL(base || adapter) at the
   sampling temperature, mean |delta logp| of the sampled tokens, argmax flips, and the summed
   delta logp per trajectory. Trajectories are cut before the first tool that returns an image
   (crop / enhancement), because the saved json does not keep intermediate images.
3. vLLM LoRA parity (--vllm-url): an *amplified* copy of the adapter (lora_B * --amp) makes the
   LoRA effect large compared with bf16 noise, so "vLLM applies the adapter" becomes decidable:
   raw vLLM logprobs (pre-temperature) with that adapter must match HF + the same adapter at T=1
   much better than they match HF without it.

    python tests/rl_policy_shift.py --model /root/models/qwen3vl4b-sft-v2-merged \
        --adapter /root/rl_runs/r2/adapters/final --data /root/rl_data/rl_train.parquet \
        --rollouts /root/rl_runs/r2/rollouts/step_0001 --n 64 [--vllm-url http://127.0.0.1:8001]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rl.kl_probe import encode, rebuild  # noqa: E402  (same rendering as the trainer)


def adapter_stats(model, adapter_name: str) -> dict:
    import torch

    rel = []
    for _, mod in model.named_modules():
        if not hasattr(mod, "lora_A") or adapter_name not in getattr(mod, "lora_A", {}):
            continue
        a = mod.lora_A[adapter_name].weight.float()
        b = mod.lora_B[adapter_name].weight.float()
        w = mod.base_layer.weight.float()
        delta = (b @ a) * mod.scaling[adapter_name]
        rel.append(float(torch.linalg.norm(delta) / torch.linalg.norm(w)))
    rel.sort()
    return {"pairs": len(rel), "rel_median": rel[len(rel) // 2], "rel_max": rel[-1],
            "rel_mean": sum(rel) / len(rel)}

def logits_at(model, ids, mask, mm, chunk=2048):
    """Full-vocab logits (fp32) at the positions that predict masked tokens, plus the targets."""
    import torch

    inner = model.base_model.model if hasattr(model, "base_model") else model
    with torch.no_grad():
        h = inner.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **mm).last_hidden_state[0]
        tgt = mask[1:].nonzero(as_tuple=True)[0] + 1
        hs, ys = h[tgt - 1], ids[0, tgt]
        out = [inner.lm_head(hs[i:i + chunk]).float() for i in range(0, len(ys), chunk)]
    return torch.cat(out), ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True, help="dirs with s*/<id>_trajectory.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-seq-len", type=int, default=24000)
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--amp", type=float, default=20.0)
    ap.add_argument("--parity-n", type=int, default=8)
    ap.add_argument("--work", default="/root/rl_diag")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="float32 removes the bf16 rounding floor of base + tiny LoRA")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply lora_B (control: KL must scale ~scale^2)")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    from opensearch_infer import config
    from opensearch_infer.runners import Qwen35VLLMRunner

    dev = "cuda:0"
    rn = Qwen35VLLMRunner(config.MODEL_REGISTRY["qwen3-vl-4b-vllm"], args.model)
    rn.load_prompt_stack()
    dt = getattr(torch, args.dtype)
    adapter = args.adapter
    if args.scale != 1.0:
        from safetensors.torch import load_file, save_file

        adapter = os.path.join(args.work, f"scale{args.scale:g}_{os.path.basename(os.path.normpath(args.adapter))}")
        os.makedirs(adapter, exist_ok=True)
        shutil.copy(os.path.join(args.adapter, "adapter_config.json"), adapter)
        sd = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
        save_file({k: (v * args.scale if "lora_B" in k else v) for k, v in sd.items()},
                  os.path.join(adapter, "adapter_model.safetensors"))
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dt, device_map={"": dev}).eval()
    model = PeftModel.from_pretrained(base, adapter, adapter_name="rl").eval()
    model.to(dt)
    res: dict = {"adapter": args.adapter, "dtype": args.dtype, "scale": args.scale, "adapter_stats": adapter_stats(model, "rl")}
    print("adapter:", json.dumps(res["adapter_stats"]), flush=True)

    rows = {}
    for p in args.data:
        for r in pd.read_parquet(p).to_dict("records"):
            rows[r["data_id"]] = r
    files = sorted(f for d in args.rollouts for f in glob.glob(os.path.join(d, "s*", "*_trajectory.json")))
    random.Random(args.seed).shuffle(files)

    T = args.temperature
    kls, dlp_abs, flips, ntok, seq_dlp, used = [], [], 0, 0, [], 0
    for f in files:
        if used >= args.n:
            break
        traj = json.load(open(f, encoding="utf-8"))
        row = rows.get(traj.get("case_id"))
        if row is None:
            continue
        enc = encode(rn, rebuild(traj, row, rn.prompt_in_system), dev, dt)
        if enc is None or enc[0].shape[1] > args.max_seq_len:
            continue
        ids, mask, mm = enc
        with model.disable_adapter():
            lb, ys = logits_at(model, ids, mask, mm)
        la, _ = logits_at(model, ids, mask, mm)
        pb, pa = torch.log_softmax(lb / T, -1), torch.log_softmax(la / T, -1)
        kl = (pb.exp() * (pb - pa)).sum(-1)
        d = (pa - pb).gather(-1, ys[:, None]).squeeze(-1)
        kls.append(float(kl.mean())); dlp_abs.append(float(d.abs().mean()))
        flips += int((lb.argmax(-1) != la.argmax(-1)).sum()); ntok += len(ys)
        seq_dlp.append(float(d.sum()))
        used += 1
        del lb, la, pb, pa, kl, d
        torch.cuda.empty_cache()
    res["shift"] = {
        "trajectories": used, "tokens": ntok, "temperature": T,
        "kl_per_token_mean": statistics.mean(kls), "kl_per_token_max_traj": max(kls),
        "abs_dlogp_sampled_mean": statistics.mean(dlp_abs),
        "argmax_flip_rate": flips / max(1, ntok),
        "seq_dlogp_mean": statistics.mean(seq_dlp), "seq_dlogp_abs_mean": statistics.mean(abs(x) for x in seq_dlp),
    }
    print("shift:", json.dumps(res["shift"]), flush=True)

    if args.vllm_url:
        import requests
        from safetensors.torch import load_file, save_file

        amp_dir = os.path.join(args.work, f"amp{args.amp:g}")
        os.makedirs(amp_dir, exist_ok=True)
        shutil.copy(os.path.join(args.adapter, "adapter_config.json"), amp_dir)
        sd = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
        save_file({k: (v * args.amp if "lora_B" in k else v) for k, v in sd.items()},
                  os.path.join(amp_dir, "adapter_model.safetensors"))
        model.load_adapter(amp_dir, adapter_name="amp")
        model.set_adapter("amp")
        print("amp adapter:", json.dumps(adapter_stats(model, "amp")), flush=True)
        rn.url = args.vllm_url.rstrip("/")
        rn.load()
        from opensearch_infer.runners import InferenceConfig

        cfg = InferenceConfig(temperature=1.0, max_tokens=96)
        gap = {"base": [], "amp": [], "hf_amp_vs_hf_base": []}
        for r in list(rows.values())[: args.parity_n]:
            from opensearch_infer import pipeline

            _, parts = pipeline._bootstrap_images(r, r["data_id"], 0, "fvqa_train", None)
            contents = [{"role": "user", "parts": parts + [{"text": r["prompt"][0]["content"]}]}]
            msgs, imgs = rn.contents_to_template(contents)
            pid = rn.build_prompt_ids(msgs, imgs)
            rn.lora_path = amp_dir
            out = rn.generate_raw(pid, imgs, cfg, logprobs=True)
            rn.lora_path = None
            gen, v_lp = out["token_ids"], out["token_logprobs"]
            if not gen or not v_lp or None in v_lp:
                continue
            ids = torch.tensor([pid + gen], device=dev)
            mask = torch.tensor([0] * len(pid) + [1] * len(gen), device=dev)
            mm = rn._template.mm_plugin._get_mm_inputs(imgs, [], [], rn._processor) if imgs else {}
            mm = {k: (v.to(dev).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(dev))
                  for k, v in mm.items() if isinstance(v, torch.Tensor)}
            ref = torch.tensor(v_lp, device=dev)
            la, ys = logits_at(model, ids, mask, mm)
            with model.disable_adapter():
                lb, _ = logits_at(model, ids, mask, mm)
            hf_amp = torch.log_softmax(la, -1).gather(-1, ys[:, None]).squeeze(-1)
            hf_base = torch.log_softmax(lb, -1).gather(-1, ys[:, None]).squeeze(-1)
            gap["amp"].append(float((hf_amp - ref).abs().mean()))
            gap["base"].append(float((hf_base - ref).abs().mean()))
            gap["hf_amp_vs_hf_base"].append(float((hf_amp - hf_base).abs().mean()))
            print(f"  {r['data_id']}: {len(gen)} tok |vLLM(amp)-HF(amp)|={gap['amp'][-1]:.4f} "
                  f"|vLLM(amp)-HF(base)|={gap['base'][-1]:.4f} |HF(amp)-HF(base)|={gap['hf_amp_vs_hf_base'][-1]:.4f}",
                  flush=True)
        res["vllm_lora_parity"] = {k: (statistics.mean(v) if v else None) for k, v in gap.items()}
        res["vllm_lora_parity"]["amp_factor"] = args.amp
        print("parity:", json.dumps(res["vllm_lora_parity"]), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
