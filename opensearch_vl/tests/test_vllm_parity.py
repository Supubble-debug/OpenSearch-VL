#!/usr/bin/env python3
"""vLLM token server == in-process prompt, token for token (needs the instance).

For real SFT trajectories, builds each assistant turn's prompt with
``Qwen35VLLMRunner`` (LLaMA-Factory template, as in training), sends it to
``vllm_token_server.py`` and checks:

1. the prompt ids vLLM actually used (after re-expanding ``<|image_pad|>`` from the
   image grid) equal the template's ids — same text tokens, same image size;
2. with ``--hf-gpu``, greedy continuations of the merged model in HF and in vLLM
   agree on the first ``--check-tokens`` tokens (catches a bad merge or different
   pixel preprocessing; small numeric drift may end agreement late, so the check
   only needs a common prefix).

    python tests/test_vllm_parity.py --model /root/models/qwen3vl4b-sft-merged \
        --data /root/datasets/sft_subset_q3vl/sft_subset.json --samples 12 --hf-gpu 1
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--registry", default="qwen3-vl-4b-vllm")
    ap.add_argument("--hf-gpu", default=None, help="GPU for the HF greedy comparison (skip if unset)")
    ap.add_argument("--check-tokens", type=int, default=16)
    args = ap.parse_args()

    from opensearch_infer import config
    from opensearch_infer.runners import InferenceConfig, Qwen35VLLMRunner

    runner = Qwen35VLLMRunner(config.MODEL_REGISTRY[args.registry], args.model)
    runner.load()
    data = json.load(open(args.data, encoding="utf-8"))
    data.sort(key=lambda s: -len(s["images"]))  # multi-image trajectories first
    picked = data[: args.samples // 2] + data[-(args.samples - args.samples // 2):]

    hf = None
    if args.hf_gpu is not None:
        import torch
        from transformers import AutoModelForImageTextToText

        hf = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": f"cuda:{args.hf_gpu}"}).eval()

    greedy = InferenceConfig(temperature=0.0, max_tokens=args.check_tokens)
    checked = agree = 0
    for sample in picked:
        conv, contents, used = sample["conversations"], [], 0
        for m in conv:
            if m["from"] == "gpt":
                msgs, images = runner.contents_to_template(contents)
                ids = runner.build_prompt_ids(msgs, images)
                out = runner.generate_raw(ids, images, greedy)
                assert out["prompt_token_ids"] == ids, (
                    f"prompt mismatch (turn {checked}): vLLM {len(out['prompt_token_ids'])} vs template {len(ids)} ids")
                if hf is not None:
                    import torch
                    kw = {"input_ids": torch.tensor([ids], device=hf.device)}
                    kw["attention_mask"] = torch.ones_like(kw["input_ids"])
                    if images:
                        mm = runner._template.mm_plugin._get_mm_inputs(images, [], [], runner._processor)
                        for k, v in mm.items():
                            if isinstance(v, torch.Tensor):
                                v = v.to(hf.device)
                                kw[k] = v.to(hf.dtype) if v.dtype.is_floating_point else v
                    with torch.no_grad():
                        g = hf.generate(**kw, max_new_tokens=args.check_tokens, do_sample=False,
                                        eos_token_id=runner._eos_ids)[0][len(ids):].tolist()
                    v = out["token_ids"]
                    n = next((i for i, (a, b) in enumerate(zip(g, v)) if a != b), min(len(g), len(v)))
                    agree += n >= min(8, len(g), len(v))
                    if n < min(8, len(g), len(v)):
                        print(f"  greedy diverges at token {n}: HF {g[:12]} vs vLLM {v[:12]}")
                checked += 1
                contents.append({"role": "model", "parts": [{"text": m["value"]}]})
                continue
            n_img = m["value"].count("<image>")
            parts = []
            for path in sample["images"][used: used + n_img]:
                with open(path, "rb") as fh:
                    parts.append({"inline_data": {"mime_type": "image/png",
                                                  "data": base64.b64encode(fh.read()).decode()}})
            used += n_img
            parts.append({"text": m["value"][len("<image>") * n_img:]})
            contents.append({"role": "user", "parts": parts})
    print(f"OK: {checked} assistant turns from {len(picked)} trajectories, prompt ids identical in vLLM"
          + (f"; greedy prefix agreement {agree}/{checked}" if hf is not None else ""))
    return 0 if hf is None or agree >= 0.9 * checked else 1


if __name__ == "__main__":
    sys.exit(main())
