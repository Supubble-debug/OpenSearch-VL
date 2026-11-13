#!/usr/bin/env python3
"""RL trainer parity: sampling policy vs the policy the update differentiates (needs the instance).

The RL update computes log-probs of the sampled tokens in HF. For that gradient to be the
gradient of the *sampling* policy, three things must hold:

1. temperature — the rollouts sample with p ∝ exp(logits / T); HF must use
   log_softmax(logits / T). This prints the gap between T and T=1 so the fix is measured,
   not assumed.
2. token identity — the trainer re-renders each assistant turn from its *text*; if that
   re-tokenisation differs from the ids vLLM actually generated, the update is computed on
   a different sequence. Compares the generated ids with the ids obtained by re-tokenising
   the same text.
3. prompt identity — the prompt ids the trainer builds must equal the ids vLLM served.

    python tests/test_rl_logprob_parity.py --model /root/models/qwen3vl4b-sft-v2-merged \
        --data /root/rl_data/rl_train.parquet --samples 8 --temperature 0.7 --hf-gpu 0
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="parquet row with prompt[0].content + images")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--hf-gpu", default="0")
    args = ap.parse_args()

    import pandas as pd
    import requests
    import torch
    from transformers import AutoModelForImageTextToText

    from opensearch_infer import config
    from opensearch_infer.runners import InferenceConfig, Qwen35VLLMRunner

    runner = Qwen35VLLMRunner(config.MODEL_REGISTRY["qwen3-vl-4b-vllm"], args.model)
    runner.load()
    hf = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     device_map={"": f"cuda:{args.hf_gpu}"}).eval()
    head = hf.lm_head

    df = pd.read_parquet(args.data).head(args.samples)
    cfg = InferenceConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    gaps_t, gaps_1, rt_diff, prompt_diff, n = [], [], 0, 0, 0
    for row in df.to_dict("records"):
        from PIL import Image

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in
                  [x["bytes"] for x in (row["images"] or [])][:1]]
        contents = [{"role": "user", "parts": ([{"inline_data": {"mime_type": "image/jpeg",
                      "data": base64.b64encode(row["images"][0]["bytes"]).decode()}}] if images else [])
                     + [{"text": row["prompt"][0]["content"]}]}]
        msgs, imgs = runner.contents_to_template(contents)
        ids = runner.build_prompt_ids(msgs, imgs)
        out = runner.generate_raw(ids, imgs, cfg, logprobs=True)
        if out.get("prompt_token_ids") != ids:
            prompt_diff += 1
        gen = out["token_ids"]
        if not gen:
            continue
        # 2. token identity: re-tokenise the generated text
        re_ids = runner._tokenizer(out["text"], add_special_tokens=False)["input_ids"]
        if re_ids != gen:
            rt_diff += 1

        # 1. log-probs of the same ids in HF, with and without temperature
        kw = {"input_ids": torch.tensor([ids + gen], device=hf.device)}
        kw["attention_mask"] = torch.ones_like(kw["input_ids"])
        if imgs:
            mm = runner._template.mm_plugin._get_mm_inputs(imgs, [], [], runner._processor)
            for k, v in mm.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(hf.device)
                    kw[k] = v.to(hf.dtype) if v.dtype.is_floating_point else v
        with torch.no_grad():
            h = hf.model(**kw, use_cache=False).last_hidden_state[0]
            lg = head(h[len(ids) - 1: len(ids) + len(gen) - 1]).float()
            y = torch.tensor(gen, device=hf.device)
            lp_t = torch.log_softmax(lg / args.temperature, -1).gather(-1, y[:, None]).squeeze(-1)
            lp_1 = torch.log_softmax(lg, -1).gather(-1, y[:, None]).squeeze(-1)
        v_lp = out.get("token_logprobs")
        if v_lp and None not in v_lp:
            ref = torch.tensor(v_lp, device=hf.device)
            gaps_t.append(float((lp_t - ref).abs().mean()))
            gaps_1.append(float((lp_1 - ref).abs().mean()))
        else:
            print("  WARN: server returned no logprobs; compare skipped (needs --logprobs support)")
        n += 1
        print(f"  {row['data_id']}: {len(gen)} tokens, prompt ids "
              f"{'MATCH' if out.get('prompt_token_ids') == ids else 'DIFFER'}, "
              f"retokenise {'MATCH' if re_ids == gen else 'DIFFER'}"
              + (f", |Δlogp| T={gaps_t[-1]:.4f} vs T=1 {gaps_1[-1]:.4f}" if gaps_t else ""))

    print(f"\n{n} samples: prompt-id mismatches {prompt_diff}, re-tokenisation mismatches {rt_diff}")
    if gaps_t:
        mt, m1 = sum(gaps_t) / len(gaps_t), sum(gaps_1) / len(gaps_1)
        print(f"mean |HF - vLLM| logprob: with temperature {mt:.4f}   without {m1:.4f}"
              f"   (ratio {m1 / max(mt, 1e-9):.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
