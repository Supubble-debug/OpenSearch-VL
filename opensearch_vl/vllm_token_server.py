#!/usr/bin/env python3
"""Token-in/token-out vLLM server for agent rollouts (runs in /root/venvs/vllm).

The client (``Qwen35VLLMRunner``) renders the prompt with LLaMA-Factory's own
template, exactly as in SFT training, and sends the token ids plus the already
resized images. Each run of ``<|image_pad|>`` tokens is collapsed to one token
and vLLM re-expands it from the image grid, so the vision encoder sees the same
pixels and the LM the same ids; ``prompt_token_ids`` (as expanded by vLLM) is
returned so the parity test can check that.

    /root/venvs/vllm/bin/python vllm_token_server.py --model /root/models/qwen3vl4b-sft-merged \
        --port 8001 --gpu 0

POST /generate {"prompt_token_ids": [...], "images": ["<base64 png>", ...], "lora_path": null,
                "max_tokens": 4096, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "stop_token_ids": [...], "seed": null}
  -> {"text", "token_ids", "finish_reason", "prompt_token_ids"}
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import uuid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-images", type=int, default=24)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--enable-lora", action="store_true", help="serve per-request LoRA adapters (RL)")
    ap.add_argument("--max-lora-rank", type=int, default=32)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import uvicorn
    from fastapi import FastAPI
    from PIL import Image
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:  # older layout
        from vllm import AsyncLLMEngine as Engine

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    engine = Engine.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            limit_mm_per_prompt={"image": args.max_images, "video": 0},
            enable_prefix_caching=True,
            **(dict(enable_lora=True, max_lora_rank=args.max_lora_rank, max_loras=2, max_cpu_loras=4)
               if args.enable_lora else {}),
        )
    )
    lora_ids: dict[str, int] = {}
    app = FastAPI()

    def collapse(ids: list[int]) -> list[int]:
        out: list[int] = []
        for t in ids:
            if t == pad_id and out and out[-1] == pad_id:
                continue
            out.append(t)
        return out

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "model": args.model}

    @app.post("/generate")
    async def generate(req: dict) -> dict:
        images = [
            Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in req.get("images") or []
        ]
        prompt = {"prompt_token_ids": collapse(req["prompt_token_ids"]) if images else req["prompt_token_ids"]}
        if images:
            prompt["multi_modal_data"] = {"image": images}
        params = SamplingParams(
            max_tokens=int(req.get("max_tokens", 4096)),
            temperature=float(req.get("temperature", 1.0)),
            top_p=float(req.get("top_p", 1.0)),
            top_k=int(req.get("top_k", -1)),
            stop_token_ids=req.get("stop_token_ids") or None,
            seed=req.get("seed"),
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )
        lora = None
        if req.get("lora_path"):
            from vllm.lora.request import LoRARequest

            path = req["lora_path"]
            lid = lora_ids.setdefault(path, len(lora_ids) + 1)
            lora = LoRARequest(f"lora{lid}", lid, path)
        final = None
        async for out in engine.generate(prompt, params, request_id=uuid.uuid4().hex, lora_request=lora):
            final = out
        c = final.outputs[0]
        return {
            "text": c.text,
            "token_ids": list(c.token_ids),
            "finish_reason": c.finish_reason,  # "stop" | "length"
            "prompt_token_ids": list(final.prompt_token_ids or []),
        }

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
