"""fp32 KL probe: how far the current LoRA policy is from the reference (SFT) policy.

bf16 cannot measure this for small adapters: on 2026-11-13 an r2 adapter scaled by 0.1 still showed
KL 8e-4 in bf16 but 1.0e-5 in fp32 (unscaled: 1.0e-3 in both), i.e. bf16 has a ~8e-4 floor. The
probe therefore keeps an fp32 copy of the base model with the same LoRA layout, copies the trainer's
LoRA weights into it, and compares adapter on/off on a fixed set of reference trajectories.

Trajectories are rebuilt from saved rollout json (trainer rendering, all assistant turns), cut
before the first tool that returned an image, since the json does not keep intermediate images.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Dict, List, Optional

TEXT_TOOLS = {"text_search", "image_search", "web_search", "search", "layout_parsing", "local_search",
              "local_image_search", "lens_scan"}


def rebuild(traj: Dict[str, Any], row: Dict[str, Any], prompt_in_system: bool) -> List[Dict[str, Any]]:
    """Saved trajectory json + dataset row -> pipeline ``contents`` (text-only observations)."""
    from opensearch_infer import pipeline, tools

    _, parts = pipeline._bootstrap_images(row, traj["case_id"], 0, "fvqa_train", None)
    text = row["prompt"][0]["content"]
    if not prompt_in_system:
        text = f"<tools>\n{tools.get_tools_definition()}\n</tools>\n\n" + text
    contents = [{"role": "user", "parts": parts + [{"text": text}]}]
    for t in traj.get("turns", []):
        rt = t.get("response_text")
        if not rt:
            break
        contents.append({"role": "model", "parts": [{"text": rt}]})
        tc = t.get("tool_call")
        if not tc or t.get("tool_output") is None:
            break
        try:
            name = json.loads(tc).get("name")
        except ValueError:
            break
        if name not in TEXT_TOOLS and not t.get("tool_error"):
            break  # this tool returned an image the json no longer has
        contents.append({"role": "user", "parts": [{"text": t["tool_output"]}]})
    return contents


def encode(rn, contents, device, dtype=None):
    """Trainer rendering of all assistant turns -> (ids, loss mask, mm inputs) or None."""
    import torch

    dtype = dtype or torch.bfloat16
    msgs, images = rn.contents_to_template(contents)
    while msgs and msgs[-1]["role"] != "assistant":
        dropped = msgs.pop()
        n = dropped["content"].count("<image>")
        if n:
            images = images[:-n]
    if not msgs:
        return None
    rendered = rn._template.mm_plugin.process_messages(msgs, images, [], [], rn._processor)
    pairs = rn._template.encode_multiturn(rn._tokenizer, rendered, rn._system, rn._tools)
    ids, mask = [], []
    for p, r in pairs:
        ids += p + r
        mask += [0] * len(p) + [1] * len(r)
    mm = rn._template.mm_plugin._get_mm_inputs(images, [], [], rn._processor) if images else {}
    mm = {k: (v.to(device).to(dtype) if v.dtype.is_floating_point else v.to(device))
          for k, v in mm.items() if isinstance(v, torch.Tensor)}
    return torch.tensor([ids], device=device), torch.tensor(mask, device=device), mm


def load_reference_trajectories(rollout_dirs: List[str], data_paths: List[str], n: int,
                                seed: int = 0) -> List[tuple]:
    """[(trajectory json, dataset row)] for the first n shuffled trajectories that have a row."""
    import pandas as pd

    rows = {}
    for p in data_paths:
        for r in pd.read_parquet(p).to_dict("records"):
            rows[r["data_id"]] = r
    files = sorted(f for d in rollout_dirs for f in glob.glob(os.path.join(d, "s*", "*_trajectory.json")))
    random.Random(seed).shuffle(files)
    out = []
    for f in files:
        traj = json.load(open(f, encoding="utf-8"))
        row = rows.get(traj.get("case_id"))
        if row is not None:
            out.append((traj, row))
        if len(out) >= n:
            break
    return out


def kl_stats(hb, ha, ys, temperature: float, head, chunk: int = 1024) -> Dict[str, float]:
    """Chunked KL(ref || policy) at temperature T from hidden states of both models."""
    import torch

    kl_sum, dlp_abs, flips, n = 0.0, 0.0, 0, 0
    for i in range(0, len(ys), chunk):
        lb = head(hb[i:i + chunk]).float()
        la = head(ha[i:i + chunk]).float()
        y = ys[i:i + chunk]
        pb, pa = torch.log_softmax(lb / temperature, -1), torch.log_softmax(la / temperature, -1)
        kl_sum += float((pb.exp() * (pb - pa)).sum())
        dlp_abs += float((pa - pb).gather(-1, y[:, None]).abs().sum())
        flips += int((lb.argmax(-1) != la.argmax(-1)).sum())
        n += len(y)
        del lb, la, pb, pa
    return {"kl_sum": kl_sum, "dlp_abs_sum": dlp_abs, "flips": flips, "tokens": n}


class KLProbe:
    """fp32 reference model + LoRA copy; ``measure(policy_peft_model)`` -> per-token KL vs reference."""

    def __init__(self, model_path: str, rn, lora_config, trajectories: List[tuple], temperature: float,
                 max_len: int = 8000, device: str = "cuda:0") -> None:
        import torch
        from peft import get_peft_model
        from transformers import AutoModelForImageTextToText

        self.torch, self.T, self.device = torch, temperature, device
        base = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.float32,
                                                           device_map={"": device})
        self.model = get_peft_model(base, lora_config).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.items = []
        for traj, row in trajectories:
            enc = encode(rn, rebuild(traj, row, rn.prompt_in_system), "cpu", torch.float32)
            if enc is not None and enc[0].shape[1] <= max_len:
                self.items.append(enc)

    def _hidden(self, ids, mask, mm):
        torch = self.torch
        inner = self.model.base_model.model
        h = inner.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **mm).last_hidden_state[0]
        tgt = mask[1:].nonzero(as_tuple=True)[0] + 1
        return h[tgt - 1], ids[0, tgt]

    def measure(self, policy_model) -> Optional[Dict[str, float]]:
        if not self.items:
            return None
        torch = self.torch
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        sd = {k: v.detach().float() for k, v in get_peft_model_state_dict(policy_model).items()}
        set_peft_model_state_dict(self.model, sd)
        tot = {"kl_sum": 0.0, "dlp_abs_sum": 0.0, "flips": 0, "tokens": 0}
        head = self.model.base_model.model.lm_head
        with torch.no_grad():
            for ids, mask, mm in self.items:
                ids, mask = ids.to(self.device), mask.to(self.device)
                mm = {k: v.to(self.device) for k, v in mm.items()}
                with self.model.disable_adapter():
                    hb, ys = self._hidden(ids, mask, mm)
                ha, _ = self._hidden(ids, mask, mm)
                for k, v in kl_stats(hb, ha, ys, self.T, head).items():
                    tot[k] += v
                del hb, ha
        torch.cuda.empty_cache()
        n = max(1, tot["tokens"])
        return {"kl_probe/kl_per_token": tot["kl_sum"] / n, "kl_probe/abs_dlogp": tot["dlp_abs_sum"] / n,
                "kl_probe/argmax_flip": tot["flips"] / n, "kl_probe/tokens": tot["tokens"],
                "kl_probe/trajectories": len(self.items)}
