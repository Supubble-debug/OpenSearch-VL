#!/usr/bin/env python3
"""Audit SFT trajectories' final answers with a local LLM judge (no gold answers exist).

Search-VL-SFT-36K stores only conversations/images/system/tools, so correctness is
judged against the trajectory's own tool evidence: is the final answer supported by
what the tools returned, not contradicted by it, and at the granularity the question
asks for (genus vs family, event vs history)? The judge is text-only and cannot see
the images, so answers that rest on reading the image alone come out "unclear" and
are not counted as errors.

Modes:
  --sample N        stratified random sample (category ratios as --ratios) to estimate the error rate
  --manifest FILE   check every id listed (selection manifest.jsonl) -> bad-id list for filtering

    python check_sft_answers.py --data-root /root/datasets/Search-VL-SFT-36K \
        --tokenizer /root/models/Qwen3-VL-4B-Instruct --sample 200 --out /tmp/audit
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

_spec = importlib.util.spec_from_file_location("select_sft_subset", Path(__file__).with_name("select_sft_subset.py"))
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

SYSTEM = (
    "You audit training trajectories of a multimodal web-search agent. You see the user's question, "
    "the tool calls with their outputs, and the agent's final answer. You cannot see the images. "
    "Reply with one JSON object only."
)
TASK = """Judge the FINAL ANSWER.

- supported: "yes" if its key claim is backed by the tool outputs; "no" if the outputs do not back it \
(e.g. the answer names something the evidence never mentions or points away from); "unclear" if it \
rests only on what the image shows, which you cannot check.
- contradicted: "yes" if the tool outputs contradict the final answer, else "no".
- granularity_ok: "yes" if it answers exactly what the question asks (right kind of entity and level, \
e.g. genus vs family, the event vs the city's history, a year vs a century), else "no".

Return {{"supported": "...", "contradicted": "...", "granularity_ok": "...", "reason": "<= 30 words"}}

QUESTION:
{question}

TOOL CALLS AND OUTPUTS:
{evidence}

FINAL ANSWER:
{answer}"""


SYSTEM_VISION = (
    "You audit training trajectories of a multimodal web-search agent. You see the user's question, "
    "the image(s) attached to the trajectory, the tool calls with their outputs, and the agent's final "
    "answer. Reply with one JSON object only."
)
TASK_VISION = """Judge the FINAL ANSWER. The attached images are, in order: {image_list}.

Tool outputs are often summaries written by another language model: they can invent details or merely \
repeat the search query. Treat a claim as backed only if the tool output gives concrete source content \
for it (a page title, a quoted passage, a matching entity), not just the agent's own wording echoed back.

- image_matches: "yes" if the question image is what the trajectory talks about (same object, script, \
chart, scene), "no" if the trajectory describes something that is not in the image.
- supported: "yes" if the key claim is backed by the tool outputs or plainly visible in the image; "no" if \
neither backs it (invented names, numbers, dates; a misread image; a guess the evidence does not confirm); \
"unclear" only if you truly cannot tell.
- contradicted: "yes" if the image or the tool outputs contradict the final answer, else "no".
- granularity_ok: "yes" if it answers exactly what the question asks (right kind of entity and level, \
e.g. a specific artifact name vs a generic category, genus vs family; a date or year is fine for a \
"what time/when" question unless an hour is clearly required); "no" otherwise, including refusals.

Return {{"image_matches": "...", "supported": "...", "contradicted": "...", "granularity_ok": "...", "reason": "<= 30 words"}}

QUESTION:
{question}

TOOL CALLS AND OUTPUTS:
{evidence}

FINAL ANSWER:
{answer}"""


def encode_images(paths: list[str], max_pixels: int = 512 * 512) -> list[str]:
    import base64, io
    from PIL import Image
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        w, h = im.size
        if w * h > max_pixels:
            f = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(28, int(w * f)), max(28, int(h * f))))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def pick_images(paths: list[str], n: int) -> tuple[list[str], str]:
    """Question image first, then the most recent tool-produced images."""
    if n <= 0 or not paths:
        return [], ""
    chosen = paths[:1] + (paths[1:][-(n - 1):] if n > 1 else [])
    names = ["the question image"] + [f"tool-produced image #{paths.index(p)}" for p in chosen[1:]]
    return chosen, "; ".join(names)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def build_prompt(sample: dict, tok, max_tokens: int, task: str = TASK, **extra) -> str | None:
    conv = sample["conversations"]
    question = conv[0]["value"].replace("<image>", "[image]").strip()
    final = conv[-1]["value"]
    answer = final[final.rfind("<response>") + len("<response>"):final.rfind("</response>")].strip()
    steps = []
    for i in range(1, len(conv) - 1, 2):
        calls = sel.TOOL_CALL_RE.findall(conv[i]["value"])
        obs = conv[i + 1]["value"] if i + 1 < len(conv) else ""
        steps.append((calls[0] if calls else "?", obs.replace("<image>", "[image]")))
    cap = 6000
    while cap >= 300:
        evidence = "\n\n".join(
            f"[{n}] CALL {c}\nOUTPUT {o[:cap]}{' ...[truncated]' if len(o) > cap else ''}"
            for n, (c, o) in enumerate(steps, 1)
        )
        prompt = task.format(question=question[:3000], evidence=evidence, answer=answer[:3000], **extra)
        if len(tok(prompt, add_special_tokens=False)["input_ids"]) <= max_tokens:
            return prompt
        cap = int(cap * 0.7)
    return None


def judge(api: str, model: str, prompt: str, images: list[str] | None = None, system: str = SYSTEM) -> dict:
    content = ([{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": prompt}]
               if images else prompt)
    body = {
        "model": model, "temperature": 0.0, "max_tokens": 200,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last = ""
    for attempt in range(4):
        try:
            r = requests.post(f"{api}/chat/completions", json=body, timeout=300)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", text, re.S)
                out = json.loads(m.group(0)) if m else {}
                if all(k in out for k in ("supported", "contradicted", "granularity_ok")):
                    return {k: str(out[k]).strip().lower() if k != "reason" else out[k] for k in out}
                last = f"unparsable: {text[:200]}"
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code == 400:
                    break
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(2 * (attempt + 1))
    return {"judge_failed": last}


def is_error(v: dict) -> bool:
    return (v.get("contradicted") == "yes" or v.get("supported") == "no" or v.get("granularity_ok") == "no"
            or v.get("image_matches") == "no")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", required=True, help="for prompt-length budgeting")
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int)
    g.add_argument("--manifest")
    ap.add_argument("--ratios", default="retrieval=0.6,crop_ocr=0.3,enhancement=0.1")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="summariser")
    ap.add_argument("--max-prompt-tokens", type=int, default=14500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--images", type=int, default=0,
                    help="attach up to N trajectory images (question image + latest tool images); "
                         "the server must accept images (SUMMARY_MAX_IMAGES>=N)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, src_dirs = {}, {}
    for src, fname in sel.SOURCES.items():
        src_dirs[src] = Path(args.data_root) / src
        for idx, s in enumerate(json.load(open(Path(args.data_root) / src / fname, encoding="utf-8"))):
            data[f"{src}:{idx}"] = s

    if args.manifest:
        ids = [json.loads(line)["id"] for line in open(args.manifest, encoding="utf-8")]
        cats = {json.loads(line)["id"]: json.loads(line).get("cat") for line in open(args.manifest, encoding="utf-8")}
    else:
        pools = collections.defaultdict(list)
        for sid, s in data.items():
            err, info = sel.check_structure(s)
            cat = sel.categorize(info["tools"]) if err is None else None
            if cat:
                pools[cat].append(sid)
        rng = random.Random(args.seed)
        ratios = {k: float(v) for k, v in (kv.split("=") for kv in args.ratios.split(","))}
        ids, cats = [], {}
        for cat, r in ratios.items():
            pick = rng.sample(sorted(pools[cat]), min(len(pools[cat]), round(args.sample * r)))
            ids += pick
            cats.update({i: cat for i in pick})
        print(f"[sample] pools { {c: len(v) for c, v in pools.items()} } -> {len(ids)} ids", flush=True)

    done = {}
    res_path = out / "verdicts.jsonl"
    if res_path.exists():  # resume
        for line in open(res_path, encoding="utf-8"):
            r = json.loads(line)
            if "judge_failed" not in r:
                done[r["id"]] = r
    todo = [i for i in ids if i not in done]
    print(f"[judge] {len(todo)} to judge ({len(done)} already done)", flush=True)

    def work(sid: str) -> dict:
        if args.images:
            d = src_dirs[sid.split(":")[0]]
            paths = [p if os.path.isabs(p) else str(d / p) for p in data[sid].get("images") or []]
            chosen, names = pick_images(paths, args.images)
            imgs = encode_images(chosen)
            # 512x512 at 32 px per merged token = 256 tokens per image (+ markers)
            prompt = build_prompt(data[sid], tok, args.max_prompt_tokens - 300 * len(imgs), task=TASK_VISION,
                                  image_list=names or "none")
            v = (judge(args.api, args.model, prompt, imgs, SYSTEM_VISION) if prompt
                 else {"judge_failed": "prompt too long"})
        else:
            prompt = build_prompt(data[sid], tok, args.max_prompt_tokens)
            v = judge(args.api, args.model, prompt) if prompt else {"judge_failed": "prompt too long"}
        return {"id": sid, "cat": cats.get(sid), "source": sid.split(":")[0], **v}

    t0 = time.time()
    with open(res_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as ex:
        for n, r in enumerate(ex.map(work, todo), 1):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            done[r["id"]] = r
            if n % 100 == 0 or n == len(todo):
                print(f"[judge] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    rows = [done[i] for i in ids if i in done]
    ok = [r for r in rows if "judge_failed" not in r]
    errs = [r for r in ok if is_error(r)]

    def rate(rs: list[dict]) -> dict:
        k = sum(is_error(r) for r in rs)
        return {"n": len(rs), "errors": k, "rate": k / len(rs) if rs else 0.0, "ci95": wilson(k, len(rs))}

    summary = {
        "judged": len(ok), "judge_failed": len(rows) - len(ok),
        "error": rate(ok),
        "by_category": {c: rate([r for r in ok if r["cat"] == c]) for c in sorted({r["cat"] for r in ok})},
        "by_source": {s: rate([r for r in ok if r["source"] == s]) for s in sorted({r["source"] for r in ok})},
        "contradicted": sum(r["contradicted"] == "yes" for r in ok),
        "unsupported": sum(r["supported"] == "no" for r in ok),
        "unclear": sum(r["supported"] == "unclear" for r in ok),
        "bad_granularity": sum(r["granularity_ok"] == "no" for r in ok),
        "error_examples": [{"id": r["id"], "reason": r.get("reason", "")} for r in errs[:25]],
    }
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(out / "bad_ids.txt", "w", encoding="utf-8") as fh:
        fh.writelines(r["id"] + "\n" for r in errs)
    print(json.dumps({k: v for k, v in summary.items() if k != "error_examples"}, ensure_ascii=False, indent=1))
    if os.getenv("OUTPUT_ROOT"):
        for name in ("summary.json", "bad_ids.txt", "verdicts.jsonl"):
            Path(os.environ["OUTPUT_ROOT"], f"answer_check_{name}").write_bytes((out / name).read_bytes())
        json.dump({"answer_error_rate": summary["error"]["rate"], "judged": len(ok)},
                  open(Path(os.environ["OUTPUT_ROOT"]) / "metrics.json", "w"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
