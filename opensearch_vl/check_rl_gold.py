#!/usr/bin/env python3
"""Audit RL gold answers with the local multimodal judge (27B, image + thinking).

RL rewards compare the policy's answer with the gold answer, so a wrong gold rewards
wrong answers. Gold answers outside demo_1k were extracted from agent-written reference
solutions; a 60-question human/Claude audit found ~18% wrong and ~35% questions without
a single checkable answer. This judge sees the question image, the question, the short
gold and the long reference, and returns:

- question_ok: "no" only if the answer cannot be read off the image / public facts (a
  value the chart does not show), the question asks for an explanation or opinion, or so
  many answers are acceptable that grading is arbitrary. "What is it called / does it
  have a name" is fine when the gold is a reasonable specific name (user review 2026-11-11).
- gold_correct: "yes" | "no" | "unsure" against the image and the judge's own knowledge;
  the reference may misread the image or invent names, dates, counts.

    python check_rl_gold.py --cands probe300.parquet reserve.parquet --out /root/rl_data/gold \
        --api http://127.0.0.1:8000/v1 [--shard 0/2]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SYSTEM = (
    "You audit gold answers of a visual question answering dataset that are used as reward targets. "
    "You see the question image, the question, the short gold answer, and the long reference solution it was "
    "taken from. The reference was written by an AI agent and may be wrong: it can misread the image (script, "
    "object type, counts) or invent specific names, dates and numbers. Think carefully, then end your reply with "
    "one JSON object."
)
TASK = """QUESTION (about the attached image):
{question}

SHORT GOLD ANSWER:
{gold}

LONG REFERENCE SOLUTION:
{long}

Decide two things.

question_ok:
- "no" only if (a) the answer cannot be determined from the image plus public facts (e.g. it asks for a value the
  chart does not show, or the image gives no way to tell which event/person is meant), (b) the question asks for an
  explanation, interpretation or opinion rather than a short checkable fact, or (c) so many different answers would
  be acceptable that grading against one gold is arbitrary.
- Otherwise "yes". Questions like "what is this called?" or "does it have a name?" are fine when the gold is a
  reasonable specific name; multi-part questions are fine when every part has a checkable answer.

gold_correct:
- "yes" if the gold is right for this image and question at the asked granularity (all parts);
- "no" if the image or well-known facts contradict it (wrong object, wrong script, wrong count, wrong entity, a
  detail the image shows to be false, a coarser category than asked, an invented part);
- "unsure" if you cannot verify it either way.

End with: {{"question_ok": "yes|no", "gold_correct": "yes|no|unsure", "reason": "<= 30 words"}}"""


def encode_image(raw: bytes, max_pixels: int = 768 * 768) -> str:
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    if w * h > max_pixels:
        f = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * f)), max(28, int(h * f))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse(text: str) -> dict:
    text = text.rsplit("</think>", 1)[-1]
    for cand in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            out = json.loads(cand)
        except ValueError:
            continue
        if "question_ok" in out and "gold_correct" in out:
            return {"question_ok": str(out["question_ok"]).strip().lower(),
                    "gold_correct": str(out["gold_correct"]).strip().lower(),
                    "reason": str(out.get("reason", ""))[:300]}
    return {}


def judge(api: str, model: str, row: dict) -> dict:
    content = [{"type": "image_url", "image_url": {"url": encode_image(row["images"][0]["bytes"])}},
               {"type": "text", "text": TASK.format(question=row["prompt"][0]["content"][:3000],
                                                     gold=row["answer"][:500],
                                                     long=(row.get("long_answer") or "")[:6000])}]
    body = {"model": model, "max_tokens": 6000, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": True}}
    last = ""
    for attempt in range(4):
        try:
            r = requests.post(f"{api}/chat/completions", json=body, timeout=900)
            if r.status_code == 200:
                out = parse(r.json()["choices"][0]["message"].get("content") or "")
                if out:
                    return out
                last = "unparsable"
            else:
                last = f"HTTP {r.status_code}: {r.text[:150]}"
                if r.status_code == 400:
                    break
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(2 * (attempt + 1))
    return {"judge_failed": last}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="summariser")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    import pandas as pd
    rows = []
    for p in args.cands:
        rows += pd.read_parquet(p).to_dict("records")
    seen, uniq = set(), []
    for r in rows:
        if r["data_id"] not in seen:
            seen.add(r["data_id"]); uniq.append(r)
    i, k = (int(x) for x in args.shard.split("/"))
    todo_rows = [r for n, r in enumerate(sorted(uniq, key=lambda r: r["data_id"])) if n % k == i]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"verdicts_{i}of{k}.jsonl"
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "judge_failed" not in r:
                done.add(r["data_id"])
    todo = [r for r in todo_rows if r["data_id"] not in done]
    print(f"[gold] {len(todo)} to judge ({len(done)} done) shard {args.shard}", flush=True)
    t0 = time.time()

    def work(r: dict) -> dict:
        return {"data_id": r["data_id"], "category": r.get("category"), **judge(args.api, args.model, r)}

    with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as ex:
        for n, v in enumerate(ex.map(work, todo), 1):
            fh.write(json.dumps(v, ensure_ascii=False) + "\n"); fh.flush()
            if n % 100 == 0 or n == len(todo):
                print(f"[gold] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
