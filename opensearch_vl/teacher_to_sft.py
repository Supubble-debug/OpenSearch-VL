#!/usr/bin/env python3
"""Filter teacher trajectories and convert them to the LLaMA-Factory sharegpt format of SFT v2.

Inputs: one or more teacher sample dirs (run_teacher.sh output, each with *_trajectory.json and
eval_report_vdr_strict_details.jsonl from eval_with_gpt4o.py --strict) and the cases parquet the
teacher ran on (for the original images).

A trajectory is kept only if:
  - the strict judge says correct (judge_failed rows are skipped, not counted as correct);
  - it ended with a <response> (stop_reason == "response") and made >= --min-calls tool calls;
  - every assistant turn has a <think> block and no turn leaks the teacher-only rules;
  - every tool call parsed (no tool_error) and every tool-returned image file exists.
At most --per-question kept trajectories per question (fewest repeated queries first).

System prompt and tool schema are copied verbatim from the v2 SFT data, so the student sees
exactly its usual prompt: the teacher's extra rules never enter the training data.

    python teacher_to_sft.py --cases /root/rl_data/sftc_v3a.parquet --v2-data /root/datasets/sft_subset_q3vl_v2/sft_subset.json \
        --samples /root/teacher_v3a/s0 /root/teacher_v3a/s1 ... --out /root/datasets/sft_v3a_new
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path

import pandas as pd

RULE_LEAK = re.compile(r"(?i)mandatory rules|rule ?[1-6]\b|per the rules?|the rules say|demonstrations? for a smaller")
IMG_PATH = re.compile(r"uploaded to: (\S+?\.(?:png|jpe?g|webp))", re.I)


def judged(sample_dir: str) -> dict:
    out = {}
    p = Path(sample_dir) / "eval_report_vdr_strict_details.jsonl"
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if str(r.get("judge_failed")).lower() == "true":
            continue
        out[r["case_id"]] = str(r.get("acc")).strip() in ("1", "1.0", "True", "true")
    return out


def convert(t: dict, orig_image: str) -> tuple[dict | None, str, int]:
    """Return (sample, reject_reason, n_repeat_queries)."""
    if t.get("stop_reason") != "response":
        return None, f"stop:{t.get('stop_reason')}", 0
    conv = [{"from": "human", "value": "<image>" + t["original_data"]["question"]}]
    images = [orig_image]
    calls, queries = 0, []
    for x in t["turns"]:
        text = x.get("response_text") or ""
        if "<think>" not in text:
            return None, "no_think", 0
        if RULE_LEAK.search(text):
            return None, "rule_leak", 0
        conv.append({"from": "gpt", "value": text.strip()})
        if not x.get("tool_call"):
            continue
        if str(x.get("tool_error")).lower() == "true":
            return None, "tool_error", 0
        calls += 1
        try:
            c = json.loads(x["tool_call"])
            q = (c.get("parameters") or {}).get("q")
            if q:
                queries.append(" ".join(str(q).lower().split()))
        except Exception:
            return None, "bad_tool_json", 0
        obs = (x.get("tool_output") or "").strip()
        m = IMG_PATH.search(obs)
        if m:
            if not os.path.exists(m.group(1)):
                return None, "missing_tool_image", 0
            images.append(m.group(1))
            obs = "<image>" + obs
        conv.append({"from": "observation", "value": obs})
    if conv[-1]["from"] != "gpt" or "<response>" not in conv[-1]["value"]:
        return None, "no_final_response", 0
    return {"conversations": conv, "images": images, "n_calls": calls}, "", len(queries) - len(set(queries))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--v2-data", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-question", type=int, default=2)
    ap.add_argument("--min-calls", type=int, default=1)
    args = ap.parse_args()

    v2 = json.load(open(args.v2_data, encoding="utf-8"))
    system, tools = v2[0]["system"], v2[0]["tools"]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    cases = pd.read_parquet(args.cases)
    orig = {}
    for _, r in cases.iterrows():
        p = out / "images" / f"{r['data_id']}.png"
        if not p.exists():
            from PIL import Image
            import io
            Image.open(io.BytesIO(r["images"][0]["bytes"])).convert("RGB").save(p)
        orig[r["data_id"]] = str(p)

    cand = collections.defaultdict(list)
    reasons = collections.Counter()
    for sd in args.samples:
        J = judged(sd)
        for p in sorted(Path(sd).glob("*_trajectory.json")):
            t = json.load(open(p, encoding="utf-8"))
            q = t["case_id"]
            if q not in J:
                reasons["judge_failed_or_missing"] += 1
                continue
            if not J[q]:
                reasons["wrong"] += 1
                continue
            s, why, rep = convert(t, orig[q])
            if s is None:
                reasons[why] += 1
                continue
            if s["n_calls"] < args.min_calls:
                reasons["too_few_calls"] += 1
                continue
            cand[q].append((rep, -s["n_calls"], sd, s))
    rows, manifest = [], []
    for q, lst in cand.items():
        lst.sort(key=lambda z: (z[0], z[1]))
        for rep, _, sd, s in lst[: args.per_question]:
            rows.append({"conversations": s["conversations"], "images": s["images"], "system": system, "tools": tools})
            manifest.append({"case_id": q, "sample": os.path.basename(sd), "n_calls": s["n_calls"], "repeat_queries": rep})
    json.dump(rows, open(out / "sft_new.json", "w", encoding="utf-8"), ensure_ascii=False)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    report = {"kept": len(rows), "questions": len(cand), "rejected": dict(reasons),
              "calls_hist": dict(collections.Counter(min(m["n_calls"], 8) for m in manifest))}
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
