#!/usr/bin/env python3
"""Summarise probe rollouts (K samples per question) and split dev / train questions.

Reads <work>/s{0..K-1}/*_trajectory.json and the strict + lenient judge details,
writes <out>/probe_report.json, probe_questions.jsonl and dev / train id lists
(+ parquet subsets of the candidates):

- reward: strict judge accuracy (the RL reward candidate); lenient shown for reference;
- per question: #correct of K -> all-wrong / mixed / all-right (only mixed groups
  give GRPO/RLOO a gradient);
- behaviour: tool calls, turns, stop reasons, tool errors; turns of correct rollouts
  (-> RL max-turn cap);
- dev: --dev questions, stratified by category, drawn from all questions so it keeps
  the natural difficulty mix; train: the remaining questions with mixed outcomes.

    python probe_summary.py --cands cands.parquet --work /root/probe --k 4 --out $OUTPUT_ROOT/probe
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
from pathlib import Path


def load_judge(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["case_id"]] = None if r.get("judge_failed") else int(r["acc"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    import pandas as pd

    cands = pd.read_parquet(args.cands, columns=["data_id", "category", "answer"])
    cat = dict(zip(cands.data_id, cands.category))
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_q = collections.defaultdict(list)
    for k in range(args.k):
        d = work / f"s{k}"
        strict = load_judge(d / "eval_report_probe_strict_details.jsonl")
        lenient = load_judge(d / "eval_report_probe_lenient_details.jsonl")
        for p in glob.glob(str(d / "*_trajectory.json")):
            t = json.load(open(p, encoding="utf-8"))
            cid = t["case_id"]
            turns = t.get("turns", [])
            tools = [json.loads(x["tool_call"]).get("name", "?") for x in turns if x.get("tool_call")]
            per_q[cid].append({
                "k": k, "strict": strict.get(cid), "lenient": lenient.get(cid),
                "stop": t.get("stop_reason", "?"), "tools": tools, "turns": len(turns),
                "tool_errors": sum("Tool execution error" in (x.get("tool_output") or "") for x in turns),
            })

    rows = []
    for cid, rs in per_q.items():
        judged = [r for r in rs if r["strict"] is not None]
        n_ok = sum(r["strict"] for r in judged)
        rows.append({
            "data_id": cid, "category": cat.get(cid, "?"), "samples": len(rs), "judged": len(judged),
            "correct": n_ok, "lenient_correct": sum(r["lenient"] or 0 for r in rs),
            "group": ("incomplete" if len(judged) < args.k else
                      "all_wrong" if n_ok == 0 else "all_right" if n_ok == args.k else "mixed"),
            "tool_calls": [len(r["tools"]) for r in rs], "stops": [r["stop"] for r in rs],
        })
    flat = [r for rs in per_q.values() for r in rs]
    judged = [r for r in flat if r["strict"] is not None]

    def pct(x: float) -> str:
        return f"{x:.1%}"

    def dist(xs) -> dict:
        return dict(sorted(collections.Counter(xs).items()))

    ok_turns = sorted(len(r["tools"]) for r in judged if r["strict"])
    report = {
        "questions": len(rows), "rollouts": len(flat), "judged": len(judged),
        "judge_failed": len(flat) - len(judged),
        "pass@1_strict": pct(sum(r["strict"] for r in judged) / max(1, len(judged))),
        "pass@1_lenient": pct(sum(r["lenient"] or 0 for r in flat) / max(1, len(flat))),
        f"pass@{args.k}_strict": pct(sum(r["correct"] > 0 for r in rows) / max(1, len(rows))),
        "correct_per_question": dist(r["correct"] for r in rows),
        "groups": dist(r["group"] for r in rows),
        "by_category": {
            c: {"n": len(g), "pass@1": pct(sum(r["correct"] for r in g) / max(1, args.k * len(g))),
                "groups": dist(r["group"] for r in g)}
            for c in sorted({r["category"] for r in rows}) for g in [[r for r in rows if r["category"] == c]]
        },
        "stop_reasons": dist(r["stop"] for r in flat),
        "tool_calls_per_rollout": dist(min(len(r["tools"]), 10) for r in flat),
        "acc_by_tool_calls": {
            n: pct(sum(r["strict"] for r in g) / len(g))
            for n in range(0, 11) for g in [[r for r in judged if min(len(r["tools"]), 10) == n]] if g
        },
        "tool_usage": dict(collections.Counter(t for r in flat for t in r["tools"]).most_common()),
        "tool_error_rate": pct(sum(r["tool_errors"] for r in flat) / max(1, sum(len(r["tools"]) for r in flat))),
        "tool_calls_of_correct_p50_p90_p95": [ok_turns[int(q * (len(ok_turns) - 1))] for q in (0.5, 0.9, 0.95)]
        if ok_turns else [],
    }
    # dev: stratified by category over all complete questions; train: remaining mixed groups
    rng = random.Random(args.seed)
    complete = [r for r in rows if r["group"] != "incomplete"]
    by_cat = collections.defaultdict(list)
    for r in complete:
        by_cat[r["category"]].append(r)
    dev = []
    for c, g in by_cat.items():
        rng.shuffle(g)
        dev += g[: round(args.dev * len(g) / max(1, len(complete)))]
    dev_ids = {r["data_id"] for r in dev}
    train = [r for r in complete if r["data_id"] not in dev_ids and r["group"] == "mixed"]
    report["dev"] = {"n": len(dev), "groups": dist(r["group"] for r in dev),
                     "pass@1": pct(sum(r["correct"] for r in dev) / max(1, args.k * len(dev)))}
    report["train_mixed"] = {"n": len(train), "by_category": dist(r["category"] for r in train)}
    report["held_back"] = dist(r["group"] for r in complete if r["data_id"] not in dev_ids and r["group"] != "mixed")

    json.dump(report, open(out / "probe_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(out / "probe_questions.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, sel in (("dev", dev), ("train", train)):
        ids = [r["data_id"] for r in sel]
        (out / f"{name}_ids.txt").write_text("\n".join(ids) + "\n")
        full = pd.read_parquet(args.cands)
        full[full.data_id.isin(ids)].to_parquet(out / f"{name}.parquet")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if os.getenv("OUTPUT_ROOT"):
        json.dump({"pass1_strict": sum(r["strict"] for r in judged) / max(1, len(judged)),
                   "mixed_questions": sum(r["group"] == "mixed" for r in rows), "questions": len(rows)},
                  open(Path(os.environ["OUTPUT_ROOT"]) / "metrics.json", "w"))
    return 0 if judged else 1


if __name__ == "__main__":
    raise SystemExit(main())
