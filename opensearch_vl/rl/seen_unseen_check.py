#!/usr/bin/env python3
"""Did an RL run improve on the questions it trained on, and on same-distribution questions it never saw?

Splits the RL train set by the finished run's state.json (order[:pos] = drawn during training,
order[pos:] = never drawn), samples --per-split questions from each, and rolls out the base policy
and base + the RL LoRA with the training sampler (same temperature, judge, reward), --samples each.
Prints per-question paired accuracy differences for both splits.

Started through run_rl.sh (services + env), e.g.
    RL_ENTRY=rl/seen_unseen_check.py bash opensearch_vl/run_rl.sh seen_check \
        --rl-run /root/rl_runs/r3 --per-split 100 --samples 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pandas as pd  # noqa: E402

from rl import train_grpo as G  # noqa: E402


def paired(a, b):
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "base": statistics.mean(a), "rl": statistics.mean(b), "diff": m,
            "ci95": [m - 1.96 * se, m + 1.96 * se], "better": sum(x > 0 for x in d),
            "worse": sum(x < 0 for x in d), "tie": sum(x == 0 for x in d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)  # accepted from run_rl.sh, unused
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--rl-run", required=True, help="finished RL run dir (state.json, adapters/final)")
    ap.add_argument("--lora", default=None, help="default <rl-run>/adapters/final")
    ap.add_argument("--per-split", type=int, default=100)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--fatal-k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--splits", default="seen,unseen", help="which splits to roll out (same sampling of questions)")
    args = ap.parse_args()

    import logging

    logging.basicConfig(format="[%(asctime)s][%(levelname)5s] %(message)s", level=logging.INFO)
    for noisy in ("opensearch_infer", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    lora = os.path.realpath(args.lora or os.path.join(args.rl_run, "adapters", "final"))
    assert os.path.exists(os.path.join(lora, "adapter_config.json")), lora
    state = json.load(open(os.path.join(args.rl_run, "state.json")))
    assert state["epoch"] == 0, "split by order[:pos] assumes the run stayed in its first epoch"
    train = pd.read_parquet(args.train)
    order, pos = state["order"], state["pos"]
    rng = random.Random(args.seed)
    seen = rng.sample(order[:pos], min(args.per_split, pos))
    unseen = rng.sample(order[pos:], min(args.per_split, len(order) - pos))
    print(f"train {len(train)} q: drawn during RL {pos}, never drawn {len(order) - pos}; "
          f"checking {len(seen)} + {len(unseen)}, {args.samples} samples each, lora {lora}", flush=True)
    want = set(args.splits.split(","))
    seen = seen if "seen" in want else []
    unseen = unseen if "unseen" in want else []
    rows = {i: train.iloc[i] for i in seen + unseen}

    os.makedirs(args.run_dir, exist_ok=True)
    roller = G.Roller(argparse.Namespace(model=args.model, temperature=args.temperature,
                                         max_tokens=args.max_tokens, fatal_k=args.fatal_k,
                                         workers=args.workers))
    acc = {}
    for tag, path in (("base", None), ("rl", lora)):
        roller.runner.lora_path = path
        out = os.path.join(args.run_dir, tag)
        shutil.rmtree(out, ignore_errors=True)
        groups = roller.run([rows[i] for i in seen + unseen], args.samples, out)
        acc[tag] = {q: [r["acc"] for r in g if not r["judge_failed"]] for q, g in groups.items()}
        flat = [r for g in groups.values() for r in g]
        print(f"{tag}: {len(flat)} rollouts, acc {statistics.mean(r['acc'] for r in flat):.3f}, "
              f"tools {statistics.mean(r['n_tools'] for r in flat):.2f}, "
              f"judge_failed {sum(r['judge_failed'] for r in flat)}", flush=True)
    res = {"lora": lora, "samples": args.samples, "temperature": args.temperature,
           "per_question": {tag: acc[tag] for tag in acc}}
    for name, idx in (("seen", seen), ("unseen", unseen)):
        if not idx:
            continue
        ids = [train.iloc[i]["data_id"] for i in idx]
        ids = [q for q in ids if acc["base"].get(q) and acc["rl"].get(q)]
        res[name] = paired([statistics.mean(acc["base"][q]) for q in ids],
                           [statistics.mean(acc["rl"][q]) for q in ids])
        print(f"{name}: " + json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(os.environ.get("OUTPUT_ROOT", args.run_dir), "seen_unseen.json"), "w"), indent=2)
    print("CHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
