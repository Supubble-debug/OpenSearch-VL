#!/usr/bin/env python3
"""Offline dry run of the RL update (no vLLM, no paid API): PPO minibatches + fp32 KL probe.

Takes one logged RL step (rollout json + replay rows with acc/hit/fmt), recomputes rewards and
RLOO advantages exactly as the trainer does, rebuilds the trajectories (text observations only,
cut before tools that returned images) and runs ``Policy.update`` for --steps steps on that same
batch, printing the trainer metrics and the fp32 KL vs the SFT model after each step. Checks the
mechanics, memory and time of the new update; the KL it reports is for repeated updates on one
batch, so it is an upper-bound-ish sanity number, not a forecast.

    python tests/rl_update_dryrun.py --model /root/models/qwen3vl4b-sft-v2-merged \
        --data /root/rl_data/rl_train.parquet --replay /root/rl_runs/replay_r2.json \
        --step-dir /root/rl_runs/r2/rollouts/step_0001 --steps 2 --run-dir /root/rl_diag/dryrun
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--run-dir", default="/root/rl_diag/dryrun")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--kl-probe-n", type=int, default=8)
    args_cli = ap.parse_args()

    import pandas as pd

    from opensearch_infer import config
    from opensearch_infer.runners import Qwen35VLLMRunner
    from rl import reward as R
    from rl import train_grpo as G
    from rl.kl_probe import KLProbe, load_reference_trajectories, rebuild

    sys.argv = ["train_grpo", "--model", args_cli.model, "--train", args_cli.data, "--run-dir", args_cli.run_dir,
                "--metrics", os.path.join(args_cli.run_dir, "m.jsonl"), "--lr", str(args_cli.lr),
                "--minibatches", str(args_cli.minibatches), "--ppo-epochs", str(args_cli.ppo_epochs)]
    args = G.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    rn = Qwen35VLLMRunner(config.MODEL_REGISTRY["qwen3-vl-4b-vllm"], args.model)
    rn.load_prompt_stack()
    policy = G.Policy(args, types.SimpleNamespace(runner=rn))

    rows = {r["data_id"]: r for r in pd.read_parquet(args.train).to_dict("records")}
    step_dir = os.path.normpath(args_cli.step_dir)
    groups = collections.defaultdict(list)
    for x in json.load(open(args_cli.replay))["rows"]:
        if os.path.normpath(os.path.dirname(os.path.dirname(x["path"]))) != step_dir:
            continue
        sc = R.combine(x["stop"], x["fmt"], int(x["acc"]), int(x["hit"]), bool(x["short"]))
        groups[x["data_id"]].append((x, sc["reward"]))
    samples = []
    for did, g in groups.items():
        rewards = [r for _, r in g]
        if len(g) < 2 or statistics.pstdev(rewards) < 1e-6 or did not in rows:
            continue
        adv = R.advantages(rewards, [x["stop"] == "fatal" for x, _ in g])
        for (x, _), a in zip(g, adv):
            traj = json.load(open(x["path"], encoding="utf-8"))
            samples.append({"contents": rebuild(traj, rows[did], rn.prompt_in_system), "fatal_turn": None, "adv": a})
    print(f"groups with signal {len({d for d, g in groups.items()})} -> samples {len(samples)}", flush=True)

    refs = load_reference_trajectories([args_cli.step_dir], [args.train], args_cli.kl_probe_n, 0)
    probe = KLProbe(args.model, rn, policy.lora_config, refs, args.temperature, args.kl_probe_max_len)
    print(f"probe trajectories {len(probe.items)}", flush=True)
    for step in range(1, args_cli.steps + 1):
        t0 = time.time()
        info = policy.update(samples)
        info["time/train_min"] = (time.time() - t0) / 60
        t1 = time.time()
        info.update(probe.measure(policy.model) or {})
        info["time/kl_probe_min"] = (time.time() - t1) / 60
        print(f"step {step}: " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items()}),
              flush=True)
    print("DRYRUN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
