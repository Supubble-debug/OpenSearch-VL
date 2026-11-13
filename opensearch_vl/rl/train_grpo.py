#!/usr/bin/env python3
"""Single-node agentic GRPO/RLOO with LoRA for the Qwen3-VL-4B search agent.

Rollouts run the exact agent/tool stack of the probes (opensearch_infer.pipeline, vLLM token
server with per-request LoRA, local 27B summariser, DeepSeek search, Doubao image search);
the policy update runs in this process with HF transformers + PEFT on the same GPU.

Per step: draw --prompts-per-step questions, --n rollouts each; score (rl/reward.py); drop
groups whose rewards are all equal (no signal) and draw more questions until enough groups
have signal (dynamic sampling, at most --max-extra-prompts extra); RLOO advantages with
fatal clamping (A >= 0 for trajectories stopped by K consecutive tool errors, post-fatal
tokens masked); PPO-clip update over --ppo-epochs x --minibatches optimizer steps (the first
minibatch is on-policy, ratio = 1); token-level loss summed and divided by a constant
(--loss-norm-tokens per sequence; no per-sequence length normalisation, which down-weights
every token of long trajectories); save the LoRA; vLLM picks it up by path. An optional fp32
KL probe (rl/kl_probe.py) measures the drift from the SFT policy every step and stops the
run past --kl-stop.

Training sequences are rendered with LLaMA-Factory's template exactly as in SFT
(Qwen35SFTRunner prompt stack); only assistant turns before the fatal step are trained.

    python rl/train_grpo.py --model /root/models/qwen3vl4b-sft-v2-merged \
        --train /root/rl_data/rl_train.parquet --dev /root/rl_data/rl_dev.parquet \
        --run-dir /root/rl_runs/v2_r1 --metrics $OUTPUT_ROOT/metrics.jsonl
Resumes from <run-dir>/state.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # opensearch_vl/

import pandas as pd  # noqa: E402

import eval_with_gpt4o as judge  # noqa: E402
from opensearch_infer import config, pipeline  # noqa: E402
from opensearch_infer.runners import InferenceConfig, build_runner  # noqa: E402
from opensearch_infer.search import FatalServiceError  # noqa: E402
from rl import reward as R  # noqa: E402

log = logging.getLogger("grpo")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged SFT checkpoint (vLLM serves the same path)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--metrics", required=True, help="jsonl of per-step metrics")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--max-extra-prompts", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ppo-epochs", type=int, default=1, help="passes over each step's samples")
    ap.add_argument("--minibatches", type=int, default=2, help="optimizer steps per pass")
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--loss-norm-tokens", type=float, default=1024.0,
                    help="loss = -sum_tokens(surrogate) / (seqs_in_minibatch * this)")
    ap.add_argument("--kl-probe-rollouts", nargs="*", default=None,
                    help="dirs of reference rollouts (s*/<id>_trajectory.json) sampled by the SFT policy")
    ap.add_argument("--kl-probe-n", type=int, default=8)
    ap.add_argument("--kl-probe-max-len", type=int, default=8000)
    ap.add_argument("--kl-stop", type=float, default=0.05,
                    help="stop (clean, final dev eval) when fp32 per-token KL vs SFT exceeds this; 0 = never")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--fatal-k", type=int, default=3)
    ap.add_argument("--estimator", choices=["rloo", "grpo"], default="rloo")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-at-start", action="store_true")
    ap.add_argument("--dev-n", type=int, default=4)
    ap.add_argument("--dev-limit", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0, help="smoke test: use the first N questions")
    ap.add_argument("--max-seq-len", type=int, default=36000)
    ap.add_argument("--logit-chunk", type=int, default=2048)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--keep-adapters", type=int, default=3, help="keep the last N step adapters (+ every 5th)")
    ap.add_argument("--w-acc", type=float, default=0.8)
    ap.add_argument("--w-query", type=float, default=0.2)
    ap.add_argument("--min-balance", type=float, default=10.0,
                    help="DeepSeek credit (CNY) below which the run stops before the next step")
    ap.add_argument("--max-judge-fail-rate", type=float, default=0.25,
                    help="stop if more than this share of rollouts had an unanswered judge call")
    ap.add_argument("--seed", type=int, default=20261113)
    return ap.parse_args()


# ----------------------------------------------------------------------------------------- utils


class Metrics:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, step: int, values: Dict[str, Any]) -> None:
        rec = {"step": step, "time": time.time(), **values}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("step %d %s", step, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in values.items()}, ensure_ascii=False))


class BudgetExhausted(RuntimeError):
    """Paid API budget or quota is (nearly) gone: stop before burning more rollouts."""


def deepseek_balance() -> float:
    """Remaining DeepSeek credit in CNY (the same key pays for text search and the judge)."""
    import requests

    r = requests.get("https://api.deepseek.com/user/balance", timeout=30,
                     headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
    r.raise_for_status()
    return float(sum(float(b["total_balance"]) for b in r.json()["balance_infos"]))


def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


# -------------------------------------------------------------------------------------- rollouts


class Roller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runner = build_runner(model_name="qwen3-vl-4b-vllm", checkpoint=args.model)
        self.runner.load()
        self.cfg = InferenceConfig(temperature=args.temperature, max_tokens=args.max_tokens)
        judge.STRICT = True
        self.stop = threading.Event()

    def one(self, row: Any, idx: int, k: int, outdir: str) -> Dict[str, Any]:
        if self.stop.is_set():
            return {"skipped": True}
        try:
            traj = pipeline.process_single_case(
                row=row, runner=self.runner, output_dir=os.path.join(outdir, f"s{k}"), case_idx=idx,
                dataset_type="train", inference_cfg=self.cfg, fatal_consecutive_errors=self.args.fatal_k)
        except FatalServiceError:
            self.stop.set()
            raise
        stop = traj.get("stop_reason", "")
        turns = traj.get("turns", [])
        fatal_turn = traj.get("fatal_turn")
        raw = row.get("aliases")
        aliases = [] if raw is None else [str(x) for x in list(raw)]  # parquet gives a numpy array
        gold = str(row["answer"])
        fmt = R.format_score(turns, stop, fatal_turn)
        acc, judge_failed = 0, False
        if stop == "response":
            question = row["prompt"][0]["content"]
            final = "\n\n".join(t.get("response_text", "") for t in turns if t.get("response_text"))
            res = judge.judge_common(question, R.judge_gold(gold, aliases), judge.extract_final_answer(final))
            acc, judge_failed = int(res["acc"]), bool(res.get("judge_failed"))
        hit = R.retrieval_hit(turns, gold, aliases)
        score = R.combine(stop, fmt, acc, hit, R.is_short_gold(gold))
        return {"data_id": row["data_id"], "k": k, "stop": stop, "fatal": stop == "fatal",
                "fatal_turn": fatal_turn, "contents": traj.get("contents"), "n_turns": len(turns),
                "n_tools": sum(1 for t in turns if t.get("tool_call")),
                "judge_failed": judge_failed, **score}

    def run(self, rows: List[Any], n: int, outdir: str) -> Dict[str, List[Dict[str, Any]]]:
        jobs = [(row, i, k) for i, row in enumerate(rows) for k in range(n)]
        groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        with ThreadPoolExecutor(self.args.workers) as ex:
            futs = [ex.submit(self.one, row, i, k, outdir) for row, i, k in jobs]
            for f in futs:
                r = f.result()  # re-raises FatalServiceError
                if not r.get("skipped"):
                    groups[r["data_id"]].append(r)
        return groups


def summarize(groups: Dict[str, List[Dict[str, Any]]], prefix: str) -> Dict[str, Any]:
    flat = [r for g in groups.values() for r in g]
    if not flat:
        return {}
    stops = collections.Counter(r["stop"] for r in flat)
    out = {f"{prefix}/{k}": mean([r[k] for r in flat]) for k in ("reward", "acc", "query", "hit", "fmt", "answered")}
    out.update({
        f"{prefix}/pass_at_n": mean([float(any(r["acc"] for r in g)) for g in groups.values()]),
        f"{prefix}/tool_calls": mean([r["n_tools"] for r in flat]),
        f"{prefix}/fatal_rate": stops.get("fatal", 0) / len(flat),
        f"{prefix}/max_turns_rate": stops.get("max_turns", 0) / len(flat),
        f"{prefix}/judge_failed": sum(r["judge_failed"] for r in flat),
        f"{prefix}/rollouts": len(flat),
        f"{prefix}/stops": dict(stops),
    })
    return out


# ---------------------------------------------------------------------------------------- policy


class Policy:
    def __init__(self, args: argparse.Namespace, roller: Roller) -> None:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForImageTextToText

        self.torch = torch
        self.args = args
        self.rn = roller.runner  # prompt stack: tokenizer, processor, LLaMA-Factory template
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            device_map={"": "cuda:0"})
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        resume = os.path.join(args.run_dir, "adapters", "latest")
        if os.path.exists(os.path.join(resume, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, resume, is_trainable=True)
            log.info("resumed adapter %s", resume)
        else:
            model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                                     target_modules=LORA_TARGETS, bias="none"))
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert names and all("visual" not in n for n in names), "LoRA must only touch the language model"
        log.info("trainable params %.1fM in %d tensors", n_train / 1e6, len(names))
        self.model = model
        self.inner = model.base_model.model  # Qwen3VLForConditionalGeneration
        self.rng = random.Random(args.seed)
        self.lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                      target_modules=LORA_TARGETS, bias="none")
        self.opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                     betas=(0.9, 0.999), weight_decay=0.0)
        opt_path = os.path.join(resume, "optimizer.pt")
        if os.path.exists(opt_path):
            self.opt.load_state_dict(torch.load(opt_path, map_location="cuda:0"))

    def encode(self, contents: List[Dict[str, Any]], fatal_turn: Any):
        """SFT-identical rendering -> (input_ids, loss mask, mm inputs) or None if too long/empty."""
        torch = self.torch
        rn = self.rn
        msgs, images = rn.contents_to_template(contents)
        while msgs and msgs[-1]["role"] != "assistant":  # trailing observation(s) are not trained
            dropped = msgs.pop()
            n_img = dropped["content"].count("<image>")
            if n_img:
                images = images[:-n_img]
        if not msgs:
            return None
        rendered = rn._template.mm_plugin.process_messages(msgs, images, [], [], rn._processor)
        pairs = rn._template.encode_multiturn(rn._tokenizer, rendered, rn._system, rn._tools)
        ids: List[int] = []
        mask: List[int] = []
        for turn, (p, r) in enumerate(pairs):
            trainable = fatal_turn is None or turn < fatal_turn
            ids += p + r
            mask += [0] * len(p) + [int(trainable)] * len(r)
        if not any(mask) or len(ids) > self.args.max_seq_len:
            return None
        mm = rn._template.mm_plugin._get_mm_inputs(images, [], [], rn._processor) if images else {}
        mm = {k: (v.to("cuda:0").to(torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda:0"))
              for k, v in mm.items() if isinstance(v, torch.Tensor)}
        return torch.tensor([ids], device="cuda:0"), torch.tensor(mask, device="cuda:0"), mm

    def token_logp(self, ids, mask, mm):
        """Log-probs of masked target tokens under the *sampling* policy.

        The rollout policy samples with temperature T (top_p=1, top_k off), i.e.
        p(x) ∝ exp(logits / T), so the log-probs the objective must use are
        log_softmax(logits / T). Computing log_softmax(logits) instead (T=1) is a
        different policy: the gradient then pushes tokens the T=0.7 policy would
        under-sample. ``--temperature`` is the same value the runner samples with.
        """
        torch = self.torch
        from torch.utils.checkpoint import checkpoint

        out = self.inner.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **mm)
        h = out.last_hidden_state[0]
        tgt = mask[1:].nonzero(as_tuple=True)[0] + 1  # positions whose token is trained
        hs, ys = h[tgt - 1], ids[0, tgt]
        head = self.inner.lm_head

        temp = max(float(self.args.temperature), 1e-6)

        def chunk(hc, yc):
            logits = head(hc).float() / temp
            return torch.log_softmax(logits, -1).gather(-1, yc[:, None]).squeeze(-1)

        c = self.args.logit_chunk
        return torch.cat([checkpoint(chunk, hs[i:i + c], ys[i:i + c], use_reentrant=False)
                          for i in range(0, len(ys), c)])

    def update(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """PPO-clip over this step's samples.

        old log-probs come from a no-grad pass before any optimizer step, so the first minibatch
        is exactly on-policy (ratio = 1) and later ones are importance-weighted and clipped.
        Per minibatch: loss = -sum_{seq, token} min(r*A, clip(r)*A) / (n_seq * loss_norm_tokens).
        """
        torch = self.torch
        args = self.args
        work = [s for s in samples if s["adv"] != 0.0]
        old, skipped = {}, 0
        self.model.train()
        with torch.no_grad():
            for i, s in enumerate(work):
                enc = self.encode(s["contents"], s["fatal_turn"])
                if enc is None:
                    skipped += 1
                    continue
                try:
                    old[i] = self.token_logp(*enc).detach().float().cpu()
                except torch.cuda.OutOfMemoryError:
                    log.warning("OOM on a %d-token sequence (old logp), skipped", enc[0].shape[1])
                    torch.cuda.empty_cache()
                    skipped += 1
                del enc
        idx = list(old)
        tokens = int(sum(len(old[i]) for i in idx))
        gnorms, clipfr, kl_approx, n_opt = [], [], [], 0
        for _ in range(args.ppo_epochs):
            self.rng.shuffle(idx)
            k = max(1, math.ceil(len(idx) / max(1, args.minibatches)))
            for mb in (idx[j:j + k] for j in range(0, len(idx), k)):
                self.opt.zero_grad(set_to_none=True)
                done = 0
                for i in mb:
                    s = work[i]
                    ids, mask, mm = self.encode(s["contents"], s["fatal_turn"])
                    try:
                        lp = self.token_logp(ids, mask, mm)
                        lp_old = old[i].to(lp.device)
                        assert lp.shape == lp_old.shape, "re-encoding changed the token count"
                        ratio = torch.exp(lp - lp_old)
                        adv = float(s["adv"])
                        surr = torch.minimum(ratio * adv, ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * adv)
                        loss = -surr.sum() / (len(mb) * args.loss_norm_tokens)
                        loss.backward()
                    except torch.cuda.OutOfMemoryError:
                        log.warning("OOM on a %d-token sequence, skipped", ids.shape[1])
                        torch.cuda.empty_cache()
                        continue
                    with torch.no_grad():
                        clipfr.append(float(((ratio - 1).abs() > args.clip_eps).float().mean()))
                        kl_approx.append(float((lp_old - lp).mean()))
                    done += 1
                    del ids, mask, mm, lp, lp_old, ratio, surr, loss
                if not done:
                    continue
                params = [p for p in self.model.parameters() if p.requires_grad]
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
                gnorms.append(gnorm)
                if math.isfinite(gnorm):
                    self.opt.step()
                    n_opt += 1
        self.opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"train/seqs": len(idx), "train/skipped": skipped, "train/tokens": tokens,
                "train/opt_steps": n_opt, "train/grad_norm": mean(gnorms), "train/grad_norm_max": max(gnorms or [0.0]),
                "train/mean_logp": mean([float(old[i].mean()) for i in idx]),
                "train/clip_frac": mean(clipfr), "train/approx_kl_bf16": mean(kl_approx),
                "train/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}

    def save(self, step: int) -> str:
        d = os.path.join(self.args.run_dir, "adapters", f"step_{step:04d}")
        self.model.save_pretrained(d)
        self.torch.save(self.opt.state_dict(), os.path.join(d, "optimizer.pt"))
        latest = os.path.join(self.args.run_dir, "adapters", "latest")
        tmp = latest + ".tmp"
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(d, tmp)
        os.replace(tmp, latest)
        return d


def prune_adapters(run_dir: str, keep: int, step: int) -> None:
    root = os.path.join(run_dir, "adapters")
    steps = sorted(int(n.split("_")[1]) for n in os.listdir(root) if n.startswith("step_"))
    for s in steps[:-keep]:
        if s % 5 and s != step:
            shutil.rmtree(os.path.join(root, f"step_{s:04d}"), ignore_errors=True)


# ------------------------------------------------------------------------------------------ main


def main() -> int:
    args = parse_args()
    logging.basicConfig(format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s", level=logging.INFO)
    for noisy in ("opensearch_infer", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if config.MAX_TURNS != args.max_turns:
        raise SystemExit(f"export AGENT_MAX_TURNS={args.max_turns} (config has {config.MAX_TURNS})")
    os.makedirs(args.run_dir, exist_ok=True)
    random.seed(args.seed)

    train = pd.read_parquet(args.train)
    if args.train_limit:
        train = train.head(args.train_limit)
    dev = pd.read_parquet(args.dev) if args.dev else None
    if dev is not None and args.dev_limit:
        dev = dev.head(args.dev_limit)
    log.info("train %d q, dev %d q", len(train), 0 if dev is None else len(dev))

    R.set_weights(args.w_acc, args.w_query)
    log.info("reward weights acc=%.2f query=%.2f", R.W_ACC, R.W_QUERY)
    roller = Roller(args)
    policy = Policy(args, roller)
    metrics = Metrics(args.metrics)
    probe = None
    if args.kl_probe_rollouts and args.kl_probe_n > 0:
        from rl.kl_probe import KLProbe, load_reference_trajectories

        refs = load_reference_trajectories(args.kl_probe_rollouts, [args.train], args.kl_probe_n, args.seed)
        probe = KLProbe(args.model, roller.runner, policy.lora_config, refs, args.temperature, args.kl_probe_max_len)
        log.info("fp32 KL probe: %d reference trajectories", len(probe.items))

    state_path = os.path.join(args.run_dir, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"step": 0, "epoch": 0, "pos": 0, "order": []}
    if state["step"] > 0:
        roller.runner.lora_path = os.path.realpath(os.path.join(args.run_dir, "adapters", "latest"))
    rng = random.Random(args.seed + state["epoch"])
    if not state["order"]:
        state["order"] = rng.sample(range(len(train)), len(train))
    total_q = int(len(train) * args.epochs)
    consumed = state.get("consumed", 0)

    def next_rows(k: int) -> List[Any]:
        nonlocal consumed
        rows = []
        while len(rows) < k and consumed < total_q:
            if state["pos"] >= len(state["order"]):
                state["epoch"] += 1
                state["pos"] = 0
                state["order"] = random.Random(args.seed + state["epoch"]).sample(range(len(train)), len(train))
            rows.append(train.iloc[state["order"][state["pos"]]])
            state["pos"] += 1
            consumed += 1
        return rows

    def evaluate(step: int) -> None:
        if dev is None:
            return
        t0 = time.time()
        rows = [dev.iloc[i] for i in range(len(dev))]
        groups = roller.run(rows, args.dev_n, os.path.join(args.run_dir, "dev", f"step_{step:04d}"))
        metrics.write(step, {**summarize(groups, "dev"), "dev/time_min": (time.time() - t0) / 60})

    if args.eval_at_start and state["step"] == 0 and not state.get("evaluated_start"):
        evaluate(0)
        state["evaluated_start"] = True
        json.dump(state, open(state_path, "w"))

    while True:
        step = state["step"] + 1
        if args.max_steps and step > args.max_steps:
            break
        t0 = time.time()
        try:
            balance = deepseek_balance()
        except Exception as exc:  # balance API hiccup must not kill training
            log.warning("balance check failed: %s", exc)
            balance = float("nan")
        if balance == balance and balance < args.min_balance:
            print(f"RL_STOPPED: DeepSeek balance {balance:.2f} < --min-balance {args.min_balance}", flush=True)
            raise BudgetExhausted(f"DeepSeek balance {balance:.2f}")
        rows = next_rows(args.prompts_per_step)
        if not rows:
            break
        outdir = os.path.join(args.run_dir, "rollouts", f"step_{step:04d}")
        groups = roller.run(rows, args.n, outdir)
        extra = 0
        while True:
            kept = {q: g for q, g in groups.items()
                    if len(g) >= 2 and not any(r["judge_failed"] for r in g)
                    and statistics.pstdev([r["reward"] for r in g]) > 1e-6}
            need = args.prompts_per_step - len(kept)
            if need <= 0 or extra >= args.max_extra_prompts:
                break
            more = next_rows(min(need, args.max_extra_prompts - extra))
            if not more:
                break
            extra += len(more)
            groups.update(roller.run(more, args.n, outdir))
        t_roll = time.time() - t0
        samples = []
        for g in kept.values():
            adv = R.advantages([r["reward"] for r in g], [r["fatal"] for r in g], args.estimator)
            for r, a in zip(g, adv):
                samples.append({**r, "adv": a})
        info = summarize(groups, "rollout")
        info["deepseek_balance"] = balance
        skipped_by_stop = collections.Counter(s["stop"] for s in samples if s["adv"] == 0.0)
        info.update({
            "train/adv_pos": sum(1 for s in samples if s["adv"] > 0),
            "train/adv_neg": sum(1 for s in samples if s["adv"] < 0),
            "train/adv_zero_by_stop": dict(skipped_by_stop),
            "train/fatal_in_samples": sum(1 for s in samples if s["fatal"]),
        })
        flat = [r for g in groups.values() for r in g]
        fail_rate = (sum(r["judge_failed"] for r in flat) / len(flat)) if flat else 0.0
        if fail_rate > args.max_judge_fail_rate:
            print(f"RL_STOPPED: judge unanswered on {fail_rate:.0%} of rollouts (balance {balance:.2f})", flush=True)
            json.dump(state, open(state_path, "w"))
            raise BudgetExhausted(f"judge failed on {fail_rate:.0%} of rollouts")
        info.update({"groups/total": len(groups), "groups/kept": len(kept), "groups/extra_prompts": extra,
                     "time/rollout_min": t_roll / 60, "epoch_progress": consumed / max(1, total_q)})
        if samples:
            t1 = time.time()
            info.update(policy.update(samples))
            info["time/train_min"] = (time.time() - t1) / 60
        kl_hit = False
        if probe is not None:
            t2 = time.time()
            info.update(probe.measure(policy.model) or {})
            info["time/kl_probe_min"] = (time.time() - t2) / 60
            kl = info.get("kl_probe/kl_per_token")
            kl_hit = bool(args.kl_stop and kl is not None and kl > args.kl_stop)
        path = policy.save(step)
        roller.runner.lora_path = os.path.realpath(path)
        prune_adapters(args.run_dir, args.keep_adapters, step)
        state.update({"step": step, "consumed": consumed})
        json.dump(state, open(state_path, "w"))
        metrics.write(step, info)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(step)
        if kl_hit:
            print(f"RL_KL_STOP: fp32 KL {info['kl_probe/kl_per_token']:.4g} > --kl-stop {args.kl_stop}", flush=True)
            break
    if dev is not None and (not args.eval_every or state["step"] % args.eval_every):
        evaluate(state["step"])
    final = os.path.join(args.run_dir, "adapters", "final")
    if os.path.exists(os.path.join(args.run_dir, "adapters", "latest")):
        shutil.copytree(os.path.realpath(os.path.join(args.run_dir, "adapters", "latest")), final, dirs_exist_ok=True)
    print("RL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BudgetExhausted as exc:  # clean, loud stop: the patrol reports RL_STOPPED
        log.error("stopping: %s", exc)
        raise SystemExit(3)
