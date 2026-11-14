#!/usr/bin/env python3
"""Minimal behavioural check of rl/train_grpo.Policy.update() with a toy model.

Drives the REAL update() code on a tiny stub model and asserts the PPO mechanics:

1. adv == 0 samples are excluded from the gradient (train/seqs counts only adv != 0);
2. --minibatches optimizer steps happen per ppo-epoch;
3. the recorded old log-probs come from the PRE-update policy: mean_logp equals the
   mean token log-prob recomputed with a frozen copy of the pre-update weights —
   if update() recomputed old logp after a minibatch step (or left it attached so
   the optimizer updates leaked in), this equality breaks;
4. the loss divides the token-summed surrogate by (len(minibatch) * loss_norm_tokens)
   — doubling loss_norm_tokens halves the gradient norm;
5. encode-failed sequences are counted in train/skipped and excluded.

Run on the instance (needs torch only, CPU is fine):
    cd /workspace/opensearch_vl && python tests/test_rl_update_minibatch.py
"""
from __future__ import annotations
import argparse, copy, json, random, sys, types

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class ToyInner(nn.Module):
    """Mimics the interface token_logp() uses: inner.model(...) -> last_hidden_state, inner.lm_head."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.lm_head = nn.Linear(dim, vocab)

    def model(self, input_ids, attention_mask, use_cache=False, **mm):
        h, _ = self.gru(self.embed(input_ids))
        return types.SimpleNamespace(last_hidden_state=h)


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = ToyInner()  # registered submodule so .parameters() sees it
        self.base_model = types.SimpleNamespace(model=self.inner)


def make_policy(args):
    """Build a real Policy object bypassing __init__ (no HF/PEFT loading), then fill fields."""
    from rl import train_grpo as G

    p = object.__new__(G.Policy)
    p.torch = torch
    p.args = args
    toy = ToyPolicy()
    p.model = toy            # only used by save()/train(), not by update()
    p.inner = toy.base_model.model
    p.opt = torch.optim.SGD([*toy.parameters()], lr=args.lr)
    p.rng = random.Random(0)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--loss-norm-tokens", type=float, default=64.0)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    args = ap.parse_args()
    args.temperature = 0.7
    args.logit_chunk = 2048
    args.max_seq_len = 1000
    args.grad_clip = 1.0

    torch.manual_seed(0)
    policy = make_policy(args)
    frozen = copy.deepcopy(policy.inner.state_dict())  # pre-update weights

    def sample(ids, adv):
        d = {"fatal_turn": None, "adv": adv,
             "ids": torch.tensor([ids]), "mask": torch.tensor([0] + [1] * (len(ids) - 1)), "mm": {}}
        d["contents"] = d  # update() passes s["contents"] to encode(); the stub reads ids from it
        return d

    def fake_encode(contents, fatal_turn):
        s = contents
        if s["ids"][0, -1].item() == 15:      # stand-in for "too long / no trainable tokens"
            return None
        return s["ids"], s["mask"], s["mm"]

    def lp_frozen(ids, mask):
        """Token logprobs under the frozen pre-update policy, exactly as token_logp slices them."""
        with torch.no_grad():
            h = policy.inner.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
            tgt = mask[1:].nonzero(as_tuple=True)[0] + 1
            logits = policy.inner.lm_head(h[tgt - 1]).float() / args.temperature
            return torch.log_softmax(logits, -1).gather(-1, ids[0, tgt][:, None]).squeeze(-1)

    s_pos = sample([1, 2, 3, 4, 5, 6], +1.0)
    s_neg = sample([7, 8, 9, 10, 11, 12], -1.0)
    s_zero = sample([13, 14, 15], 0.0)
    samples = [s_pos, s_neg, s_zero]

    policy.encode = fake_encode
    info = policy.update(samples)

    ok = True
    if info["train/seqs"] != 2:
        print(f"FAIL 1: train/seqs={info['train/seqs']}, expected 2 (zero-adv sample must be dropped)")
        ok = False
    if info["train/opt_steps"] != args.minibatches:
        print(f"FAIL 2: opt_steps={info['train/opt_steps']}, expected {args.minibatches}")
        ok = False

    # old logp must be the PRE-update policy's: compare with frozen weights.
    # (recompute on the CURRENT module would differ; so re-point a fresh module at the
    #  frozen state dict and reuse lp_frozen's math)
    saved = copy.deepcopy(policy.inner.state_dict())
    policy.inner.load_state_dict(frozen)
    frozen_mean = float(torch.cat([lp_frozen(s["ids"], s["mask"]) for s in (s_pos, s_neg)]).mean())
    policy.inner.load_state_dict(saved)
    if abs(frozen_mean - info["train/mean_logp"]) > 1e-5:
        print(f"FAIL 3: mean_logp {info['train/mean_logp']:.6f} != frozen pre-update mean {frozen_mean:.6f}"
              " — old logp was not taken from the pre-update policy")
        ok = False
    else:
        print(f"OK 3: mean_logp matches the pre-update policy ({frozen_mean:.6f})")

    args2 = argparse.Namespace(**vars(args)); args2.loss_norm_tokens = args.loss_norm_tokens * 2
    p2 = make_policy(args2)
    p2.inner.load_state_dict(frozen)
    p2.encode = fake_encode
    info2 = p2.update([s_pos, s_neg])
    ratio = info2["train/grad_norm"] / max(info["train/grad_norm"], 1e-12)
    if not (0.45 < ratio < 0.55):
        print(f"FAIL 4: grad-norm ratio with 2x loss-norm-tokens is {ratio:.3f}, expected ~0.5")
        ok = False
    else:
        print(f"OK 4: grad-norm ratio {ratio:.3f} with 2x loss-norm-tokens")

    if not (0.0 <= info["train/clip_frac"] <= 1.0):
        print(f"FAIL 5: clip_frac {info['train/clip_frac']} out of range")
        ok = False

    s_bad = sample([21, 22, 15], +0.5)  # ends in 15 -> fake_encode returns None
    info3 = policy.update([s_pos, s_neg, s_bad])
    if info3["train/skipped"] != 1:
        print(f"FAIL 6: train/skipped={info3['train/skipped']}, expected 1")
        ok = False
    else:
        print("OK 6: encode-failed sample counted in train/skipped and excluded")

    print("update info:", json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                       for k, v in info.items()}))
    print("MBATCH_OK" if ok else "MBATCH_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
