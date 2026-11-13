"""RL reward (agreed 2026-11-12, after OpenSearch-VL arXiv 2605.05185 Eq. 9).

    r = r_fmt * (0.8 * r_acc + 0.2 * r_query)        if the agent gave a <response>
    r = 0                                             otherwise (max turns, fatal, no answer)

- r_fmt: mean over assistant steps before the fatal step of a 0/1 step score. A step is valid if
  it has <think>...</think> and then either one parseable JSON <tool_call> whose tool did not
  return an execution error (non-final steps) or a <response>...</response> block (final step).
  Short text between </think> and the call/response is allowed: the SFT data does this in ~25%
  of steps, and the paper's "immediately followed" rule would penalise the SFT style itself.
- r_acc: strict DeepSeek judge (eval_with_gpt4o.JUDGE_PROMPT_STRICT) against gold + aliases.
- r_query: retrieval hit = every part of the gold (or of a full alias) appears in the outputs of
  search / OCR tools; max(hit, acc) so a correct answer is never penalised; = r_acc for short
  golds (yes/no, <4 chars, small numbers) where substring hits are meaningless.
- fatal (K consecutive tool errors): post-fatal tokens masked, advantage clamped at >= 0 (trainer).
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

W_ACC, W_QUERY = 0.8, 0.2


def set_weights(w_acc: float, w_query: float) -> None:
    """Reward weights as a run-level setting (the ablation in 2026-11-13 needs 1.0 / 0.0)."""
    global W_ACC, W_QUERY
    assert abs(w_acc + w_query - 1.0) < 1e-9, "weights must sum to 1"
    W_ACC, W_QUERY = float(w_acc), float(w_query)
SEARCH_TOOLS = {"text_search", "image_search", "web_search", "search", "layout_parsing"}
_THINK = re.compile(r"^\s*<think>.*?</think>(.*)$", re.S)
_CALL = re.compile(r"<tool_call>\s*(\{.*\})\s*</tool_call>\s*$", re.S)
_RESP = re.compile(r"<response>.*</response>\s*$", re.S)
_YESNO = {"yes", "no", "是", "否", "true", "false"}


def norm(text: Any) -> str:
    s = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in s if ch.isalnum())


def parts(answer: str) -> List[str]:
    return [p for p in (norm(x) for x in re.split(r"[;；]", answer or "")) if p]


def judge_gold(gold: str, aliases: List[str]) -> str:
    aliases = [a for a in aliases or [] if a and str(a).strip()]
    return gold + (f" (also acceptable: {'; '.join(aliases)})" if aliases else "")


def is_short_gold(gold: str) -> bool:
    p = parts(gold)
    total = "".join(p)
    return (not p) or total in _YESNO or len(total) < 4 or (total.isdigit() and len(total) <= 4)


def step_valid(text: str, final: bool, tool_error: bool = False) -> bool:
    m = _THINK.match(text or "")
    if not m:
        return False
    rest = m.group(1).strip()
    if final:
        return bool(_RESP.search(rest))
    if tool_error:
        return False
    c = _CALL.search(rest)
    if not c or rest.count("<tool_call>") != 1:
        return False
    try:
        return bool(json.loads(c.group(1)).get("name"))
    except (ValueError, AttributeError):
        return False


def format_score(turns: List[Dict[str, Any]], stop: str, fatal_turn: Optional[int]) -> float:
    upto = len(turns) if fatal_turn is None else fatal_turn
    scores = []
    for i, t in enumerate(turns[:upto]):
        final = stop == "response" and i == len(turns) - 1
        scores.append(float(step_valid(t.get("response_text", ""), final, bool(t.get("tool_error")))))
    return sum(scores) / len(scores) if scores else 0.0


def retrieval_hit(turns: List[Dict[str, Any]], gold: str, aliases: List[str]) -> int:
    text = ""
    for t in turns:
        try:
            name = json.loads(t.get("tool_call") or "{}").get("name")
        except ValueError:
            name = None
        if name in SEARCH_TOOLS and not t.get("tool_error"):
            text += " " + (t.get("tool_output") or "")
    hay = norm(text)
    g = parts(gold)
    cands = [g] + [parts(a) for a in aliases or [] if len(parts(a)) >= max(1, len(g))]
    return int(any(c and all(p in hay for p in c) for c in cands))


def combine(stop: str, fmt: float, acc: int, hit: int, short_gold: bool) -> Dict[str, float]:
    query = float(acc) if short_gold else float(max(hit, acc))
    answered = stop == "response"
    reward = fmt * (W_ACC * acc + W_QUERY * query) if answered else 0.0
    return {"reward": reward, "fmt": fmt, "acc": float(acc), "query": query, "hit": float(hit),
            "answered": float(answered)}


def advantages(rewards: List[float], fatal: List[bool], estimator: str = "rloo") -> List[float]:
    n = len(rewards)
    mean = sum(rewards) / n
    if estimator == "grpo":
        std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
        adv = [(r - mean) / (std + 1e-6) for r in rewards]
    else:  # RLOO: leave-one-out baseline
        adv = [(r - (sum(rewards) - r) / (n - 1)) if n > 1 else 0.0 for r in rewards]
    return [max(a, 0.0) if f else a for a, f in zip(adv, fatal)]
