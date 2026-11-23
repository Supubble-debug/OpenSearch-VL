"""Evidence judge v2 for the RL query reward (replaces the string retrieval hit when --query-signal evidence).

Validated 2026-11-22 on hand-labelled RL rollouts (output/poolclean/evidence_v2_validation.md): on 35 clear
string-hit / old-judge disagreements it was right on 32 (string hit 10, old strict judge 25); judge failures 1/100.

The judge splits the reference answer into parts and must copy, verbatim, the tool-output text supporting every part.
Code then checks each quote against the tool outputs (normalised substring). Verdict = judge says supported AND every
part has a verified quote AND no verified contradiction quote. Thinking on.

Env: EVIDENCE_API_BASE / EVIDENCE_API_KEY (default BAILIAN_API_BASE / BAILIAN_API_KEY), EVIDENCE_MODEL
(default deepseek-v4.1-flash-1017), EVIDENCE_MAX_CONCURRENCY (16), EVIDENCE_MAX_CHARS (60000, total tool text).
judge() returns True / False, or None when no verdict could be obtained (the caller drops the group).
"""
from __future__ import annotations

import json
import os
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PROMPT = """Question (about an image): {q}
Reference answer: {g}

Below are the tool outputs an agent saw, each tagged with its turn number.
{ev}

Task: decide whether these tool outputs contain evidence for the reference answer.

Step 1. Split the reference answer into its parts: the separate things the question asks for (usually separated by ";").
"also acceptable" lists alternative forms; one form per part is enough.
Step 2. For every part, find text in the tool outputs that supports it and COPY it exactly, character for character
(a short span is fine, e.g. one sentence or one image-search title). If nothing supports the part, leave the quote empty.
Step 3. Apply these rules:
 - Every part needs its own supporting quote. Aliases, translations and equivalent formats count.
 - The same word in an unrelated context is not support (e.g. "women" in a sentence about something else).
 - A related but different entity, event or value is not support (e.g. "Prime Minister's Questions" does not support
   "Prime Minister's Statement"; a neighbouring station does not support the station asked for).
 - Image-search titles that name the entity count as evidence for what the image shows, unless the titles name
   conflicting entities and none clearly dominates.
 - A summary saying the page does not contain the information is not evidence.
 - If an output contradicts the reference answer (a different value for the same thing, a source saying it is wrong)
   and no other output settles it in favour of the reference, copy the contradicting text into "contradiction".
Reply with one JSON object only:
{{"parts": [{{"part": "<part>", "quote": "<exact text or empty>", "turn": <int or null>}}], "contradiction": "<exact text or empty>", "supported": true|false}}"""

_SEM = threading.Semaphore(int(os.environ.get("EVIDENCE_MAX_CONCURRENCY", "16")))


def norm(s: Any) -> str:
    s = str(s).replace("\\n", " ").replace("\\t", " ")
    return "".join(c for c in unicodedata.normalize("NFKC", s).lower() if c.isalnum())


def tools_text(turns: List[Dict[str, Any]]) -> str:
    """Same layout as the validation export, without the 2500-char cap per output."""
    s = ""
    for n, t in enumerate(turns):
        if t.get("tool_call"):
            s += f"\n[turn {n}] {str(t['tool_call'])[:200]}:\n{t.get('tool_output') or ''}\n"
    cap = int(os.environ.get("EVIDENCE_MAX_CHARS", "60000"))
    return s if len(s) <= cap else s[-cap:]


def verify(v: Dict[str, Any], tools: str) -> Tuple[bool, List[Dict[str, Any]]]:
    allt = norm(tools)
    parts = []
    for p in v.get("parts") or []:
        q = norm(p.get("quote") or "")
        parts.append({**p, "verified": len(q) >= 2 and q in allt})
    con = norm(v.get("contradiction") or "")
    con_ok = len(con) >= 2 and con in allt
    final = bool(v.get("supported")) and bool(parts) and all(p["verified"] for p in parts) and not con_ok
    return final, parts


def _ask(prompt: str, attempts: int = 6, max_tokens: int = 12000) -> Optional[Dict[str, Any]]:
    base = os.environ.get("EVIDENCE_API_BASE") or os.environ.get("BAILIAN_API_BASE", "")
    key = os.environ.get("EVIDENCE_API_KEY") or os.environ.get("BAILIAN_API_KEY", "")
    model = os.environ.get("EVIDENCE_MODEL", "deepseek-v4.1-flash-1017")
    if not base or not key:
        raise RuntimeError("evidence judge: set EVIDENCE_API_BASE/KEY or BAILIAN_API_BASE/KEY")
    think = {"thinking": {"type": "enabled"}} if model.startswith("deepseek") else {"enable_thinking": True}
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, **think,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for k in range(attempts):
        try:
            with _SEM:
                r = json.load(urllib.request.urlopen(req, timeout=300))
            c = r["choices"][0]["message"].get("content") or ""
            s, e = c.find("{"), c.rfind("}")
            v = json.loads(c[s:e + 1])
            if "supported" in v and isinstance(v.get("parts"), list):
                return v
        except urllib.error.HTTPError as exc:
            time.sleep(min(60, 5 * (k + 1)) if exc.code == 429 else 3 * (k + 1))
            continue
        except Exception:
            pass
        time.sleep(3 * (k + 1))
    return None


def judge(question: str, gold_with_aliases: str, turns: List[Dict[str, Any]]) -> Optional[bool]:
    tools = tools_text(turns)
    if not tools.strip():
        return False
    v = _ask(PROMPT.format(q=question, g=gold_with_aliases, ev=tools))
    if v is None:
        return None
    return verify(v, tools)[0]
