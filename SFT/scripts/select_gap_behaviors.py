#!/usr/bin/env python3
"""Pick unused, already-audited SFT-36K trajectories that demonstrate the student's gap behaviours.

Pool = ids that passed the v2 two-pass vision answer filter (final_keep.jsonl) minus the ids already
in the v2 subset (and in any --exclude manifest). Each trajectory is scored on the behaviours the
79-question failure attribution found missing in the student (2026-11-16):

  entity      image_search (on the image or on a crop) before the first text/web search
  rewrite     >= 3 text/web searches whose queries differ substantially (word-Jaccard < 0.5 to every
              earlier query) -> the agent reformulates instead of repeating
  long        >= 5 tool calls (multi-hop / verification before answering)
  read        crop or layout_parsing used before answering (reads text/details in the image)

Writes keep_ids.txt (top --n by score, ties broken by a seeded shuffle, at least one behaviour) and
a report with feature counts, to feed select_sft_subset.py --keep-ids.

    python select_gap_behaviors.py --data-root /root/datasets/Search-VL-SFT-36K \
        --audited /root/sft-audit/v2/final_keep.jsonl \
        --exclude /root/datasets/sft_subset_q3vl_v2/manifest.jsonl /root/datasets/sft_val_200/manifest.jsonl \
        --n 1000 --out /root/sft-audit/v3b
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import re
from pathlib import Path

_spec = importlib.util.spec_from_file_location("select_sft_subset", Path(__file__).with_name("select_sft_subset.py"))
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
SEARCH = {"text_search", "web_search"}


def calls_of(sample: dict) -> list[tuple[str, dict]]:
    out = []
    for c in sample.get("conversations") or []:
        if c.get("from") != "gpt":
            continue
        for m in CALL_RE.finditer(c.get("value") or ""):
            try:
                j = json.loads(m.group(1))
            except Exception:
                continue
            out.append((j.get("name") or "", j.get("arguments") or j.get("parameters") or {}))
    return out


def words(q: str) -> set[str]:
    return set(re.findall(r"\w+", str(q).lower()))


def features(calls: list[tuple[str, dict]]) -> dict:
    names = [n for n, _ in calls]
    first_search = next((i for i, n in enumerate(names) if n in SEARCH), len(names))
    entity = "image_search" in names[:first_search]
    qs = [words(a.get("q") or a.get("query") or "") for n, a in calls if n in SEARCH]
    distinct = []
    for q in qs:
        if q and all(len(q & p) / max(1, len(q | p)) < 0.5 for p in distinct):
            distinct.append(q)
    return {"entity": entity, "rewrite": len(distinct) >= 3, "long": len(calls) >= 5,
            "read": any(n in ("crop", "layout_parsing") for n in names)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--audited", default=None, help="v2 final_keep.jsonl (required unless --unaudited-from)")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--unaudited-from", default=None,
                    help="candidates mode: score trajectories NOT in this manifest (e.g. the v2 cand10000 one) and NOT\n"
                         "excluded, keep those with rewrite or long, write cand_ids.txt (to be audited)")
    ap.add_argument("--cap-per-source", type=int, default=400)
    args = ap.parse_args()
    if args.unaudited_from:
        return candidates(args)
    if not args.audited:
        ap.error("--audited is required unless --unaudited-from is given")

    audited = [json.loads(l) for l in open(args.audited, encoding="utf-8")]
    used = set()
    for p in args.exclude:
        used |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}
    pool = {r["id"]: r for r in audited if r["id"] not in used}

    root = Path(args.data_root)
    scored = []
    for src, fname in sel.SOURCES.items():
        p = root / src / fname
        if not p.exists():
            continue
        for idx, s in enumerate(json.load(open(p, encoding="utf-8"))):
            sid = f"{src}:{idx}"
            if sid not in pool:
                continue
            f = features(calls_of(s))
            scored.append((sid, src, sum(f.values()), f))
    rng = random.Random(args.seed)
    rng.shuffle(scored)
    scored.sort(key=lambda x: -x[2])
    pick = [x for x in scored if x[2] >= 1][: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keep_ids.txt").write_text("".join(x[0] + "\n" for x in pick))
    cnt = lambda xs: {k: sum(x[3][k] for x in xs) for k in ("entity", "rewrite", "long", "read")}
    report = {"audited": len(audited), "pool_unused": len(pool), "scored": len(scored), "picked": len(pick),
              "pool_features": cnt(scored), "picked_features": cnt(pick),
              "picked_score_hist": dict(collections.Counter(x[2] for x in pick)),
              "picked_by_source": dict(collections.Counter(x[1] for x in pick)),
              "picked_by_cat": dict(collections.Counter(pool[x[0]].get("cat") for x in pick))}
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(json.dumps(report))
    return 0


def candidates(args) -> int:
    seen = {json.loads(l)["id"] for l in open(args.unaudited_from, encoding="utf-8")}
    for p in args.exclude:
        seen |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}
    root = Path(args.data_root)
    by_src = collections.defaultdict(list)
    for src, fname in sel.SOURCES.items():
        p = root / src / fname
        if not p.exists():
            continue
        for idx, s in enumerate(json.load(open(p, encoding="utf-8"))):
            sid = f"{src}:{idx}"
            if sid in seen:
                continue
            f = features(calls_of(s))
            if f["rewrite"] or f["long"]:
                by_src[src].append((sid, sum(f.values())))
    rng = random.Random(args.seed)
    ids = []
    for src, lst in by_src.items():
        rng.shuffle(lst)
        lst.sort(key=lambda x: -x[1])
        ids += [x[0] for x in lst[: args.cap_per_source]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cand_ids.txt").write_text("".join(i + "\n" for i in ids))
    rep = {"by_source_available": {k: len(v) for k, v in by_src.items()}, "candidates": len(ids)}
    json.dump(rep, open(out / "cand_report.json", "w"), indent=1)
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
