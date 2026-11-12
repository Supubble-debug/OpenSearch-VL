#!/usr/bin/env python3
"""Build RL / probe candidate questions from OpenSearch-VL/Search-VL-RL-8K.

RL-8K ships {question, answer, images, dataset}. Only the ``demo_1k`` subset has
short reference answers; the other ~7k ``answer`` fields are long agent responses
(same style and sources as the SFT trajectories), some of them refusals. So:

1. drop duplicates (same normalised question + same primary-image pHash);
2. drop overlap with the evaluation benchmarks (text exact / 5-gram Jaccard /
   image pHash, same rules as SFT selection) and with the SFT-36K questions
   (text; plus image pHash when --sft-images is given, i.e. on the instance —
   generic questions such as "where is this?" are flagged only when the image
   also matches);
3. drop refusal-like long answers;
4. stratified sample: --demo-quota from demo_1k, the rest across the other
   sources with quota ~ sqrt(pool size);
5. with --extract, an LLM (OpenAI-compatible, JUDGE_API_* env) turns every
   reference into one short canonical answer or rejects it (hedged, several
   candidates, refusal, open-ended explanation);
6. write an eval-format parquet (data_id, prompt, images, category, answer,
   long_answer, extract_status) that run_infer.py / eval_with_gpt4o.py consume,
   plus report.json.

    python build_rl_candidates.py --rl8k /data/rl8k --eval /data/vdr-eval/*.parquet /data/fvqa/fvqa_test.parquet \
        --sft-root /data/sft36k --out /data/rl_candidates --n 800 --demo-quota 320 --extract
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import io
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "select_sft_subset", _here.parent / "SFT" / "scripts" / "select_sft_subset.py")
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

REFUSAL_RE = re.compile(
    r"cannot (?:identify|determine|find|be determined|confirm)|unable to (?:identify|determine|find)|"
    r"could not (?:identify|find|determine)|not (?:able|possible) to (?:identify|determine)|"
    r"无法(?:确定|识别|判断|找到)|不能确定", re.I)

EXTRACT_SYSTEM = "You turn reference solutions into short gold answers for automatic grading. Reply with one JSON object only."
EXTRACT_TASK = """QUESTION:
{question}

REFERENCE SOLUTION (written by an agent; may be long):
{answer}

Extract the single final answer to the QUESTION as stated by the reference: the shortest phrase a grader can
compare against (a name, number, date, title, place, yes/no, or a short phrase; keep the original language;
for multi-part questions join the parts with "; "). Do not add facts the reference does not state.

status:
- "ok": the reference commits to one definite answer;
- "hedged": it only guesses or offers the answer with "possibly / likely / may be";
- "multiple": it lists several alternative candidates without choosing;
- "refusal": it says the answer cannot be determined;
- "open_ended": the question asks for an explanation/description with no short checkable answer.

Return {{"answer": "<short answer or empty>", "status": "ok|hedged|multiple|refusal|open_ended"}}"""


def norm_q(q: str) -> str:
    return sel.norm_text(re.sub(r"<image>", " ", q or ""))


def load_sft_items(root: Path, with_images: bool) -> list[dict]:
    items = []
    for src, fname in sel.SOURCES.items():
        p = root / src / fname
        if not p.exists():
            print(f"[sft] missing {p}, skipped", flush=True)
            continue
        for idx, s in enumerate(json.load(open(p, encoding="utf-8"))):
            conv = s.get("conversations") or []
            q = conv[0].get("value", "") if conv else ""
            ph = []
            if with_images and s.get("images"):
                ip = s["images"][0]
                ip = ip if os.path.isabs(ip) else str(root / src / ip)
                try:
                    from PIL import Image
                    ph = [sel.phash_int(Image.open(ip))]
                except Exception:
                    pass
            items.append({"bench": f"sft:{src}", "id": f"{src}:{idx}", "qnorm": norm_q(q), "phashes": ph,
                          "q": q[:200]})
    return items


def extract_one(api: str, key: str, model: str, extra: dict, question: str, answer: str) -> dict:
    import requests
    body = {"model": model, "temperature": 0.0, "max_tokens": 200,
            "messages": [{"role": "system", "content": EXTRACT_SYSTEM},
                         {"role": "user", "content": EXTRACT_TASK.format(question=question[:2000], answer=answer[:6000])}],
            **extra}
    last = ""
    for attempt in range(4):
        try:
            r = requests.post(f"{api.rstrip('/')}/chat/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code in (401, 402):
                raise SystemExit(f"extraction API: HTTP {r.status_code} (key or balance)")
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", text, re.S)
                out = json.loads(m.group(0)) if m else {}
                if out.get("status"):
                    return {"short": str(out.get("answer") or "").strip(), "status": str(out["status"]).strip()}
                last = f"unparsable: {text[:120]}"
            else:
                last = f"HTTP {r.status_code}"
        except (ValueError, KeyError) as exc:
            last = repr(exc)
        except Exception as exc:  # noqa: BLE001  (network)
            last = repr(exc)
        time.sleep(2 * (attempt + 1))
    return {"short": "", "status": f"failed: {last}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl8k", required=True, help="dir with rl_data.jsonl and images/")
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--sft-root", default=None, help="Search-VL-SFT-36K dir (json files; images optional)")
    ap.add_argument("--sft-images", action="store_true", help="also pHash SFT images (needs extracted images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-ids", nargs="*", default=[],
                    help="parquet(s) or text file(s) of data_ids already sampled; they are not drawn again")
    ap.add_argument("--skip-sources", nargs="*", default=[],
                    help="RL-8K sources not to sample (e.g. palace: AI references invent catalogue names)")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--demo-quota", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.rl8k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(root / "rl_data.jsonl", encoding="utf-8")]
    reasons = collections.Counter()
    from PIL import Image

    recs, seen = [], set()
    for i, r in enumerate(rows):
        img = root / r["images"][0]
        try:
            with Image.open(img) as im:
                ph = sel.phash_int(im)
        except Exception:
            reasons["img:undecodable"] += 1
            continue
        qn = norm_q(r["question"])
        dkey = (qn, ph)
        if dkey in seen:
            reasons["dup:question+image"] += 1
            continue
        seen.add(dkey)
        recs.append({"rid": f"rl8k_{i:05d}", "src": r["dataset"], "q": r["question"], "qnorm": qn, "phash": ph,
                     "answer": r["answer"], "images": [str(root / p) for p in r["images"]]})
    print(f"[load] {len(rows)} rows, {len(recs)} after dedup", flush=True)
    if args.exclude_ids:
        import pandas as pd
        seen_ids = set()
        for p in args.exclude_ids:
            seen_ids |= (set(pd.read_parquet(p, columns=["data_id"]).data_id) if p.endswith(".parquet")
                         else set(open(p).read().split()))
        reasons["already_sampled"] += sum(r["rid"] in seen_ids for r in recs)
        recs = [r for r in recs if r["rid"] not in seen_ids]
        print(f"[exclude] {len(seen_ids)} ids already sampled, {len(recs)} remain", flush=True)

    indexes = []
    if args.eval:
        items, counts = sel.load_eval(args.eval)
        print(f"[eval] {len(items)} eval items {counts}", flush=True)
        indexes.append(("eval", sel.OverlapIndex(items, 0.7, 8, 6)))
    if args.sft_root:
        sft_items = load_sft_items(Path(args.sft_root), args.sft_images)
        print(f"[sft] {len(sft_items)} SFT questions (images hashed: {args.sft_images})", flush=True)
        indexes.append(("sft", sel.OverlapIndex(sft_items, 0.7, 8, 6)))
    examples = collections.defaultdict(list)
    kept = []
    for r in recs:
        hit = None
        short = len(r["qnorm"].split()) < 8
        for name, idx in indexes:
            # A short generic question ("where is this?") matching by text alone is not overlap:
            # for those only the image counts (SFT images are hashed only with --sft-images).
            m = idx.match("" if short else r["qnorm"], r["phash"])
            if not m:
                continue
            kind, item = m
            hit = f"{name}:{kind}"
            if len(examples[hit]) < 5:
                examples[hit].append({"rl": r["q"][:160], "other": item.get("q", "")[:160], "where": item["bench"]})
            break
        if hit:
            reasons[hit] += 1
            continue
        if r["src"] != "demo_1k" and REFUSAL_RE.search(r["answer"][:800]):
            reasons["answer:refusal_like"] += 1
            continue
        kept.append(r)
    pools = collections.defaultdict(list)
    for r in kept:
        pools[r["src"]].append(r)
    print(f"[filter] kept {len(kept)}; reasons {dict(reasons)}; pools { {k: len(v) for k, v in pools.items()} }",
          flush=True)

    rng = random.Random(args.seed)
    quota = {"demo_1k": min(args.demo_quota, len(pools.get("demo_1k", [])))}
    others = [s for s in pools if s != "demo_1k" and s not in args.skip_sources]
    rest = args.n - quota["demo_1k"]
    w = {s: math.sqrt(len(pools[s])) for s in others}
    for s in others:
        quota[s] = min(len(pools[s]), round(rest * w[s] / sum(w.values())))
    picked = []
    for s, k in quota.items():
        picked += rng.sample(pools[s], k)
    rng.shuffle(picked)
    print(f"[sample] {len(picked)} candidates {quota}", flush=True)

    if args.extract:
        api, key = os.environ.get("JUDGE_API_BASE_URL", ""), os.environ.get("JUDGE_API_KEY", "")
        model = os.environ.get("JUDGE_MODEL", "deepseek-flash")
        extra = json.loads(os.environ.get("LLM_EXTRA_BODY_JSON") or "{}")
        if not (api and key):
            raise SystemExit("--extract needs JUDGE_API_BASE_URL and JUDGE_API_KEY")
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda r: extract_one(api, key, model, extra, r["q"], r["answer"]), picked))
        for r, e in zip(picked, res):
            r.update(e)
    else:
        for r in picked:
            r.update({"short": "", "status": "not_extracted"})

    import pandas as pd
    out_rows = []
    for r in picked:
        images = []
        for p in r["images"]:
            with open(p, "rb") as fh:
                images.append({"bytes": fh.read()})
        out_rows.append({"data_id": r["rid"], "id": r["rid"], "category": r["src"], "data_source": "rl8k",
                         "prompt": [{"role": "user", "content": r["q"]}], "images": images,
                         "answer": r["short"] or r["answer"], "long_answer": r["answer"],
                         "extract_status": r["status"]})
    df = pd.DataFrame(out_rows)
    df.to_parquet(out / "candidates_all.parquet")
    ok = df[df.extract_status == "ok"] if args.extract else df
    ok.to_parquet(out / "candidates.parquet")
    report = {
        "params": vars(args), "rows": len(rows), "after_dedup": len(recs), "kept_after_filters": len(kept),
        "reasons": dict(reasons.most_common()), "pools": {k: len(v) for k, v in pools.items()},
        "quota": quota, "extract_status": dict(collections.Counter(df.extract_status)),
        "usable_by_source": dict(collections.Counter(ok.category)), "usable": len(ok),
        "overlap_examples": examples,
    }
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("params", "overlap_examples")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
