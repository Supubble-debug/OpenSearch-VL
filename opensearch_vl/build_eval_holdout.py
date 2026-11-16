#!/usr/bin/env python3
"""Seal a held-out benchmark test set disjoint from paper7_n50 and from all training data.

Same sampling protocol as prepare_eval_benchmarks.py (per-benchmark shuffle seeded with
"<seed>:<bench>", rows without image/answer skipped): the first --skip valid rows of each
shuffle are exactly the existing dev set (checked against --dev), the next valid rows that
do not overlap anything become the test set (--per-bench per benchmark).

Overlap = text (exact / 5-gram Jaccard >= 0.7 for questions of >= 8 words) or image pHash
(Hamming <= 6) against: the dev set, SFT-36K, every RL parquet given with --rl, and earlier
test rows (no near-duplicate pairs inside the test set). Short questions match by image only.

    python build_eval_holdout.py --src /root/datasets/vdr-eval/data \
        --dev /root/autodl-fs/datasets/imagentrl/eval/paper7_n50.parquet \
        --sft-root /root/datasets/Search-VL-SFT-36K --rl /root/rl_data/*.parquet \
        --out /root/autodl-fs/datasets/imagentrl/eval/paper7_test50.parquet
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import build_rl_candidates as brc
from prepare_eval_benchmarks import BENCHMARKS, PREFIX_RE

sel = brc.sel


def _hash_path(path: str):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return sel.phash_int(im)
    except Exception:
        return None


def _hash_bytes(b: bytes):
    from PIL import Image
    try:
        return sel.phash_int(Image.open(io.BytesIO(b)))
    except Exception:
        return None


def _first_bytes(images):
    imgs = list(images) if images is not None else []
    if not len(imgs):
        return None
    x = imgs[0]
    return x.get("bytes") if isinstance(x, dict) else x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--sft-root", required=True)
    ap.add_argument("--rl", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=50)
    ap.add_argument("--per-bench", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    items = brc.load_sft_items(Path(args.sft_root), with_images=False)
    paths = []
    root = Path(args.sft_root)
    for src, fname in sel.SOURCES.items():
        p = root / src / fname
        if not p.exists():
            continue
        for s in json.load(open(p, encoding="utf-8")):
            imgs = s.get("images") or []
            q = imgs[0] if imgs else ""
            paths.append(q if not q or os.path.isabs(q) else str(root / src / q))
    assert len(paths) == len(items), (len(paths), len(items))
    with ProcessPoolExecutor(args.workers) as ex:
        hs = list(ex.map(_hash_path, paths, chunksize=256))
    for it, h in zip(items, hs):
        it["phashes"] = [] if h is None else [h]
    print(f"[sft] {len(items)} items, {sum(h is not None for h in hs)} images hashed", flush=True)

    dev = pd.read_parquet(args.dev)
    dev_ids = set(dev.data_id)
    for name, df in [("dev", dev)] + [(Path(p).stem, pd.read_parquet(p)) for p in args.rl]:
        if "prompt" not in df.columns or "images" not in df.columns:
            print(f"[skip] {name}: no prompt/images columns", flush=True)
            continue
        qs = [(r["prompt"][0]["content"] if len(r["prompt"]) else "") for _, r in df.iterrows()]
        with ProcessPoolExecutor(args.workers) as ex:
            hs = list(ex.map(_hash_bytes, [_first_bytes(x) for x in df["images"]], chunksize=32))
        for did, q, h in zip(df.data_id, qs, hs):
            items.append({"bench": name, "id": str(did), "qnorm": brc.norm_q(q),
                          "phashes": [] if h is None else [h], "q": str(q)[:200]})
        print(f"[ref] {name}: {len(df)} rows", flush=True)

    index = sel.OverlapIndex(items, 0.7, 8, 6)
    rows, report = [], {"seed": args.seed, "skip": args.skip, "per_bench": args.per_bench, "benches": {}}
    src = Path(args.src)
    for stem, bench in BENCHMARKS.items():
        df = pd.read_parquet(src / f"{stem}.parquet")
        order = list(range(len(df)))
        random.Random(f"{args.seed}:{bench}").shuffle(order)
        valid, kept, dropped, test_here = 0, 0, [], []
        for i in order:
            if kept >= args.per_bench:
                break
            row = df.iloc[i]
            packed = row["image_packed"]
            packed = json.loads(packed) if isinstance(packed, str) else list(packed)
            images = [{"bytes": base64.b64decode(x["data"])} for x in packed if x.get("data")]
            answer = row["answer"]
            if not images or answer is None or not str(answer).strip():
                continue
            valid += 1
            case_id = brc.re.sub(r"[^\w.\-]", "_", f"{bench}__{row['id']}")
            if valid <= args.skip:
                assert case_id in dev_ids, f"protocol mismatch: {case_id} not in dev set"
                continue
            q = PREFIX_RE.sub("", str(row["question"]))
            qn = brc.norm_q(q)
            ph = _hash_bytes(images[0]["bytes"])
            m = index.match("" if len(qn.split()) < 8 else qn, ph)
            if m is None:  # near-duplicates inside the test set
                for o in test_here:
                    if (qn and len(qn.split()) >= 8 and qn == o[0]) or (ph is not None and o[1] is not None
                                                                         and bin(ph ^ o[1]).count("1") <= 6):
                        m = ("test_dup", {"bench": bench, "id": o[2]})
                        break
            if m:
                dropped.append({"id": case_id, "kind": m[0], "with": f"{m[1]['bench']}:{m[1]['id']}"})
                continue
            test_here.append((qn, ph, case_id))
            rows.append({"data_id": case_id, "id": case_id, "category": bench, "data_source": stem,
                         "prompt": [{"role": "user", "content": q}], "images": images, "answer": str(answer)})
            kept += 1
        report["benches"][bench] = {"kept": kept, "dropped": dropped}
        print(f"[{bench}] kept {kept}, dropped {len(dropped)} overlapping", flush=True)
        assert kept == args.per_bench, f"{bench}: only {kept} clean rows"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    report["ids"] = [r["data_id"] for r in rows]
    json.dump(report, open(out.with_suffix(".report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {len(rows)} cases to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
