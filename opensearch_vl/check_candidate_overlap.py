#!/usr/bin/env python3
"""Re-check RL candidate parquets against SFT-36K with image pHash (needs SFT images).

build_rl_candidates.py ran where only the SFT question texts were available; this
adds the image check. Short generic questions are matched by image only, long
ones by text or image (same rules as build_rl_candidates). Writes, per input
parquet, ``<name>.clean.parquet`` without the overlapping rows and a report.

    python check_candidate_overlap.py --sft-root /root/datasets/Search-VL-SFT-36K \
        --out /root/rl_data/overlap /root/rl_data/probe300.parquet /root/rl_data/reserve.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_rl_candidates", _here / "build_rl_candidates.py")
brc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brc)
sel = brc.sel


def _hash_one(path: str):
    from PIL import Image
    try:
        with Image.open(path) as im:
            return sel.phash_int(im)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--sft-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    import os
    import pandas as pd
    from PIL import Image

    root = Path(args.sft_root)
    items = brc.load_sft_items(root, with_images=False)
    paths = []
    for src, fname in sel.SOURCES.items():
        for s in json.load(open(root / src / fname, encoding="utf-8")):
            imgs = s.get("images") or []
            p = imgs[0] if imgs else ""
            paths.append(p if not p or os.path.isabs(p) else str(root / src / p))
    assert len(paths) == len(items), (len(paths), len(items))
    with ProcessPoolExecutor(args.workers) as ex:
        hashes = list(ex.map(_hash_one, paths, chunksize=256))
    for it, h in zip(items, hashes):
        it["phashes"] = [] if h is None else [h]
    print(f"[sft] {len(items)} questions, {sum(h is not None for h in hashes)} images hashed", flush=True)
    index = sel.OverlapIndex(items, 0.7, 8, 6)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for pq in args.parquets:
        df = pd.read_parquet(pq)
        drop, examples = [], []
        for _, r in df.iterrows():
            q = r["prompt"][0]["content"]
            qn = brc.norm_q(q)
            ph = sel.phash_int(Image.open(io.BytesIO(r["images"][0]["bytes"])))
            m = index.match("" if len(qn.split()) < 8 else qn, ph)
            if m:
                drop.append(r["data_id"])
                if len(examples) < 10:
                    examples.append({"id": r["data_id"], "kind": m[0], "rl": q[:120], "sft": m[1]["q"][:120],
                                     "sft_id": m[1]["id"]})
        name = Path(pq).stem
        df[~df.data_id.isin(drop)].to_parquet(out / f"{name}.clean.parquet")
        report[name] = {"rows": len(df), "dropped": len(drop), "kept": len(df) - len(drop),
                        "dropped_ids": drop, "examples": examples}
        print(f"[{name}] {len(df)} rows, {len(drop)} overlap with SFT images/text", flush=True)
    json.dump(report, open(out / "overlap_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
