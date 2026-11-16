#!/usr/bin/env python3
"""Sample SFT trajectories for an offline LLM spot check (answer / grounding / search quality).

Writes one JSON list of compact cases: question, primary image (JPEG, <=1024 px, base64), and a
digest of the trajectory (each assistant turn's reasoning tail + tool call, each observation head,
final answer). Draws --n samples from each named dataset with a fixed seed.

    python spotcheck_prep.py --set v3b=/root/datasets/sft_v3b_new/sft_subset.json \
        --set v2=/root/datasets/sft_subset_q3vl_v2/sft_subset.json --n 50 --out /root/sft-audit/v3b/spotcheck.json
"""
import argparse, base64, io, json, random, re

from PIL import Image

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def digest(conv: list) -> tuple[str, str, str]:
    q = squash(conv[0]["value"].replace("<image>", ""))
    lines, final = [], ""
    for c in conv[1:]:
        v = c.get("value") or ""
        if c["from"] == "gpt":
            m = CALL.search(v)
            think = squash(re.sub(r"<tool_call>.*", "", v, flags=re.S).replace("<think>", "").replace("</think>", ""))
            if m:
                lines.append(f"[agent] {think[-300:]}\n  CALL {squash(m.group(1))[:200]}")
            else:
                final = squash(v)
                lines.append(f"[agent final] {final[:700]}")
        elif c["from"] == "observation":
            lines.append(f"  OUTPUT {squash(v.replace('<image>', ''))[:700]}")
    return q, "\n".join(lines), final[:700]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="name=path/to/sharegpt.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for spec in a.set:
        name, path = spec.split("=", 1)
        data = json.load(open(path, encoding="utf-8"))
        for i in random.Random(f"{a.seed}:{name}").sample(range(len(data)), min(a.n, len(data))):
            s = data[i]
            q, traj, final = digest(s["conversations"])
            im = Image.open(s["images"][0]).convert("RGB")
            im.thumbnail((1024, 1024))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            out.append({"case_id": f"{name}:{i}", "set": name, "question": q, "trajectory": traj[:14000],
                        "final": final, "image_b64": base64.b64encode(buf.getvalue()).decode()})
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(out)} cases to {a.out}")


if __name__ == "__main__":
    main()
