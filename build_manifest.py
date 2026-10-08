"""
build_manifest.py -- BANK_dataset/ -> data/manifest.jsonl (one line per item).

  {"qid": "FloorPlan10_delete_Plate_NA_16", "dataset": "ithor_procthor",
   "tier": "delete", "answer": "C", "scene": "ithor_procthor|FloorPlan10",
   "family": null, "band": "II", "magnitude": "NA",
   "images": [...A,B,C,D...], "rel_images": [...], "image_sha256": [...],
   "sup": {"edit_type": ..., "object": ..., "edit_description": ...,
           "gt_reason": ..., "distractor": 0, ...}}

`scene` is taken from the scene id the generator declared (scene_id /
floorplan / scene), not from the item name: e.g. Infinigen's 160 items come
from only 45 rooms, which the item names do not reveal. 3D-FRONT declares no
scene id and falls back to the item-name prefix.

`sup` (the generator's record of the edit, canonical letters) is a training
target only; it is never model input and plays no part in the split.
Only A.png .. D.png are read as panels.

  python build_manifest.py --out data/manifest.jsonl
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import (LETTERS, find_bank, panel_letter_leftovers, sha256_file,
                    supervision_from_row)

DATASET_TAGS = {
    "final_iTHOR_ProcTHOR_383": "ithor_procthor",
    "final_RoboTHOR_107": "robothor",
    "final_TDW_147": "tdw",
    "final_3dfront_102": "3dfront",
    "final_SAGE_101": "sage",
    "final_Infinigen_Indoors_160": "infinigen",
}
SCENE_KEYS = ("scene_id", "floorplan", "scene")
TIER_TOKENS = ("rotate", "rot", "mirror", "move", "delete", "del",
               "detail", "det", "swap")
TIER_ALIASES = {"remove": "delete", "rot": "rotate", "del": "delete", "det": "detail"}


def tag_of(folder):
    return DATASET_TAGS.get(folder, folder.replace("final_", "").lower())


def load_meta(item_dir):
    f = next((c for c in (item_dir / "repro.json", item_dir / "row.json",
                          item_dir / "v2_info.json") if c.exists()), None)
    if f is None:
        qs = sorted(item_dir.glob("q*.json"))
        if not qs:
            sys.exit(f"{item_dir}: no metadata json")
        f = qs[0]
    return json.loads(f.read_text())


def scene_of(tag, qid, meta):
    for k in SCENE_KEYS:
        v = meta.get(k)
        if v not in (None, "", []):
            return f"{tag}|{v}"
    toks = qid.split("_")                      # 3D-FRONT: item-name prefix
    for i, t in enumerate(toks):
        if t.lower() in TIER_TOKENS:
            return f"{tag}|{'_'.join(toks[:i])}"
    return f"{tag}|{qid}"


def magnitude_of(row, meta):
    for k in ("rotate_degree", "move_length"):
        v = (row.get(k) or "").strip()
        if v and v.upper() != "NA":
            return v
    v = meta.get("magnitude") or meta.get("change_param")
    return str(v) if v not in (None, "", []) else "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None)
    ap.add_argument("--out", default="data/manifest.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    bank = Path(args.bank or find_bank() or "BANK_dataset").resolve()
    if not bank.is_dir():
        sys.exit(f"{bank}: no BANK_dataset folder; pass --bank")

    items = []
    for ds_dir in sorted(p for p in bank.iterdir() if p.is_dir()):
        tag = tag_of(ds_dir.name)
        csvs = sorted(ds_dir.glob("manifest_*.csv"))
        if len(csvs) != 1:
            sys.exit(f"{ds_dir}: expected exactly one manifest_*.csv")
        rows = list(csv.DictReader(open(csvs[0], newline="")))
        for r in rows:
            qid = r["item_id"]
            it = ds_dir / qid
            meta = load_meta(it)
            # The CSV is the source of truth for tier and gt. The CSV's
            # per-model answer/correct columns are never read.
            tier = TIER_ALIASES.get(r["tier"].lower(), r["tier"].lower())
            gt = r["gt"].strip().upper()
            assert gt in LETTERS, qid
            imgs = [it / f"{L}.png" for L in LETTERS]
            assert all(p.exists() for p in imgs), qid
            fam = meta.get("family_id")
            items.append({
                "qid": qid,
                "dataset": tag,
                "dataset_dir": ds_dir.name,
                "tier": tier,
                "gt": gt,
                "answer": gt,
                "scene": scene_of(tag, qid, meta),
                "family": f"{tag}|fam|{fam}" if fam not in (None, "", []) else None,
                "band": meta.get("band") or meta.get("rung") or "NA",
                "magnitude": magnitude_of(r, meta),
                "images": [str(p) for p in imgs],
                "sup": supervision_from_row(r),
                "rel_images": ["%s/%s/%s" % (ds_dir.name, qid, p.name) for p in imgs],
            })

    flat = [p for it in items for p in it["images"]]
    with ThreadPoolExecutor(args.workers) as ex:
        digests = list(ex.map(sha256_file, flat))
    for i, it in enumerate(items):
        it["image_sha256"] = digests[4 * i: 4 * i + 4]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    by_ds = defaultdict(list)
    for it in items:
        by_ds[it["dataset"]].append(it)
    print(f"wrote {len(items)} items -> {out}")
    for ds, v in sorted(by_ds.items()):
        print(f"  {ds:16s} n={len(v):4d} scenes={len({x['scene'] for x in v}):4d} "
              f"tiers={dict(sorted(Counter(x['tier'] for x in v).items()))}")
    shared = sum(1 for c in Counter(flat_h for it in items
                                    for flat_h in it["image_sha256"]).values() if c > 1)
    print(f"  panel files shared by more than one item: {shared}")
    left = [it["qid"] for it in items
            if panel_letter_leftovers(it["sup"]["gt_reason"])]
    if left:
        print(f"  warning: {len(left)} gt_reason strings name a panel outside a "
              f"'view X' span: {left[:5]}")


if __name__ == "__main__":
    main()
