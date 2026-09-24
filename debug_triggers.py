#!/usr/bin/env python3
"""Per-trigger breakdown of the trigger cutflow step, read straight from NanoAOD with uproot.

Uses the same triggers.json entry and the same lumimask as categorizer.py, so the
"after trigger (OR)" line should equal cutflow-v2's trigger row. Run from the repo root.

With --target N, also searches every subset of the trigger list for ORs that give exactly
N events (e.g. a reference cutflow's trigger row), to identify which triggers it used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import uproot

from hbb.corrections import lumiMasks


def read_patterns(files, triggers, lumimask):
    """Encode each event's fired triggers as a bitmask (bit i = triggers[i])."""
    n_total = n_lumi = 0
    patterns = Counter()
    first_run = {}
    last_run = {}
    missing = Counter()

    for path in files:
        tree = uproot.open(path)["Events"]
        keys = set(tree.keys())
        present = [(i, t) for i, t in enumerate(triggers) if f"HLT_{t}" in keys]
        for t in set(triggers) - {t for _, t in present}:
            missing[t] += 1

        arrays = tree.arrays(
            ["run", "luminosityBlock"] + [f"HLT_{t}" for _, t in present], library="np"
        )
        n_total += len(arrays["run"])
        good = np.asarray(lumimask(arrays["run"], arrays["luminosityBlock"]), dtype=bool)
        n_lumi += int(good.sum())
        runs = arrays["run"][good]

        bits = np.zeros(int(good.sum()), dtype=np.int64)
        for i, t in present:
            fired = arrays[f"HLT_{t}"][good].astype(bool)
            bits |= fired.astype(np.int64) << i
            if fired.any():
                first_run[t] = min(first_run.get(t, np.inf), int(runs[fired].min()))
                last_run[t] = max(last_run.get(t, -np.inf), int(runs[fired].max()))
        uniq, counts = np.unique(bits, return_counts=True)
        patterns.update(dict(zip(uniq.tolist(), counts.tolist())))

    return n_total, n_lumi, patterns, first_run, last_run, missing


def or_count(pats, cnts, mask):
    return int(cnts[(pats & mask) != 0].sum())


def subset_search(pats, cnts, n_triggers, target, chunk=4096):
    """Return all trigger-subset bitmasks whose OR gives exactly `target` events,
    plus the closest counts if there is no exact match."""
    exact = []
    closest = []
    for start in range(1, 1 << n_triggers, chunk):
        masks = np.arange(start, min(start + chunk, 1 << n_triggers), dtype=np.int64)
        hit = (pats[None, :] & masks[:, None]) != 0
        totals = hit @ cnts
        exact.extend(masks[totals == target].tolist())
        order = np.argsort(np.abs(totals - target))[:5]
        closest.extend(zip(np.abs(totals[order] - target).tolist(), masks[order].tolist(),
                           totals[order].tolist()))
    closest.sort()
    return exact, closest[:5]


def names(mask, triggers):
    return [t for i, t in enumerate(triggers) if mask >> i & 1]


def main(args):
    triggers = json.loads(Path(args.triggers).read_text())[args.year]
    lumimask = lumiMasks[args.year[:4]]
    print(f"triggers loaded for {args.year} ({len(triggers)}): {triggers}\n")

    n_total, n_lumi, patterns, first_run, last_run, missing = read_patterns(
        args.files, triggers, lumimask
    )
    pats = np.array(list(patterns.keys()), dtype=np.int64)
    cnts = np.array(list(patterns.values()), dtype=np.int64)
    all_mask = (1 << len(triggers)) - 1
    n_or = or_count(pats, cnts, all_mask)

    print(f"total events:       {n_total}")
    print(f"after lumimask:     {n_lumi}")
    print(f"after trigger (OR): {n_or}\n")

    width = max(len(t) for t in triggers)
    print(f"{'trigger':<{width}}  {'fired':>9}  {'only this':>9}  {'OR without it':>13}"
          f"  {'first run':>9}  {'last run':>9}")
    for i, t in enumerate(triggers):
        bit = 1 << i
        fired = or_count(pats, cnts, bit)
        only = int(cnts[pats == bit].sum())
        print(f"{t:<{width}}  {fired:>9}  {only:>9}  {n_or - only:>13}"
              f"  {first_run.get(t, '-'):>9}  {last_run.get(t, '-'):>9}")

    met = sum(1 << i for i, t in enumerate(triggers) if t.startswith("PFMET"))
    vbf = sum(1 << i for i, t in enumerate(triggers) if t.startswith("VBF"))
    print(f"\nMET triggers only (OR):       {or_count(pats, cnts, met)}")
    print(f"VBF triggers only (OR):       {or_count(pats, cnts, vbf)}")
    print(f"VBF fired, no MET fired:      "
          f"{int(cnts[((pats & vbf) != 0) & ((pats & met) == 0)].sum())}")

    if missing:
        print("\nMISSING from some/all files (never fire there):")
        for t, n in missing.items():
            print(f"  {t}: missing in {n}/{len(args.files)} file(s)")

    if args.target is None:
        return

    print(f"\nSearching all {2 ** len(triggers) - 1} trigger subsets for OR == {args.target} ...")
    exact, closest = subset_search(pats, cnts, len(triggers), args.target)
    if not exact:
        print("No subset reproduces the target exactly, so the reference is not a plain OR of")
        print("any subset of these triggers (look at run-dependent use / dataset overlap).")
        print("Closest subsets:")
        for diff, mask, total in closest:
            print(f"  {total} (off by {diff}): {names(mask, triggers)}")
        return

    print(f"{len(exact)} subset(s) reproduce it exactly. Per trigger, across those subsets:")
    for i, t in enumerate(triggers):
        n_in = sum(m >> i & 1 for m in exact)
        verdict = "always in" if n_in == len(exact) else "never in" if n_in == 0 else "either"
        print(f"  {t:<{width}}  {verdict}")
    smallest = min(exact, key=lambda m: bin(m).count("1"))
    print(f"\nSmallest matching subset: {names(smallest, triggers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="NanoAOD file(s): local paths or root:// URLs")
    parser.add_argument("--year", required=True)
    parser.add_argument("--triggers", default="src/hbb/triggers.json")
    parser.add_argument("--target", type=int, default=None,
                        help="Reference trigger-row count to reproduce by subset search")
    main(parser.parse_args())
