# import pickle, numpy as np

# with open("histograms/cutflow_2023_signal-all.pkl", "rb") as f:
#     cutflows = pickle.load(f)

# labels = ["no_cut", "pt_50", "msd_40", "txbb_0p95"]
# for proc, h in cutflows.items():
#     print(f"\n--- {proc} ---")
#     h1 = h[{"genflavor": sum}]
#     v, e = h1.values(), np.sqrt(h1.variances())
#     for lbl, vi, ei in zip(labels, v, e):
#         print(f"  {lbl:12s} {vi:12.2f} +/- {ei:.2f}")


#!/usr/bin/env python
"""Print the contents of cutflow_{year}_{region}.pkl as a table."""

import pickle
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- edit these
PKLFILE = "histograms/cutflow_2023_signal-all.pkl"

LABELS = [
    "no_cut",
    "2jets_pt50",
    "1fatjet_pt50",
    "lead_pt200",
    "trail_pt0",
    "met_250",
    "has_fatjet",
    "lead_eta2p5",
    "lead_pt300",
    "W_mass_window",
    "sub_pt300",
    "sub_eta2p5",
    "sub_msd0",
    "tagjets",
    "opp_hemi",
    "deta_2p5",
    "mjj_500",
]
# ---------------------------------------------------------------------------

path = Path(PKLFILE)
if not path.is_file():
    raise SystemExit(f"File not found: {path}")

with path.open("rb") as f:
    cutflows = pickle.load(f)

print(f"Processes: {list(cutflows.keys())}")

for proc, h in cutflows.items():
    # sum over every axis except `cut`
    names = [ax.name for ax in h.axes]
    if "cut" not in names:
        print(f"\n--- {proc} --- no 'cut' axis (axes: {names}); skipping")
        continue
    index = {n: sum for n in names if n != "cut"}
    h1 = h[index] if index else h

    values = h1.values()

    var = h1.variances()
    if var is None:
        errors = np.sqrt(np.abs(values))
        note = "  (no variances stored; errors are sqrt(N))"
    else:
        errors = np.sqrt(var)
        note = ""

    # pad/trim labels to match the number of filled bins
    labels = list(LABELS[: len(values)])
    labels += [f"cut{i}" for i in range(len(labels), len(values))]

    first = values[0] if len(values) and values[0] != 0 else np.nan

    print(f"\n--- {proc} ---{note}")
    print(f"  {'cut':<16}{'yield':>14}{'error':>12}{'abs eff':>10}{'rel eff':>10}")
    print("  " + "-" * 62)
    for i, (lbl, v, e) in enumerate(zip(labels, values, errors)):
        abs_eff = v / first if first == first else np.nan
        if i == 0:
            rel_s = "     ---"
        else:
            prev = values[i - 1]
            rel_s = f"{v / prev:10.4f}" if prev > 0 else "     ---"
        print(f"  {lbl:<16}{v:14.2f}{e:12.2f}{abs_eff:10.4f}{rel_s}")