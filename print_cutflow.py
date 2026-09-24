#!/usr/bin/env python3
"""Print the nominal cutflow histogram from one or more categorizer run.py/condor output
pickles as a table, combining them (summed) if more than one is given.

run.py's pickle is keyed by fileset dataset name (per coffea's apply_to_fileset), e.g.
{"Test": {"nominal": {"cutflow": ...}}} -- use --sample to pick one if there's more than one.
For condor output, e.g. all out_*.pkl files from one submit_from_yaml.py sample, pass them
all (or a glob pattern) to get the combined cutflow across every job/file-range.
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

# Mirrors categorizer.py's `regions` dict (cut index 0 is always the "no cuts"
# baseline, added implicitly by skim()). Keep this in sync by hand -- the cutflow
# histogram itself only stores numeric cut indices, not names, and `regions` is
# rebuilt inline per-event inside process_shift rather than being importable.
REGION_CUTS = {
    "signal-all": ["trigger", "lumimask", "metfilter"],
    "cutflow-resolved": [
        "trigger",
        "metfilter",
        "no_tight_muons",
        "no_tight_electrons",
        "no_taus",
        "atleast2jets50",
        "atleast1subjet",
        "atleast1fatjet50",
        "lead_pt200",
        "trail_pt0",
        "puppimet250",
        "has_fatjet",
        "lead_pt200",
        "lead_eta2p5",
        "lead_pt300",
        "lead_eta2p5",
        "wtag075",
        "W_mass_window",
        "trail_pt0",
        "sub_pt300",
        "sub_eta2p5",
        "sub_msd0",
        "sub_msd9999",
        "zero_btagged",
        "atleast2_nonbtagged",
        "atleast2_tagjets",
        "opp_hemi",
        "deta_2p5",
        "mjj_500",
    ],
    "cutflow-v2": [
        "lumimask",
        "trigger",
        "flag_EcalDeadCellTriggerPrimitiveFilter",
        "flag_BadPFMuonFilter",
        "flag_eeBadScFilter",
        "flag_BadPFMuonDzFilter",
        "flag_goodVertices",
        "flag_hfNoisyHitsFilter",
        "flag_globalSuperTightHalo2016Filter",
        "flag_ecalBadCalibFilter",
        "no_loose_muons",
        "no_loose_electrons",
        "no_taus_v2",
        "fatjet_pt200_raw",
        "puppimet_ge250",
        "tagjetpair_loose",
        "puppimet_ge250",
        "wtag_lead_pt250",
        "wtag_lead_eta2p5",
        "wtag_dphi_met",
        "btagpass_ge0",
        "btagfail_ge0",
        "atleast2_final_tagjets",
        "finaltagjet_deta2p5",
        "finaltagjet_mjj500",
    ],
    "signal-ggf-BDT": [
        "trigger",
        "lumimask",
        "metfilter",
        "ak4jetveto",
        "minjetkin",
        "antiak4btagMediumOppHem",
        "lowmet",
        "noleptons",
        "BDTisggF",
    ],
    "signal-vh-BDT": [
        "trigger",
        "lumimask",
        "metfilter",
        "ak4jetveto",
        "minjetkin",
        "antiak4btagMediumOppHem",
        "lowmet",
        "noleptons",
        "BDTisVH",
    ],
    "signal-vbf-BDT": [
        "trigger",
        "lumimask",
        "metfilter",
        "ak4jetveto",
        "minjetkin",
        "antiak4btagMediumOppHem",
        "lowmet",
        "noleptons",
        "BDTisVBF",
    ],
}


def cut_name(region, cut_index):
    if cut_index == 0:
        return "(none -- baseline)"
    names = REGION_CUTS.get(region)
    if names is None or cut_index > len(names):
        return "?"
    return names[cut_index - 1]


def format_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt_row(row) for row in rows]
    return "\n".join(lines)


def pct(numer, denom):
    return f"{100 * numer / denom:.2f}" if denom else "-"


def print_region_table(cutflow, region, dataset, max_cuts):
    h = cutflow[{"region": region, "dataset": dataset}][{"genflavor": sum}]
    values = h.values()
    errors = np.sqrt(h.variances())

    # bins past the last selection actually applied to this region stay at 0
    # forever (see categorizer.py's `regions` dict), so trim trailing zeros
    last_nonzero = 0
    for i, v in enumerate(values):
        if v != 0:
            last_nonzero = i
    ncuts = min(max_cuts, last_nonzero + 1) if max_cuts is not None else last_nonzero + 1

    total = values[0]
    headers = ["cut", "name", "events", "+/- stat", "eff %", "cumeff %"]
    rows = []
    for i in range(ncuts):
        eff = pct(values[i], values[i - 1]) if i > 0 else "-"
        cumeff = pct(values[i], total)
        rows.append([i, cut_name(region, i), f"{values[i]:.2f}", f"{errors[i]:.2f}", eff, cumeff])

    print(f"\nRegion: {region}  |  Dataset: {dataset}")
    print(format_table(headers, rows))


def unwrap_nominal(raw, sample):
    """coffea's apply_to_fileset nests output as {dataset_name: {shift_name: output}},
    so run.py's pickle has an extra dataset-name level above what process() returns."""
    if "nominal" in raw:
        return raw["nominal"]

    keys = list(raw.keys())
    if sample is not None:
        if sample not in raw:
            raise SystemExit(f"Unknown --sample {sample!r}. Available: {keys}")
        chosen = raw[sample]
    elif len(keys) == 1:
        chosen = raw[keys[0]]
    else:
        raise SystemExit(
            f"Pickle contains multiple datasets {keys}; pass --sample <name> to pick one."
        )

    if "nominal" not in chosen:
        raise SystemExit(f"No 'nominal' shift found; available shifts: {list(chosen.keys())}")
    return chosen["nominal"]


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths


def load_cutflow(path, sample):
    with Path(path).open("rb") as f:
        raw = pickle.load(f)
    return unwrap_nominal(raw, sample)["cutflow"]


def main(args):
    paths = expand_paths(args.files)
    if not paths:
        raise SystemExit(f"No files matched: {args.files}")

    cutflow = None
    for path in paths:
        h = load_cutflow(path, args.sample)
        if cutflow is None:
            cutflow = h
        else:
            try:
                cutflow = cutflow + h
            except ValueError as e:
                raise SystemExit(
                    f"Could not combine {path} with the previous file(s): {e}. "
                    "This usually means the pickles were produced by different "
                    "categorizer.py versions (mismatched cutflow axis definitions) "
                    "-- re-run all jobs against the same code."
                ) from e

    print(f"Combined {len(paths)} file(s): {paths}")

    available_regions = list(cutflow.axes["region"])
    available_datasets = list(cutflow.axes["dataset"])

    regions = args.region if args.region else available_regions
    datasets = args.dataset if args.dataset else available_datasets

    for name, requested, available in [
        ("region", regions, available_regions),
        ("dataset", datasets, available_datasets),
    ]:
        unknown = [r for r in requested if r not in available]
        if unknown:
            raise SystemExit(f"Unknown {name}(s) {unknown}. Available: {available}")

    for region in regions:
        for dataset in datasets:
            print_region_table(cutflow, region, dataset, args.max_cuts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Path(s) to pickle file(s) produced by run.py/condor -- accepts multiple "
        "files and/or glob patterns (e.g. 'out_*.pkl'); combined cutflows are summed",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Fileset dataset key to select (only needed if the pickle has more than one)",
    )
    parser.add_argument(
        "--region", nargs="*", default=None, help="Only show these regions (default: all)"
    )
    parser.add_argument(
        "--dataset", nargs="*", default=None, help="Only show these datasets (default: all)"
    )
    parser.add_argument(
        "--max-cuts",
        type=int,
        default=None,
        help="Cap the number of cut rows shown per region (default: trim trailing empty bins only)",
    )
    args = parser.parse_args()
    main(args)
