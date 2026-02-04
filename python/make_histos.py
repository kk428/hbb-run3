#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hist
import numpy as np
from common import common_mc, data_by_year

from hbb import utils

# Define the possible ptbins
ptbins = np.array([300, 450, 500, 550, 600, 675, 800, 1200])

# Define the histogram axes
axis_to_histaxis = {
    "pt1": hist.axis.Variable(ptbins, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Variable(ptbins, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 40, 201, name="msd1", label="Jet 0 $m_{sd}$ [GeV]"),
    "mass1": hist.axis.Regular(30, 0, 200, name="mass1", label="Jet 0 PNet mass [GeV]"),
    "category": hist.axis.StrCategory([], name="category", label="Category", growth=True),
    "genflavor": hist.axis.IntCategory([0, 1, 2, 3], name="genflavor", label="Gen Flavor"),
}

# add more as needed
axis_to_column = {
    "pt1": "FatJet0_pt",
    "pt2": "FatJet1_pt",
    "msd1": "FatJet0_msd",
    "mass1": "FatJet0_pnetMass",
    "category": "category",
    "genflavor": "GenFlavor",
}


# --- FUNCTION MODIFIED ---
# It now takes an existing histogram `h` as an argument to fill
def fill_ptbinned_histogram(h, events, axis):
    """
    Fills a histogram with events from a single dataset.
    """
    for _process_name, data in events.items():
        weight_val = data["finalWeight"].astype(float)
        var = data[axis_to_column[axis]]

        isRealData = "GenFlavor" not in data.columns
        genflavordata = (
            data["GenFlavor"].astype(int) if not isRealData else np.zeros_like(var, dtype=int)
        )

        # Event selection
        Txbb = data["FatJet0_pnetTXbb"]
        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]
        pre_selection = (msd > 40) & (msd < 200) & (pt > 300) & (pt < 1200)
        selection_dict = {
            "pass": pre_selection & (Txbb > 0.95),
            "fail": pre_selection & (Txbb < 0.95),
        }

        # Fill histograms
        for category, selection in selection_dict.items():
            h.fill(
                var[selection],
                pt[selection],
                category=category,
                genflavor=genflavordata[selection],
                weight=weight_val[selection],
            )
    return h


def main(args):
    year = args.year
    region = args.region

    MAIN_DIR = "/eos/uscms/store/group/lpchbbrun3/"
    dir_name = "gmachado/25Aug27_v12"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    load_columns_mc = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb",
        "GenFlavor",
    ]
    load_columns_data = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb",
    ]
    filters = None

    histograms = {}
    data_dir = Path(path_to_dir) / year
    samples = {
        **common_mc,
        "data": data_by_year[year],
    }

    # --- MAIN LOOP RESTRUCTURED ---
    # Loop through each process
    for process, datasets in samples.items():
        load_columns = load_columns_data if process == "data" else load_columns_mc
        print(f"Processing {process} for year {year}...")

        # Create a new histogram for each process
        h = hist.Hist(
            axis_to_histaxis["msd1"],
            axis_to_histaxis["pt1"],
            axis_to_histaxis["category"],
            axis_to_histaxis["genflavor"],
        )

        # Loop through each dataset within the process
        for dataset in datasets:
            # Load only one dataset at a time to save memory
            search_path = Path(data_dir / dataset / "parquet" / region)
            print(f"\n[DEBUG] Script is searching for files in: {search_path}\n")

            events = utils.load_samples(
                data_dir,
                {process: [dataset]},  # Pass a list with a single dataset
                columns=load_columns,
                region=region,
                filters=filters,
            )

            if not events:
                print(f"No events found for dataset {dataset} in year {year}. Skipping.")
                continue

            # Fill the histogram with the events from this single dataset
            h = fill_ptbinned_histogram(h, events, "msd1")

        # --- ADDED CHECK ---
        # Only add the histogram to our dictionary if it has entries
        if h.sum() == 0:
            print(
                f"WARNING: No events were found for the entire '{process}' process group. Skipping."
            )
            continue
        # Add the fully filled histogram for the process to the dictionary
        histograms[process] = h

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"histograms_{year}_{region}.pkl"

    with output_file.open("wb") as f:
        pickle.dump(histograms, f)

    print(f"Histograms saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument(
        "--region",
        help="region",
        type=str,
        required=True,
        choices=[
            "signal-all",
            # "signal-ggf",
            # "control-zgamma",
            # "control-tt",
            # "signal-vh",
            # "signal-vbf",
        ],  # add more as needed
    )
    parser.add_argument(
        "--outdir", help="Output directory to save histograms.", type=str, default="histograms"
    )
    args = parser.parse_args()

    main(args)
