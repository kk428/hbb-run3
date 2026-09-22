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
    #"mass1": hist.axis.Regular(30, 0, 200, name="mass1", label="Jet 0 PNet mass [GeV]"),
    "category": hist.axis.StrCategory([], name="category", label="Category", growth=True),
    "genflavor": hist.axis.IntCategory([0, 1, 2, 3], name="genflavor", label="Gen Flavor"),
    "cut": hist.axis.IntCategory([], name="cut", label="Cut index", growth=True),
}

# add more as needed
axis_to_column = {
    "pt1": "FatJet0_pt",
    "pt2": "FatJet1_pt",
    "msd1": "FatJet0_msd",
    #"mass1": "FatJet0_pnetMass",
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

        weight_val = data["finalWeight"].to_numpy(dtype=np.float64)
        var  = data.columns.to_numpy(dtype=np.float32)
        Txbb = data["FatJet0_ParTPXbbVsQCD"].to_numpy(dtype=np.float32)
        msd  = data["FatJet0_msd"].to_numpy(dtype=np.float32)
        pt   = data["FatJet0_pt"].to_numpy(dtype=np.float32)
        # weight_val = data["finalWeight"].astype(float)
        # var = data[axis_to_column[axis]]

        isRealData = "GenFlavor" not in data.columns
        genflavordata = (
            data["GenFlavor"].astype(int) if not isRealData else np.zeros_like(var, dtype=int)
        )

        # Event selection
        # Txbb = data["FatJet0_ParTPXbbVsQCD"]
        # msd = data["FatJet0_msd"]
        # pt = data["FatJet0_pt"]
        pre_selection = np.ones_like(pt, dtype=bool) # (msd > 40) & (msd < 200) & (pt > 300) & (pt < 1200)
        selection_dict = {
            "pass": pre_selection, # & (Txbb > 0.95),
            "fail": pre_selection # & (Txbb < 0.95),
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


def fill_cutflow(h, events):
    for _process_name, data in events.items():
        w   = data["finalWeight"].astype(float)
        fj0_pt, fj0_eta, fj0_msd = data["FatJet0_pt"], data["FatJet0_eta"], data["FatJet0_msd"]
        fj1_pt, fj1_eta, fj1_msd = data["FatJet1_pt"], data["FatJet1_eta"], data["FatJet1_msd"]
        met, nfj = data["MET"], data["nFatJet"]
        j0_eta, j1_pt, j1_eta = data["Jet0_eta"], data["Jet1_pt"], data["Jet1_eta"]
        mjj, deta = data["VBFPair_mjj"], data["VBFPair_deta"]

        isData = "GenFlavor" not in data.columns
        gf = np.zeros_like(fj0_pt, dtype=int) if isData else data["GenFlavor"].astype(int)

        stages = [
            ("no_cut",        np.ones_like(fj0_pt, dtype=bool)),
            ("2jets_pt50",    (j1_pt > 50) & (np.abs(j1_eta) < 5.131)),
            ("1fatjet_pt50",  (fj0_pt > 50) & (np.abs(fj0_eta) < 2.5)),
            ("lead_pt200",    fj0_pt >= 200),
            ("trail_pt0",     fj1_pt >= 0),
           # ("met_250",       met < 250),
            ("has_fatjet",    nfj >= 1),
            ("lead_eta2p5",   np.abs(fj0_eta) < 2.5),
            ("lead_pt300",    fj0_pt > 300),
            ("W_mass_window", (fj0_msd > 65) & (fj0_msd < 105)),
            ("sub_pt300",     fj1_pt < 300),
            ("sub_eta2p5",    np.abs(fj1_eta) < 2.5),
            ("sub_msd0",      fj1_msd > 0),
            ("tagjets",       mjj > 0),
            ("opp_hemi",      j0_eta * j1_eta < 0),
            ("deta_2p5",      deta > 2.5),
            ("mjj_500",       mjj > 500),
        ]

        cum = np.ones_like(fj0_pt, dtype=bool)
        for i, (_name, sel) in enumerate(stages):
            cum = cum & np.asarray(sel, dtype=bool)
            h.fill(cut=i, genflavor=gf[cum], weight=w[cum])
    return h

def main(args):
    year = args.year
    region = args.region

    MAIN_DIR = "/eos/uscms/store/user/kkrzyzan/"
    dir_name = "081926-0_v15"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    load_columns_mc = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_ParTPXbbVsQCD",
        "GenFlavor",
    ]
    load_columns_data = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_ParTPXbbVsQCD", 
        "FatJet0_eta", "FatJet1_pt", "FatJet1_eta", "FatJet1_msd",
        "MET", "nFatJet", "nJet",
        "Jet0_pt", "Jet0_eta", "Jet1_pt", "Jet1_eta",
        "VBFPair_mjj", "VBFPair_deta",
    ]
    filters = None

    histograms = {}
    cutflows = {}
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
            storage=hist.storage.Weight(), 
        )

        hcf = hist.Hist(
            hist.axis.IntCategory([], name="cut", growth=True),
            axis_to_histaxis["genflavor"],
            storage=hist.storage.Weight(),
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
            hcf = fill_cutflow(hcf, events)
            

        # --- ADDED CHECK ---
        # Only add the histogram to our dictionary if it has entries
        if h.sum() == 0:
            print(
                f"WARNING: No events were found for the entire '{process}' process group. Skipping."
            )
            continue
        # Add the fully filled histogram for the process to the dictionary
        histograms[process] = h
        cutflows[process] = hcf

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"histograms_{year}_{region}.pkl"

    with output_file.open("wb") as f:
        pickle.dump(histograms, f)

    print(f"Histograms saved to {output_file}")


    cutflow_file = output_dir / f"cutflow_{year}_{region}.pkl"
    with cutflow_file.open("wb") as f:
        pickle.dump(cutflows, f)
    print(f"Cutflows saved to {cutflow_file}")


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
            "signal-ggf",
            "signal-vh",
            "signal-vbf",
            "control-tt",
            "control-zgamma",
        ],
    )
    parser.add_argument(
        "--outdir", help="Output directory to save histograms.", type=str, default="histograms"
    )
    args = parser.parse_args()

    main(args)
