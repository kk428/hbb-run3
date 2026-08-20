#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import json

import hist
import numpy as np
from common import common_mc, data_by_year

from hbb import utils
from axis_info import axis_to_histaxis, axis_to_column

# --- FUNCTION MODIFIED ---
# It now takes an existing histogram `h` as an argument to fill
def fill_ptbinned_histogram(events, axis_label, region, dataset, isData=False, do_cutflow=False):
    """
    Fills a histogram with events from a single dataset.
    """
    column = axis_to_column[axis_label]
            
    h = hist.Hist(
        axis_to_histaxis[axis_label],
        axis_to_histaxis["category"],
        storage=hist.storage.Weight()
    )

    if do_cutflow:
        cutflow = hist.Hist(
            hist.axis.StrCategory([], growth=True, name="region", label="Region"),
            hist.axis.StrCategory([], growth=True, name="category", label="Category"),
            hist.axis.StrCategory([], growth=True, name="dataset", label="Dataset"),
            hist.axis.Regular(4, 0, 10000, name="h_pt", label="Higgs AK8 pt"),
            hist.axis.Regular(15, 0, 15, name="cut", label="Cut index"),
            storage=hist.storage.Weight()
        )

    for _process_name, data in events.items():
        weight_val = data["finalWeight"].astype(float)
        if isData:
            # print("DATA HOORAY")
            weight_val = data['weight_noxsec'].astype(float)
        var = data[column]

        # Event selection
        Txbb = data["HiggsAK8_ParTPXbbVsQCD"]
        msd = data["HiggsAK8_msd"]
        pt = data["HiggsAK8_pt"]
        mjj = data["VBFPair_mjj"]
        deta = data["VBFPair_deta"]
        ll_flav = data["LeadingLep_flavor"]
        sl_flav = data["SubLeadingLep_flavor"]
        pre_selection = (msd >= 40) & (pt >= 250) # & (mjj > 250) & (deta > 2.5)

        all_selections = {
            "preselection": pre_selection,
            "no_cut": np.ones(len(pt), dtype=bool),
            "pt_50": (pt > 50),
            "hbb_score_0p1": (Txbb > 0.1),
            "vbf_deta_2p5": (deta > 2.5),
            "vbf_mjj_250": (mjj > 250),
            "same_flavor": (ll_flav==sl_flav),
            "opposite_flavor": (ll_flav!=sl_flav),
            "both_electrons": (ll_flav==sl_flav) & (ll_flav == 1.),
            "both_muons": (ll_flav==sl_flav) & (ll_flav == 0.)
        }

        selection_dict = {
            "preselection": ["preselection"],
            "preselection_ee": ["preselection", "both_electrons"],
            "preselection_mumu": ["preselection", "both_muons"],
            "preselection_emu": ["preselection", "opposite_flavor"],
            "signal_region": ["preselection", "hbb_score_0p1", "vbf_deta_2p5", "vbf_mjj_250"],
            "pt50_test": ["no_cut", "pt_50", "preselection"],
        }

        # Fill histograms
        for category, selection_list in selection_dict.items():
            full_selection = None
            for i, selection in enumerate(selection_list):
                if i==0:
                    full_selection = all_selections[selection]
                else:
                    full_selection = full_selection & all_selections[selection]

                if do_cutflow:
                    cutflow.fill(
                        region = region,
                        dataset = dataset,
                        category = category,
                        h_pt = pt[full_selection],
                        cut = i,
                        weight=weight_val[full_selection]
                    )


            h.fill(
                var[full_selection],
                # pt1=pt[selection],
                category=category,
                # genflavor=genflavordata[selection],
                weight=weight_val[full_selection],
            )
    if do_cutflow:
        return cutflow
    return h


def main(args):
    year = args.year
    region = args.region

    MAIN_DIR = "/eos/uscms/store/user/kkrzyzan/"
    dir_name = "25Sep23_main_v12"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    load_columns = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb",
    ]
    for axis in axis_to_column.keys():
        column = axis_to_column[axis]
        if column not in load_columns:
            load_columns.append(column) 
    filters = None

    histograms = {}
    data_dir = Path(path_to_dir) / year
    samples = {
        **common_mc,
        "data": data_by_year[year],
    }

    histograms = {column: {} for column in axis_to_column.keys()}

    # --- MAIN LOOP RESTRUCTURED ---
    # Loop through each process
    for process, datasets in samples.items():
        print(f"Processing {process} for year {year}...")


        # Loop through each dataset within the process
        for dataset in datasets:
            # Load only one dataset at a time to save memory
            search_path = Path(data_dir / dataset / "parquet" / region / "nominal")
            # print(f"\n[DEBUG] Script is searching for files in: {search_path}\n")

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

            cflow = fill_ptbinned_histogram(
                    events=events, 
                    axis_label="msd1",
                    # axis_label=axis, 
                    region=region,
                    dataset=dataset,
                    isData="_Run20" in dataset,
                    do_cutflow=True
                    )
            picklename = f"{data_dir}/{dataset}/pickles/postprocessing_cutflow.pkl"
            picklename.parent.mkdir(parents=True, exist_ok=True)
            with picklename.open("wb") as f:
                pickle.dump(cflow, f, protocol=-1)
            pickle.dump(cflow, cflowfile, protocol=-1)

            for axis in axis_to_column.keys():
                column = axis_to_column[axis]
                # Fill the histogram with the events from this single dataset
                h = fill_ptbinned_histogram(
                    events=events, 
                    axis_label=axis, 
                    region=region,
                    dataset=dataset,
                    isData="_Run20" in dataset,
                    do_cutflow=False
                    )

                if h.sum() == 0:
                    continue

                if not process in histograms[axis]:
                    histograms[axis][process] = h
                else:
                    histograms[axis][process] += h

        # --- ADDED CHECK ---
        # Only add the histogram to our dictionary if it has entries

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
        choices=["2018", "2016", "2016APV", "2017","2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument(
        "--region",
        help="region",
        type=str,
        required=True,
        choices=[
            "signal-wwh",
            "signal-all",
            "signal-vbf",
            "signal-zzh-1FJ",
            "signal-wzh-zzh-2FJ"
        ],
    )
    parser.add_argument(
        "--outdir", help="Output directory to save histograms.", type=str, default="histograms"
    )
    args = parser.parse_args()

    main(args)