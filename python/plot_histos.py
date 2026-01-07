#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import yaml
from plotting import ratio_plot

from hbb.common_vars import LUMI

hep.style.use("CMS")


def plot_ptbin_stack(hists, category, year, outdir, save_individual):
    pt_axis_name = "pt1"

    # Load style configuration
    # This assumes you have a YAML file named "style_hbb.yaml" in the current directory
    style_path = Path("style_hbb.yaml")
    with style_path.open() as stream:
        style = yaml.safe_load(stream)

    # Axis to project onto
    # proj_axis_name = "msd1"
    mass_lo = 115  # GeV, lower edge of the mass window to blind
    mass_hi = 135  # GeV, upper edge of the mass window to blind

    first_hist = next(iter(hists.values()))

    pt_axis = first_hist.axes[pt_axis_name]
    pt_edges = pt_axis.edges

    print("Axis order:", first_hist.axes.name)

    for i in range(len(pt_edges) - 1):
        histograms_to_plot = {}

        # Localize the pt bin edges
        pt_low = pt_edges[i]
        pt_high = pt_edges[i + 1]
        i_start = pt_axis.index(pt_low)
        i_stop = pt_axis.index(pt_high)
        print(f"Processing pt bin: {pt_low} - {pt_high} (indices: {i_start}, {i_stop})")

        # Select bin range on pt1
        for process, h in hists.items():
            # Select that pt bin using the start index
            h_proj = h[:, i_start, category]

            # For debugging, print the yield of the histogram
            # if process == "ggf-hbb":
            #    print(sum(h_proj.values()))

            if process == "data":
                # Blind the mass window
                edges = h_proj.axes[0].edges  # Assuming the first axis corresponds to mass
                mask = (edges[:-1] >= mass_lo) & (
                    edges[:-1] < mass_hi
                )  # Mask for bins in the specified range
                # Set the values to zero for these bins in both data and total background
                data_val = h_proj.values()
                data_val[mask] = 0  # Set bins to zero in the data histogram
                # Update the data histogram for plotting with masked values set to zero
                h_proj.values()[:] = data_val  # Update h_data with modified values

            histograms_to_plot[process] = h_proj

        # print(histograms_to_plot)

        # TODO: figure out the yield for each the processes (except QCD) so that they are sorted by yield
        # For now, we will use a fixed order
        bkg_order = ["zjets", "wjets", "other", "top"]

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=["hbb"],
            bkgs=bkg_order,
            onto="qcd",
            style=style,
        )
        # CMS label
        luminosity = (
            LUMI[year] / 1000.0
            if "-" not in year
            else sum(LUMI[y] / 1000.0 for y in year.split("-"))
        )
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.0f}",
            com=13.6,
            year=year,
        )
        fig.savefig(
            f"{outdir}/{year}_{category}_ptbin{pt_low}_{pt_high}.png", dpi=300, bbox_inches="tight"
        )

        if save_individual:
            # Save individual histograms for debugging
            for process, histo in histograms_to_plot.items():
                fig_indiv, ax_indiv = plt.subplots(figsize=(8, 6))
                hep.histplot(histo, ax=ax_indiv, histtype="step", color="black")
                ax_indiv.set_title(f"{process} - {category} - ptbin {pt_low}_{pt_high}")
                ax_indiv.set_ylabel("Events")
                ax_indiv.grid(True)
                plt.savefig(
                    f"hist_{process}_{category}_pt{pt_low}_{pt_high}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig_indiv)


def main(args):

    # load histograms
    histograms = {}
    for year in args.year:
        print(f"Loading histograms for year: {year}")
        pkl_path = Path(args.indir) / f"histograms_{year}.pkl"
        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)

        print("Histograms loaded successfully!")
        # Print the structure of the first histogram for debugging
        for h in histograms_tmp.values():
            print(f"Histogram structure: {h} \n")
            break

        if not histograms:
            histograms = histograms_tmp
        else:
            # Combine histograms for the same process across years
            for process, h in histograms_tmp.items():
                if process in histograms:
                    histograms[process] = histograms[process] + h
                else:
                    histograms[process] = h

    print("Processes in histograms:", histograms.keys())
    # Join the years into a single string if multiple years are provided
    year = args.year[0] if len(args.year) == 1 else "-".join(f"{y}" for y in args.year)

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category = "pass"  # Assuming we are plotting the 'pass' category
    print(
        f"Plotting histograms for category: {category}, year: {year}, output directory: {args.outdir} \n"
    )
    plot_ptbin_stack(histograms, category, year, args.outdir, save_individual=args.save_individual)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="List of years",
        type=str,
        required=True,
        nargs="+",  # Accepts one or more arguments, if more arguments are given, then histograms are summed
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument(
        "--indir",
        help="Input directory containing histograms",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        help="Output directory for saving histograms",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_individual",
        help="Save individual histograms for each process",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    main(args)
