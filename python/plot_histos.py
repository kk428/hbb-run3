#!/usr/bin/env python3
"""
Unified plotting script for the Hbb analysis.

This script serves as a central manager for creating various types of plots
from the histogram `.pkl` files produced by `make_histos.py`. It can generate
three main types of plots, selectable via the `--plot-type` argument:

1.  `process`: Standard stacked data vs. Monte Carlo plots, with samples
    grouped by their physics process (e.g., Top, W+jets, Z+jets).

2.  `flavor`: Detailed stacked plots where the W+jets and Z+jets backgrounds
    are further broken down by their generator-level quark flavor (b-jet,
    c-jet, light-jet).

3.  `qcd_shape`: A diagnostic plot comparing the normalized shapes of the
    QCD MC distribution in the 'pass' and 'fail' regions to validate
    background estimation techniques.

Example usage:
# To plot stacked by process for a single year
python python/plot_manager.py --year 2022EE --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type process

# To plot with flavor breakdown for multiple years combined
python python/plot_manager.py --year 2022EE 2023 --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type flavor

# To plot the QCD shape comparison
python python/plot_manager.py --year 2022EE --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type qcd_shape --norm-type density
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import yaml
from plotting import ratio_plot

from hbb.common_vars import LUMI

hep.style.use("CMS")

# --- Globals for Plotting Logic ---
process_grouping = {
    "QCD": ["qcd"],
    "Z->qq": ["zjets"],
    "W->qq": ["wjets"],
    "Top": ["tt", "singletop"],
    "Other": ["diboson", "ewkv"],
    "H->bb": ["ggf-hbb", "vbf-hbb", "vh-hbb"],
}


mass_lo = 115  # GeV, lower edge of the mass window to blind
mass_hi = 135  # GeV, upper edge of the mass window to blind


flavor_map = {3: "b-jet", 2: "c-jet", 1: "light-jet"}


# --- Function 1: Plotting Stacked by Process ---
def plot_by_process(hists, category, year_str, outdir, region, style):
    """Plots a stacked histogram for each pt bin, with grouping handled by the style file."""

    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        # Project all raw histograms to 1D for this pt bin
        histograms_to_plot = {}
        for process, h in hists.items():
            h_proj = h[:, i_start, category, :].project("msd1")

            if process == "data":
                # Blind the mass window
                edges = h_proj.axes[0].edges
                mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
                data_val = h_proj.values()
                data_val[mask] = 0
                h_proj.values()[:] = data_val

            histograms_to_plot[process] = h_proj

        # Define the lists of signals and backgrounds using the final group names
        # These names must have a corresponding entry in the style file with a 'contains' key
        bkg_order = ["zjets", "wjets", "other", "top"]
        signals = ["hbb"]

        legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=signals,
            bkgs=bkg_order,
            onto="qcd",
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
        )

        luminosity = sum(LUMI[y] / 1000.0 for y in year_str.split("-") if y != "all")
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
        )

        output_name = f"{outdir}/{year_str}_{region}_{category}_process_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Function 2: Plotting Stacked by Flavor ---
def plot_by_flavor(hists, category, year_str, outdir, region, style):
    """Plots a stacked histogram for each pt bin, splitting W/Z jets by flavor."""
    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    mass_lo = 115
    mass_hi = 135

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        histograms_to_plot = {}
        for process, h in hists.items():
            if process in ["wjets", "zjets"]:
                h_2d = h[:, i_start, category, :]
                for flavor_code, flavor_name in flavor_map.items():
                    new_key = f"{process}_{flavor_name}"
                    histograms_to_plot[new_key] = h_2d[:, hist.loc(flavor_code)]
            else:
                h_proj = h[:, i_start, category, :].project("msd1")
                # --- ADDED: Data Blinding ---
                if process == "data":
                    edges = h_proj.axes[0].edges
                    mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
                    data_val = h_proj.values()
                    data_val[mask] = 0
                    h_proj.values()[:] = data_val
                histograms_to_plot[process] = h_proj

        bkg_order = [
            "hbb",
            "other",
            "top",
            "wjets_light-jet",
            "wjets_c-jet",
            "zjets_light-jet",
            "zjets_c-jet",
            "zjets_b-jet",
        ]

        # --- ADDED: Legend Title and Sorting ---
        legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=[],
            bkgs=bkg_order,
            onto="qcd",
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
        )

        luminosity = sum(LUMI[y] / 1000.0 for y in year_str.split("-") if y != "all-years")
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
            loc=0,
        )

        output_name = f"{outdir}/{year_str}_{region}_{category}_flavor_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Function 3: QCD Pass/Fail Shape Comparison ---
def plot_qcd_shapes(hists, year_str, outdir, region, norm_type):
    """For each pt bin, plots the normalized 'pass' and 'fail' distributions for the QCD sample."""
    if "qcd" not in hists:
        print("No 'qcd' histogram found in the input file. Exiting.")
        return
    h_qcd = hists["qcd"]
    pt_axis = h_qcd.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"Processing pt bin: {pt_low} - {pt_high}")

        h_pass = h_qcd[:, i_start, "pass", :].project("msd1")
        h_fail = h_qcd[:, i_start, "fail", :].project("msd1")

        if h_pass.sum() == 0 or h_fail.sum() == 0:
            print("  Skipping pt bin due to zero events in pass or fail.")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # --- UPDATED LOGIC ---
        if norm_type == "shape":
            # Use density=True to automatically create a probability density
            hep.histplot(
                h_fail,
                ax=ax,
                label="QCD MC fail",
                color="blue",
                histtype="errorbar",
                yerr=True,
                density=True,
            )
            hep.histplot(
                h_pass,
                ax=ax,
                label="QCD MC pass",
                color="black",
                histtype="errorbar",
                yerr=True,
                density=True,
            )
            ylabel = "Probability Density"

        elif norm_type == "density":
            # Keep the manual logic for physical density (scaling fail to pass)
            bin_width = h_pass.axes[0].widths[0]
            pass_yield = h_pass.sum()
            fail_yield = h_fail.sum()
            h_fail_scaled = h_fail * (pass_yield / fail_yield)
            h_pass_toplot = h_pass / bin_width
            h_fail_toplot = h_fail_scaled / bin_width
            ylabel = f"Events / {bin_width:g} GeV"
            hep.histplot(
                h_fail_toplot,
                ax=ax,
                label="QCD MC fail",
                color="blue",
                histtype="errorbar",
                yerr=True,
            )
            hep.histplot(
                h_pass_toplot,
                ax=ax,
                label="QCD MC pass",
                color="black",
                histtype="errorbar",
                yerr=True,
            )
        # --- END UPDATED LOGIC ---

        ax.set_xlabel("Jet $m_{sd}$ [GeV]")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        hep.cms.label("Private Work", data=False, ax=ax, com=13.6, year=year_str)

        ax.legend(
            title=f"{pt_low:g} < $p_T$ < {pt_high:g} GeV",
            prop={"size": 14},
            title_fontsize=16,
            loc="upper right",
        )

        output_name = f"{outdir}/{year_str}_{region}_qcd_{norm_type}_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_name}")
        plt.close(fig)


# --- Main Function: The Control Center ---
def main(args):
    histograms = {}
    year_str = "all-years" if len(args.year) > 3 else "-".join(args.year)

    for year in args.year:
        pkl_path = Path(args.indir) / f"histograms_{year}_{args.region}.pkl"
        if not pkl_path.exists():
            print(f"Error: File not found at {pkl_path}. Skipping.")
            continue
        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)
            for process, h in histograms_tmp.items():
                if process in histograms:
                    histograms[process] += h
                else:
                    histograms[process] = h

    if not histograms:
        print("No histograms were loaded. Exiting.")
        return

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_path = Path("style_hbb.yaml")
    with style_path.open() as f:
        style = yaml.safe_load(f)

    # Call the correct plotting function based on --plot-type
    if args.plot_type == "process":
        for category in ["pass", "fail"]:
            print(f"Plotting histograms by process for category: {category}, year: {year_str}...")
            plot_by_process(histograms, category, year_str, args.outdir, args.region, style)
    elif args.plot_type == "flavor":
        for category in ["pass", "fail"]:
            print(f"Plotting histograms by flavor for category: {category}, year: {year_str}...")
            plot_by_flavor(histograms, category, year_str, args.outdir, args.region, style)
    elif args.plot_type == "qcd_shape":
        print(f"Plotting QCD pass/fail shapes for year: {year_str}...")
        plot_qcd_shapes(histograms, year_str, args.outdir, args.region, args.norm_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified plotting script for Hbb analysis.")
    parser.add_argument(
        "--year",
        help="List of years",
        type=str,
        required=True,
        nargs="+",  # Accepts one or more arguments, if more arguments are given, then histograms are summed
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument("--indir", help="Input directory for .pkl files", type=str, required=True)
    parser.add_argument("--outdir", help="Output directory for plots", type=str, required=True)
    parser.add_argument("--region", help="Analysis region", type=str, required=True)
    parser.add_argument(
        "--plot-type",
        help="Type of plot to produce",
        type=str,
        default="process",
        choices=["process", "flavor", "qcd_shape"],
    )
    parser.add_argument(
        "--norm-type",
        help="Normalization for QCD shape plot ('shape' or 'density')",
        type=str,
        default="shape",
        choices=["shape", "density"],
    )
    args = parser.parse_args()
    main(args)
