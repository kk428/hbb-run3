# Hbb Analysis Plotting

This repository contains scripts to produce histograms from skimmed parquet files and generate various analysis plots.

## 1. Making Histograms from Parquet Files

The `make_histos.py` script reads the skimmed parquet files for a given year and region, applies event selections, and saves the resulting histograms into a `.pkl` file.

### Usage

Run the script specifying the year, region, and an output directory for the `.pkl` files.

**Example:**
```bash
python python/make_histos.py --year 2022EE --region signal-all --outdir histograms/25Aug27
```
## 2. Plotting from Histograms

The `plot_manager.py` script is a unified tool that reads the `.pkl` files created by `make_histos.py` and can generate three different kinds of plots, controlled by the `--plot-type` argument.

### Plot Type 1: Stacked by Process

This is the standard data vs. MC plot, where backgrounds are grouped by their physics process (e.g., Top, W+jets).

**Example (single year):**
```bash
python python/plot_histos.py --year 2022EE --region signal-all --indir histograms/25Aug27 --outdir plots/by_process --plot-type process
```

### Plot Type 2: Stacked by Flavor
This plot provides a more detailed view, splitting the W+jets and Z+jets backgrounds by their quark flavor.

**Example (combining multiple years):**
```bash
python python/plot_histos.py --year 2022EE 2023 --region signal-all --indir histograms/25Aug27 --outdir plots/by_flavor --plot-type flavor

```

### Plot Type 3: QCD Pass/Fail Shape Comparison
This is a diagnostic plot to validate the QCD background estimation method. You can choose between two normalization schemes with the --norm-type flag.

**Example (using density normalization):**
```bash
python python/plot_histos.py --year 2022EE --region signal-all --indir histograms/25Aug27 --outdir plots/qcd_shapes --plot-type qcd_shape --norm-type density

```
