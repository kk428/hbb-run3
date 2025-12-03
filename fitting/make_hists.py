#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import hist
import json
import os
import uproot

from hbb import utils

def fill_hists(outdict, events, region, reg_cfg, obs_cfg, qq_true, s, j_var=None):

    h = hist.Hist(hist.axis.Regular(obs_cfg["nbins"], obs_cfg["min"], obs_cfg["max"], name=obs_cfg["name"], label=obs_cfg["name"]))

    bins_list = reg_cfg["bins"]
    bin_pname = reg_cfg["bin_pname"]
    str_bin_br = reg_cfg["branch_name"]

    for _process_name, data in events.items():

        if j_var or s == "nominal":
            weight_val = data["finalWeight"].astype(float)
            if j_var:
                s = j_var
        else:
            weight_val = data[s].astype(float) / data["sum_genWeight"].astype(float)

        bin_br = data[str_bin_br]
        obs_br = data[obs_cfg["branch_name"]]

        Txbb = data["FatJet0_ParTPXbbVsQCD"]
        Txcc = data["FatJet0_ParTPXccVsQCD"]
        Txbbxcc = data["FatJet0_ParTPXbbXcc"]
        genf = data["GenFlavor"]

        pre_selection = (obs_br > obs_cfg["min"]) & (obs_br < obs_cfg["max"])

        selection_dict = {
            "pass_bb": pre_selection & (Txbbxcc  > 0.82) & (Txbb  > Txcc),
            "pass_cc": pre_selection & (Txbbxcc  > 0.82) & (Txcc  > Txbb),
            "fail": pre_selection & (Txbbxcc <= 0.82),
            "pass": pre_selection & (Txbbxcc  > 0.82)
        }

        cut_bb = (genf == 3)
        cut_qq = (genf > 0) & (genf < 3)
        cut_c = (genf == 2)
        cut_light = (genf > 0) & (genf < 2)

        def fill_h(name, sel):
            h.view()[:] = 0
            h.fill(
                obs_br[sel],
                weight=weight_val[sel],
            )
            if not name in outdict:
                outdict[name] = h.copy()
            else:
                outdict[name] += h.copy()
            return

        for i in range(len(bins_list) - 1):
            bin_cut = (bin_br > bins_list[i]) & (bin_br < bins_list[i+1]) & pre_selection

            for category, selection in selection_dict.items():
                if qq_true:
                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}_{s}"
                    fill_h(name, (selection & bin_cut & cut_qq))

                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}bb_{s}"
                    fill_h(name, (selection & bin_cut & cut_bb))

                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}c_{s}"
                    fill_h(name, (selection & bin_cut & cut_c))

                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}light_{s}"
                    fill_h(name, (selection & bin_cut & cut_light))

                else:

                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}_{s}"
                    fill_h(name, (selection & bin_cut))

    return outdict

def main(args):
    year = args.year
    tag = args.tag

    path_to_dir = f"/eos/uscms/store/group/lpchbbrun3/skims/{tag}"
    
    samples_qq = ['Wjets','Zjets','EWKW','EWKZ','EWKV']
    
    columns = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_ParTPXbbVsQCD",
        "FatJet0_ParTPXccVsQCD",
        "FatJet0_ParTPXbbXcc",
        "VBFPair_mjj",
        "GenFlavor",
    ]

    energy_variations = [
        None,
        "JES",
        "JER",
        "UES",
        'MuonPTScale',
        'MuonPTRes'
    ]

    systs = [
        'ISRPartonShower',
        'FSRPartonShower', 
        'aS_weight',
        'PDF_weight',  
        'PDFaS_weight', 
        'scalevar_7pt', 
        'scalevar_3pt',
        'pileup',
        'btagSFb_correlated',
        'btagSFc_correlated',
        'btagSFlight_correlated'
    ]

    year_systs = [
        'btagSFb',
        'btagSFc',
        'btagSFlight',
    ]

    cr_systs = {
        "mucr" : ["muon_ID", "muon_ISO"],
        "zgcr" : ["photon_ID"]
    }

    data_dirs = {year: Path(path_to_dir) / year}
    if args.year == "Run3":
        data_dirs={
            "2022":Path(path_to_dir) / "2022",
            "2022EE":Path(path_to_dir) / "2022EE",
            "2023":Path(path_to_dir) / "2023",
            "2023BPix":Path(path_to_dir) / "2023BPix",
        }

    out_path = f"results/{tag}/{year}"
    output_file = f"{out_path}/signalregion.root"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if os.path.isfile(output_file):
        os.remove(output_file)
    fout = uproot.create(output_file)

    # So I can remember the settings I used for each set of results produced
    os.popen(f'cp setup.json {out_path}')
    with open('setup.json') as f:
        setup = json.load(f)
        cats = setup["categories"]
        obs_cfg = setup["observable"]

    with open('pmap_run3.json') as f:
        pmap = json.load(f)
    
    filters = [
       ("FatJet0_pt", ">", 450),
       ("FatJet0_pt", "<", 1200),
       ("VBFPair_mjj", ">", -2),
       ("VBFPair_mjj", "<", 13000),
    ]

    if not obs_cfg["branch_name"] in columns:
        columns.append(obs_cfg["branch_name"])

    out_hists = {}
    for process, datasets in pmap.items():
        for dataset in datasets:
            for reg, cfg in cats.items():
                for year, data_dir in data_dirs.items():
                    for var in energy_variations:

                        if not var:
                            c_systs_full = systs + [f"{syst}_{year}" for syst in year_systs]
                            c_systs_full = c_systs_full + cr_systs[reg] if reg in cr_systs else c_systs_full
                            c_systs_full = [f"{syst}{dir}" for syst in c_systs_full for dir in ["Up", "Down"]]
                            events = utils.load_samples(
                                data_dir,
                                {process: [dataset]},
                                columns=columns if "data" in process else columns+c_systs_full,
                                region=cfg["name"],
                                filters=filters,
                                variation=var
                            )

                            if not events:
                                continue

                            fill_hists(out_hists, events, reg, cfg, obs_cfg, (process in samples_qq), "nominal", var)

                            if "data" not in process:
                                for syst in c_systs_full:
                                    fill_hists(out_hists, events, reg, cfg, obs_cfg, (process in samples_qq), f"{syst}", var)

                        else:   #energy variations
                            for direction in ["Up", "Down"]:
                                var_jerc = f"{var}{direction}"

                                events = utils.load_samples(
                                    data_dir,
                                    {process: [dataset]},
                                    columns=columns,
                                    region=cfg["name"],
                                    filters=filters,
                                    variation=var_jerc
                                )

                                if not events:
                                    continue

                                fill_hists(out_hists, events, reg, cfg, obs_cfg, (process in samples_qq), var_jerc, var_jerc)


    for name, h in out_hists.items():
        fout[name] = h

    print(f"Histograms saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "Run3"],
    )
    parser.add_argument(
        "--tag",
        help="tag",
        type=str,
        required=True
    )
    args = parser.parse_args()

    main(args)
