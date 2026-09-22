import os, subprocess
import pandas as pd
import pickle
from common import common_mc, data_by_year
from pathlib import Path
import numpy as np


def build_inverse_map(year):

    outmap = {}
    for dataset in data_by_year[year]:
        outmap[dataset]= "data"
    for process, datasets in common_mc.items(): #common_mc[year].items():
        for dataset in datasets:
            outmap[dataset]= process
    return outmap
        
    if data:
        
        return {
            datasets: "data"
            for datasets in common_mc[year]
        }

    return {
        dataset: process
        for process, datasets in common_mc[year].items()
        for dataset in datasets
    }


# year_str = 'Run2'
# years = ['2016APV','2016','2017','2018']
year_str = '2023'
years = ['2023']
region = 'signal-all'
pp_category = 'signal_region'
tag = '25Sep23_output'
coffeadir_prefix = "/eos/uscms/store/user/kkrzyzan/25Sep23_main_v12"
outdir=Path(f'plots/{tag}/{year_str}')
outdir.mkdir(parents=True, exist_ok=True)

sigs = ["vbs-hvv-c2v-1p0-c3-10p0", "vbs-hvv-c2v-1p0-c3-1p0", "vbs-hvv-c2v-1p5-c3-1p0"]
bkgs = ["zjets","tt","ewkv","higgs","diboson","qcd","singletop","wjets","triboson"]
bkg_unc = [f"{bkg}_unc" for bkg in bkgs]
print_order = ["data", "tot_bkg", "vbs-hvv-c2v-1p0-c3-10p0", "vbs-hvv-c2v-1p0-c3-1p0", "vbs-hvv-c2v-1p5-c3-1p0"]
print_order_all = print_order + bkgs

cut_index = [   
    "RDF",
    "twoleptons",
    "oppsign",
    "lepdR",
    "notZpeak",
    "onegoodAK8",
    "antiak4btagMedium",
    "2ak4s"
]

cut_pp_index = [
    "preselection",
    "hbb_score_0p1", 
    "vbf_deta_2p5", 
    "vbf_mjj_250"
]


cutflow = {}
cutflow_pp = {}
h = {}
h_pp = {}
inverse_map = {}
inverse_map_data = {}
total_map = {}

for year in years:
    # inverse_map[year] = build_inverse_map(common_mc, year)
    # inverse_map_data[year] = build_inverse_map(data_by_year, year, data=True)
    total_map[year] = build_inverse_map(year)

    procs = subprocess.getoutput(f"ls {coffeadir_prefix}/{year}").split()
    #LOADING PROCESSOR CUTFLOW - created in categorizer.py
    for proc in procs:
    
        filename = f"{coffeadir_prefix}/{year}/{proc}/pickles/out_0.pkl"
        
        print(filename)
        if os.path.isfile(filename):
            with open(filename, 'rb') as openfile:
                data = pickle.load(openfile)
                
            print(proc)
            if not year in cutflow:
                cutflow[year] = data[proc]['cutflow']
            else:
                cutflow[year] += data[proc]['cutflow']
                
        else:
            print(f'Missing file: {proc}')

    #LOADING POSTPROCESSOR CUTFLOW - created in python/make_histos.py

        filename = f"{coffeadir_prefix}/{year}/{proc}/pickles/histograms_2023_signal-all.pkl" 

        if os.path.isfile(filename):
            with open(filename, 'rb') as openfile:
                data = pickle.load(openfile)
                
            if not year in cutflow_pp:
                cutflow_pp[year] = data
            else:
                cutflow_pp[year] += data
                
        else:
            print(f'Missing post processing file: {proc}')
        
    

# merge cutflows into processes
h_dict = {}
hpp_dict = {}
for year in years:
    print(cutflow[year])
    print(cutflow_pp[year])
    h[year] = cutflow[year].integrate('h_pt').integrate('region',[region])
    h_pp[year] = cutflow_pp[year].integrate('h_pt').integrate('region',[region])

    for i, proc in enumerate(h[year].axes["dataset"]):

        if not total_map[year][proc] in h_dict:
            h_dict[total_map[year][proc]] = h[year][{"dataset": proc}]
        else:
            h_dict[total_map[year][proc]] += h[year][{"dataset": proc}]

    for i, proc in enumerate(h_pp[year].axes["dataset"]):

        if not total_map[year][proc] in hpp_dict:
            hpp_dict[total_map[year][proc]] = h_pp[year][{"dataset": proc}][{"category": pp_category}]
        else:
            hpp_dict[total_map[year][proc]] += h_pp[year][{"dataset": proc}][{"category": pp_category}]


# -- Cutflow data frame manipulation

df_pre_tot = pd.DataFrame([])   # pre processing
df_pp_tot = pd.DataFrame([])    # post processing
in_zeros = [0.0] * 15

for proc in h_dict:
    df_pre_tot[proc] = h_dict[proc].values()
    df_pp_tot[proc] = hpp_dict[proc].values() if proc in hpp_dict else in_zeros

    #weighted errors
    df_pre_tot[f'{proc}_unc'] = np.sqrt(h_dict[proc].variances())
    df_pp_tot[f'{proc}_unc'] = np.sqrt(hpp_dict[proc].variances()) if proc in hpp_dict else in_zeros

df_pre_tot = df_pre_tot[:-(15 - len(cut_index))].astype('float')
df_pre_tot.index = cut_index
df_pre_tot["tot_bkg"]=df_pre_tot.reindex(columns=bkgs).sum(axis=1 ) #total bkg mc
df_pre_tot["tot_bkg_unc"]= np.sqrt((df_pre_tot.reindex(columns=bkg_unc) ** 2).sum(axis=1)) #propogated error

#save pre-processing cutflow
df_pre_tot.to_string(buf=f'{outdir}/cutflow-preprocessing.txt')
df_pre_tot[print_order].to_string(buf=f'{outdir}/cutflow-min-preprocessing.txt')

df_pp_tot = df_pp_tot[:-(15 - len(cut_pp_index))].astype('float')
df_pp_tot.index = cut_pp_index
df_pp_tot["tot_bkg"]=df_pp_tot.reindex(columns=bkgs).sum(axis=1 ) #total bkg mc
df_pp_tot["tot_bkg_unc"]= np.sqrt((df_pp_tot.reindex(columns=bkg_unc) ** 2).sum(axis=1)) #propogated error
df_finalregion = pd.concat([df_pre_tot, df_pp_tot])

#save sr cutflow
df_finalregion.to_string(buf=f'{outdir}/cutflow-signalregion.txt')
df_finalregion[print_order].to_string(buf=f'{outdir}/cutflow-min-signalregion.txt')

#save only bkg mc
df_finalregion.reindex(columns=bkgs).to_string(buf=f'{outdir}/cutflow-signalregion-bkg.txt')