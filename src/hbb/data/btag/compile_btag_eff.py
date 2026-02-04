import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from coffea.lookup_tools.dense_lookup import dense_lookup

plt.style.use(hep.style.CMS)

flavor_names = ['light', 'c', 'b']

hard_flavor_edges = np.array([0, 4, 5, 6]) 
hard_pt_edges = np.array([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000])
hard_abseta_edges = np.linspace(0, 2.5, 5)


def plot_efficiency_2d(hist, tagger, dataset, year, output_dir, empty=False):

    if not empty:
        try:
            efficiencyinfo = hist[{'tagger': tagger}]
        except KeyError:
            print(f"  Warning: Tagger '{tagger}' not found in {dataset}, creating empty plots")
            plot_efficiency_2d(hist, tagger, dataset, year, output_dir, empty=True)
            return
        
        flavor_axis = efficiencyinfo.axes['flavor']
        pt_axis = efficiencyinfo.axes['pt']
        abseta_axis = efficiencyinfo.axes['abseta']

    else:
        pt_edges = hard_pt_edges
        abseta_edges = hard_abseta_edges
    
    if not empty:
        num_flavor_bins = len(flavor_axis.edges) - 1
        pt_edges = pt_axis.edges
        abseta_edges = abseta_axis.edges
    else:
        num_flavor_bins = len(hard_flavor_edges) - 1
    
    for flavor_idx in range(num_flavor_bins):
        flavor_name = flavor_names[flavor_idx]
        
        if not empty:
            passed = efficiencyinfo[{'passWP': 1, 'flavor': flavor_idx}].values()
            total = efficiencyinfo[{'passWP': sum, 'flavor': flavor_idx}].values()
        
        else:
            shape = (len(pt_edges)-1, len(abseta_edges)-1)
            passed = np.zeros(shape)
            total = np.zeros(shape)

        eff = np.divide(passed, total, out=np.zeros_like(passed), where=total!=0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        X, Y = np.meshgrid(abseta_edges, pt_edges)
        im = ax.pcolormesh(X, Y, eff, cmap='viridis', vmin=0, vmax=1)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Efficiency', fontsize=14)
        
        ax.set_xlabel('Jet $|\eta|$', fontsize=14)
        ax.set_ylabel('Jet $p_T$ [GeV]', fontsize=14)
        ax.set_title(f'{dataset} - {tagger} - {flavor_name} jets ({year})', fontsize=16)
        
        for i in range(len(pt_edges) - 1):
            for j in range(len(abseta_edges) - 1):
                text_color = 'white' if eff[i, j] < 0.5 else 'black'
                ax.text((abseta_edges[j] + abseta_edges[j+1])/2, 
                       (pt_edges[i] + pt_edges[i+1])/2,
                       f'eff = {eff[i, j]:.3f},   total = {total[i, j]}', 
                       ha='center', va='center', 
                       color=text_color, fontsize=8)
        
        output_file = output_dir / f'{dataset}_{flavor_name}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot: {output_file}")


def main(args):
    year = args.year
    tagger = args.tagger

    base_dir = Path(args.indir) / year
    plot_dir = Path(f'plots/{year}')
    plot_dir.mkdir(exist_ok=True, parents=True)

    merged_dict = {}
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue

        dataset = subdir.name
        print(f"Loading {dataset}")

        pickles_dir = subdir / 'pickles'
        for pkl_file in pickles_dir.glob('*.pkl'):
            with open(pkl_file, 'rb') as f:
                hist_dict = pickle.load(f)
        
            hist = hist_dict[dataset]['nominal']

            def add_hist(key):
                if key in merged_dict:
                    merged_dict[key] += hist
                else:
                    merged_dict[key] = hist.copy()

            add_hist(dataset)

            if "TTto" in dataset or "QCD" in dataset:
                add_hist("TTbar+QCD")

            if "TTto" in dataset:
                add_hist("TTbar")

            if "QCD" in dataset:
                add_hist("QCD")
                    
    out_dict = {}
    lookup_sets = ["TTbar+QCD", "TTbar", "QCD"]
    for dataset, hist in merged_dict.items():

        #plot everything set loaded and compiled
        plot_efficiency_2d(hist, tagger, dataset, year, plot_dir)

        #only create lookup table of "TTbar+QCD", "TTbar", "QCD"
        #user chooses in corrections.py
        if dataset in lookup_sets:
            print(f"Creating lookup table for {dataset}")

            try:
                efficiencyinfo = hist[{'tagger': tagger}]
                eff = efficiencyinfo[{'passWP': 1}].values() / efficiencyinfo[{'passWP': sum}].values()
                edges = [efficiencyinfo[{'passWP': 1}].axes[ax_name].edges for ax_name in ['flavor', 'pt', 'abseta']]
                out_dict[dataset] = dense_lookup(eff, edges)

            except KeyError:
                print(f"  Warning: Tagger '{tagger}' not found in {dataset}, check your pickle files")
                return

    outfile = f'./mc_eff_{tagger}_{year}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(out_dict, f)
    print(f"Saved {outfile}")

    print("\n\n-------- Lookup Table Contents --------")
    print(out_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accumulate b-tagger efficiencies.")
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument(
        "--indir",
        help="indir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tagger",
        help="AK4 tagger to calculate efficiency. See taggers.py for integrated options.",
        type=str,
        required=True,
        choices=["btagPNetB", "btagUParTAK4B"],
    )
    args = parser.parse_args()

    main(args)