import argparse
import pickle

from pathlib import Path
from coffea.lookup_tools.dense_lookup import dense_lookup


def main(args):
    year = args.year
    tagger = args.tagger

    base_dir = Path(args.indir) / year / "pickles"

    merged_dict = {}
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue

        key = subdir.name
        for pkl_file in subdir.glob('*.pkl'):
            with open(pkl_file, 'rb') as f:
                hist_dict = pickle.load(f)

            hist = hist_dict[key]['nominal']
            if key in merged_dict:
                merged_dict[key] += hist
            else:
                merged_dict[key] = hist.copy()

    out_dict = {}
    for dataset, hist in merged_dict.items():
        efficiencyinfo = hist[{'tagger': tagger}]
        eff = efficiencyinfo[{'passWP': 1}].values() / efficiencyinfo[{'passWP': sum}].values()
        edges = [efficiencyinfo[{'passWP': 1}].axes[ax_name].edges for ax_name in ['flavor', 'pt', 'abseta']]
        out_dict[dataset] = dense_lookup(eff, edges)


    outfile = f'./mc_eff_{tagger}_{year}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(out_dict, f)
    print(f"Saved {outfile}")

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
        choices=["btagPNetB"],
    )
    args = parser.parse_args()

    main(args)