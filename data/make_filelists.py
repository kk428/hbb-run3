from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"


def eos_rec_search(startdir, suffix, dirs):
    print(f"EOS Recursive search in {startdir}.")
    eosbase = "root://cmseos.fnal.gov/"
    try:
        dirlook = (
            subprocess.check_output(f"eos {eosbase} ls {startdir}", shell=True)
            .decode("utf-8")
            .split("\n")[:-1]
        )
    except:
        print(f"No files found for {startdir}")
        return dirs

    donedirs = [[] for d in dirlook]
    for di, d in enumerate(dirlook):
        # print(f"Looking in {dirlook}.")
        if d.endswith(suffix):
            donedirs[di].append(startdir + "/" + d)
        elif d == "log":
            continue
        else:
            donedirs[di] = donedirs[di] + eos_rec_search(
                startdir + "/" + d, suffix, dirs + donedirs[di]
            )
    donedir = [d for da in donedirs for d in da]
    return dirs + donedir


def get_files(dataset, version):
    if "private" in version:
        files = eos_rec_search(dataset, ".root", [])
        return [f"root://cmseos.fnal.gov/{f}" for f in files]
    else:
        import requests
        from rucio_utils import get_dataset_files, get_proxy_path

        proxy = get_proxy_path()
        if "USER" in dataset:
            link = f"https://cmsweb.cern.ch:8443/dbs/prod/phys03/DBSReader/files?dataset={dataset}&detail=True"
        else:
            link = f"https://cmsweb.cern.ch:8443/dbs/prod/global/DBSReader/files?dataset={dataset}&detail=True"
        r = requests.get(
            link,
            cert=proxy,
            verify=False,
        )
        filesjson = r.json()
        files = []
        not_valid = []
        for fj in filesjson:
            if "is_file_valid" in fj:
                if fj["is_file_valid"] == 0:
                    # print(f"ERROR: File not valid on DAS: {fj['logical_file_name']}")
                    not_valid.append(fj["logical_file_name"])
                else:
                    files.append(fj["logical_file_name"])
            else:
                continue

        if "USER" in dataset:
            files_valid = [f"root://cmseos.fnal.gov/{f}" for f in files]
            return files_valid

        if len(files) == 0:
            print(f"Found 0 files for sample {dataset}!")
            return []

        # Now query rucio to get the concrete dataset passing the sites filtering options
        sites_cfg = {
            "whitelist_sites": [],
            "blacklist_sites": [
                "T2_FR_IPHC" "T2_US_MIT",
                "T2_US_Vanderbilt",
                "T2_UK_London_Brunel",
                "T2_UK_SGrid_RALPP",
                "T1_UK_RAL_Disk",
                "T2_PT_NCG_Lisbon",
            ],
            "regex_sites": None,
        }
        if not "private" in version:
            sites_cfg["whitelist_sites"] = ["T1_US_FNAL_Disk", "T3_US_FNALLPC"]

        files_rucio, sites = get_dataset_files(dataset, **sites_cfg, output="first")

        # print(dataset, sites)

        # Get rid of invalid files
        files_valid = []
        for f in files_rucio:
            invalid = False
            for nf in not_valid:
                if nf in f:
                    invalid = True
                    break
            if not invalid:
                files_valid.append(f)

        return files_valid


def main():
    version = sys.argv[1]
    allowed_values = ["v12", "v12v2_private", "v14_private", "v15"]

    try:
        version = sys.argv[1]
        if version not in allowed_values:
            raise ValueError(f"Invalid value '{version}'. Must be one of: {allowed_values}")
        print(f"Argument '{version}' is valid.")
    except IndexError:
        print("No argument provided. Please specify one of:", allowed_values)
    except ValueError as e:
        print(e)

    mod = importlib.import_module(version)

    datasets = mod.get_datasets()
    index = datasets.copy()
    for year, ydict in datasets.items():
        print(year)
        for sample, sdict in ydict.items():
            print(sample)
            for sname, dataset in sdict.items():
                if isinstance(dataset, list):
                    files = []
                    for d in dataset:
                        files.extend(get_files(d, version))
                    index[year][sample][sname] = files
                else:
                    index[year][sample][sname] = get_files(dataset, version)

    with Path(f"nanoindex_{version}.json").open("w") as f:
        json.dump(index, f, indent=4)


if __name__ == "__main__":
    main()
