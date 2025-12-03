from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def parse_common_args(parser):
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument(
        "--samples",
        default=[],
        help="which samples to run",  # , default will be all samples",
        nargs="*",
    )
    parser.add_argument(
        "--subsamples",
        default=[],
        help="which subsamples, by default will be all in the specified sample(s)",
        nargs="*",
    )
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--nano-version",
        type=str,
        default="v12",
        choices=[
            "v12",
            "v12v2_private",
            "v14_private"
        ],
        help="NanoAOD version",
    )


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


def check_branch(git_branch: str, allow_diff_local_repo: bool = False):
    """Check that specified git branch exists in the repo, and local repo is up-to-date"""
    assert not bool(
        os.system(
            f'git ls-remote --exit-code --heads "https://github.com/DAZSLE/hbb-run3" "{git_branch}"'
        )
    ), f"Branch {git_branch} does not exist"

    print(f"Using branch {git_branch}")

    # check if there are uncommitted changes
    uncommited_files = int(subprocess.getoutput("git status -s | wc -l"))

    if uncommited_files:
        print("There are local changes that have not been committed!")
        os.system("git status -s")
        if allow_diff_local_repo:
            print("Proceeding anyway...")
        else:
            print("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)

    # check that the local repo's latest commit matches that on github
    remote_hash = subprocess.getoutput(f"git show origin/{git_branch} | head -n 1").split(" ")[1]
    local_hash = subprocess.getoutput("git rev-parse HEAD")

    if remote_hash != local_hash:
        print("Latest local and github commits do not match!")
        print(f"Local commit hash: {local_hash}")
        print(f"Remote commit hash: {remote_hash}")
        if allow_diff_local_repo:
            print("Proceeding anyway...")
        else:
            print("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)


def get_fileset(
    year: int,
    version: str,
    samples: list,
    subsamples: list,
    starti: int = 0,
    endi: int = -1,
    get_num_files: bool = False,
    check_subsamples=True,  # to check that all subsamples will be processed
):
    """
    Get the fileset for a given year and version of the nanoAOD files.
    Fileset is a dictionary of dictionaries, with the following structure:
    {
        "year_subsample1": [
            "file1.root",
            "file2.root",
            ...
        ],
        "year_subsample2": [
            "file1.root",
            "file2.root",
            ...
        ],
    }

    """
    with Path(f"data/nanoindex_{version}.json").open() as f:
        full_fileset_nano = json.load(f)

    fileset = {}

    for sample in samples:
        sample_set = full_fileset_nano[year][sample]
        set_subsamples = list(sample_set.keys())

        # check if any subsamples for this sample have been specified
        get_subsamples = set(set_subsamples).intersection(subsamples)

        # identify which subsamples are not in the full set
        if check_subsamples and len(subsamples):
            for subs in subsamples:
                if subs not in get_subsamples:
                    raise ValueError(f"Subsample {subs} not found for sample {sample}!")

        # if the intersection is nonzero, keep only that subset
        if len(get_subsamples):
            sample_set = {subsample: sample_set[subsample] for subsample in get_subsamples}

        if get_num_files:
            # return only the number of files per subsample (for splitting up jobs)
            fileset[sample] = {}
            for subsample, fnames in sample_set.items():
                fileset[sample][subsample] = len(fnames)

        else:
            # return all files per subsample
            sample_fileset = {}

            for subsample, fnames in sample_set.items():
                run_fnames = fnames[starti:] if endi < 0 else fnames[starti:endi]
                sample_fileset[subsample] = run_fnames

            fileset = {**fileset, **sample_fileset}

    return fileset


def get_dataset_spec(
    fileset: dict,
):
    """
    Get the dataset specification for a given fileset using Coffea
    :param fileset: dict
        Dictionary of fileset with the following structure:
        {
            "year_subsample1": [
                "file1.root",
                "file2.root",
                ...
            ],
        }
    :return: dict
        Dictionary of dataset specification with the following structure:
        {
            "year_subsample1": {
                "files": {
                    "file1.root": "Events",
                    "file2.root": "Events",
                    ...
                },
                "metadata": {
                    "dataset": "year_subsample1",
                },
            },
        }
    """
    dict_process_files = {}
    for dataset, files in fileset.items():
        dict_process_files[dataset] = {
            "files": dict.fromkeys(files, "Events"),
            "metadata": {
                "dataset": dataset,
            },
        }
    return dict_process_files
