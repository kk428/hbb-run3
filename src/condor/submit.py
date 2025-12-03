"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

from __future__ import annotations

import argparse
import os
import warnings
from math import ceil
from pathlib import Path
from string import Template

from hbb import run_utils

t2_redirectors = {
    "lpc": "root://cmseos.fnal.gov//",
}


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with Path(templ_file).open() as f:
        templ = Template(f.read())

    with Path(out_file).open("w") as f:
        f.write(templ.substitute(templ_args))


def main(args):
    # check that branch exists
    run_utils.check_branch(args.git_branch, args.allow_diff_local_repo)

    if args.site == "lpc":
        try:
            proxy = os.environ["X509_USER_PROXY"]
        except:
            print("No valid proxy. Exiting.")
            exit(1)
    else:
        raise ValueError(f"Invalid site {args.site}")

    if args.site not in args.save_sites:
        warnings.warn(
            f"Your local site {args.site} is not in save sites {args.sites}!", stacklevel=1
        )

    t2_prefixes = [t2_redirectors[site] for site in args.save_sites]

    username = os.environ["USER"]

    tag = f"{args.tag}_{args.nano_version}"

    # make eos dir
    pdir = Path(f"store/user/lpchbbrun3/{username}/")
    outdir = pdir / tag

    # make local directory
    local_dir = Path(f"condor/{tag}")
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print("Condor work dir: ", local_dir)

    print(args.subsamples)
    fileset = run_utils.get_fileset(
        args.year,
        args.nano_version,
        args.samples,
        args.subsamples,
        get_num_files=True,
    )

    print(f"fileset: {fileset}")

    jdl_templ = "src/condor/submit.templ.jdl"
    sh_templ = "src/condor/submit.templ.sh"

    # submit jobs
    nsubmit = 0
    for sample in fileset:
        for subsample, tot_files in fileset[sample].items():
            if args.submit:
                print("Submitting " + subsample)

            sample_dir = outdir / args.year / subsample
            njobs = ceil(tot_files / args.files_per_job)

            for j in range(njobs):

                prefix = f"{args.year}_{subsample}"
                localcondor = f"{local_dir}/{prefix}_{j}.jdl"
                jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
                write_template(jdl_templ, localcondor, jdl_args)

                localsh = f"{local_dir}/{prefix}_{j}.sh"
                sh_args = {
                    "branch": args.git_branch,
                    "script": args.script,
                    "year": args.year,
                    "starti": j * args.files_per_job,
                    "endi": (j + 1) * args.files_per_job,
                    "sample": sample,
                    "subsample": subsample,
                    "t2_prefixes": " ".join(t2_prefixes),
                    "outdir": sample_dir,
                    "jobnum": j,
                    "nano_version": args.nano_version,
                    "run_mode": args.run_mode
                }
                write_template(sh_templ, localsh, sh_args)
                os.system(f"chmod u+x {localsh}")

                if Path(f"{localcondor}.log").exists():
                    Path(f"{localcondor}.log").unlink()

                if args.submit:
                    os.system(f"condor_submit {localcondor}")
                else:
                    print("To submit ", localcondor)
                nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


def parse_args(parser):
    parser.add_argument("--script", default="src/run.py", help="script to run", type=str)
    parser.add_argument(
        "--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str
    )
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc"],
    )
    parser.add_argument(
        "--save-sites",
        default=["lpc"],
        help="tier 2s in which we want to save the files",
        type=str,
        nargs="+",
        choices=["lpc"],
    )
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    run_utils.add_bool_arg(
        parser, "submit", default=False, help="submit files as well as create them"
    )
    parser.add_argument("--git-branch", required=True, help="git branch to use", type=str)
    run_utils.add_bool_arg(
        parser,
        "allow-diff-local-repo",
        default=False,
        help="Allow the local repo to be different from the specified remote repo (not recommended!)."
        "If false, submit script will exit if the latest commits locally and on Github are different.",
    )
    parser.add_argument(
        "--run-mode",
        default="save-skim",
        help="skimmer run mode",
        type=str,
        choices=["save-skim", "btag-eff", "save-skim-nosysts"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse_args(parser)
    run_utils.parse_common_args(parser)
    args = parser.parse_args()
    main(args)
