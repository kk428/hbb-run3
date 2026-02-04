### Going from skims to datacards

Set up fitting environment (only needs to be done once)
```
micromamba activate hbb
```

```
micromamba install root cms-combine -c conda-forge
```

Definition of categories and fit observable are in setup.json

Create signal region root pass / fail histograms from skims
```
python3 make_hists.py --year 2022EE --tag 25July21
```

Create your datacards and fit your QCD MC transfer factors
```
python3 make_datacards.py --year 2022EE --tag 25July21
```

