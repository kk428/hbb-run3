import pickle
import hist
import numpy as np

fn = "/eos/uscms/store/user/kkrzyzan/25Sep23_main_v12/2023/JetMET_Run2023Cv1/pickles/out_0.pkl"
proc = "JetMET_Run2023Cv1"
region = "signal-all"

with open(fn, "rb") as f:
    d = pickle.load(f)

cf = d[proc]["cutflow"]
if hasattr(cf, "compute"):
    cf = cf.compute()

# re-wrap so axes support name lookup
if not isinstance(cf, hist.Hist):
    cf = hist.Hist(cf)

print([ax.name for ax in cf.axes])
print("regions:", list(cf.axes["region"]))
print("datasets:", list(cf.axes["dataset"]))

h1 = cf[{"region": region, "dataset": sum, "genflavor": sum}]
vals = h1.values()
errs = np.sqrt(h1.variances())
for i, (v, e) in enumerate(zip(vals, errs)):
    print(f"cut {i:2d}: {v:12.2f} +/- {e:.2f}")