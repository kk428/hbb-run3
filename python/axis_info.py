import hist


#axis labels associated columns
axis_to_column = {
    "pt1": "HiggsAK8_pt",
    "msd1": "HiggsAK8_msdmatched",
    "pt2": "VAK8_pt",
    "msd2": "VAK8_msd",
    "mjj": "VBFPair_mjj",
    "deta": "VBFPair_deta",
    "met": "MET",
    "pt3": "LeadingLep_pt",
    "pt4": "SubLeadingLep_pt",
    "lepmass": "LepPair_mass",
    "lepmass2": "LepPair_mass",

}

# Define the histogram axes
axis_to_histaxis = {
    "pt1": hist.axis.Regular(25, 250, 1000, name="pt1", label=r"Higgs AK8 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Regular(25, 250, 1000, name="pt2", label=r"V AK8 $p_{T}$ [GeV]"),
    "pt3": hist.axis.Regular(25, 0, 1000, name="pt3", label=r"Leading Lepton $p_{T}$ [GeV]"),
    "pt4": hist.axis.Regular(25, 0, 600, name="pt4", label=r"Subleading Lepton $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 40, 201, name="msd1", label="Higgs AK8 $m_{sd}$ [GeV]"),
    "msd2": hist.axis.Regular(23, 40, 201, name="msd2", label="V AK8 $m_{sd}$ [GeV]"),
    "mass1": hist.axis.Regular(30, 0, 200, name="mass1", label="Higgs AK8 PNet mass [GeV]"),
    "category": hist.axis.StrCategory([], name="category", label="Category", growth=True),
    "genflavor": hist.axis.IntCategory([0, 1, 2, 3], name="genflavor", label="Gen Flavor"),
    "mjj": hist.axis.Regular(25, 0, 4000 , name="mjj", label="$m_{jj}$ [GeV]"),
    "deta": hist.axis.Regular(20, 2, 10 , name="deta", label="$d\eta_{jj}$ [GeV]"),
    "met": hist.axis.Regular(25, 0, 1000, name="met", label=r"Puppi MET [GeV]"),
    "lepmass": hist.axis.Regular(25, 0, 1000, name="lepmass", label="$m_{ll}$ [GeV]"),
    "lepmass2": hist.axis.Regular(25, 80, 100, name="lepmass2", label="$m_{ll}$ [GeV]"),
}