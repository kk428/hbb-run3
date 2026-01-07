from __future__ import annotations

year_map = {
    "2022": ["2022"],
    "2022EE": ["2022EE"],
    "2023": ["2023"],
    "2023BPix": ["2023BPix"],
    "2022-2023": ["2022", "2022EE", "2023", "2023BPix"],
    "2024": ["2024"],
}

common_mc = {
    "ggf-hbb": {"GluGluHto2B_PT-200_M-125"},
    "vh-hbb": {
        "WplusH_Hto2B_Wto2Q_M-125",
        "WminusH_Hto2B_Wto2Q_M-125",
        "ggZH_Hto2B_Zto2Q_M-125",
        "ZH_Hto2B_Zto2Q_M-125",
    },
    "vbf-hbb": {
        "VBFHto2B_M-125_dipoleRecoilOn",
    },
    "qcd": {
        "QCD_HT-200to400",
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",
    },
    "tt": {"TTto2L2Nu", "TTto4Q", "TTtoLNu2Q"},
    "singletop": {
        "TBbarQ_t-channel_4FS",
        "TbarBQ_t-channel_4FS",
        "TWminusto2L2Nu",
        "TWminusto4Q",
        "TWminustoLNu2Q",
        "TbarWplusto2L2Nu",
        "TbarWplusto4Q",
        "TbarWplustoLNu2Q",
    },
    "diboson": {
        "WW",
        "WZ",
        "ZZ",
    },
    "wjets": {
        "Wto2Q-3Jets_HT-200to400",
        "Wto2Q-3Jets_HT-400to600",
        "Wto2Q-3Jets_HT-600to800",
        "Wto2Q-3Jets_HT-800",
    },
    "zjets": {
        "Zto2Q-4Jets_HT-200to400",
        "Zto2Q-4Jets_HT-400to600",
        "Zto2Q-4Jets_HT-600to800",
        "Zto2Q-4Jets_HT-800",
    },
    "ewkv": {
        "VBFWtoLNu",
        "VBFWto2Q",
        "VBFZto2Q",
        "VBFZto2L",
        "VBFZto2Nu",
    },
}

data_by_year = {
    "2022": {
        "JetMET_Run2022C_single",
        "JetMET_Run2022C",
        "JetMET_Run2022D",
    },
    "2022EE": {
        "JetMET_Run2022E",
        "JetMET_Run2022F",
        "JetMET_Run2022G",
    },
    "2023": {
        "JetMET_Run2023C",
    },
    "2023BPix": {
        "JetMET_Run2023D",
    },
}
