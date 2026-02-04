from __future__ import annotations

qcd_ht_bins = [
    "100to200",
    "200to400",
    "400to600",
    "600to800",
    "800to1000",
    "1000to1200",
    "1200to1500",
    "1500to2000",
    "2000",
]

vgamma_pt_bins = ["10to100", "100to200", "200to400", "400to600", "600"]


# OFFICIAL NANOAODv15
def get_datasets():
    return {
        "2024": {
            "JetMET": {
                "JetMET_Run2024C": [
                    "/JetMET0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                    "/JetMET1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                ],
                "JetMET_Run2024D": [
                    "/JetMET0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                    "/JetMET1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                ],
                "JetMET_Run2024E": [
                    "/JetMET0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                    "/JetMET1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                ],
                "JetMET_Run2024F": [
                    "/JetMET0/Run2024F-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024F-MINIv6NANOv15-v2/NANOAOD",
                ],
                "JetMET_Run2024G": [
                    "/JetMET0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                ],
                "JetMET_Run2024H": [
                    "/JetMET0/Run2024H-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024H-MINIv6NANOv15-v2/NANOAOD",
                ],
                "JetMET_Run2024I": [
                    "/JetMET0/Run2024I-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024I-MINIv6NANOv15-v1/NANOAOD",
                ]
            },
            "Muon": {
                "Muon_Run2024C": [
                    "/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                ],
                "Muon_Run2024D": [
                    "/Muon0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                ],
                "Muon_Run2024E": [
                    "/Muon0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                ],
                "Muon_Run2024F": [
                    "/Muon0/Run2024F-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024F-MINIv6NANOv15-v1/NANOAOD",
                ],
                "Muon_Run2024G": [
                    "/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                ],
                "Muon_Run2024H": [
                    "/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                ],
                "Muon_Run2024I": [
                    "/Muon0/Run2024I-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024I-MINIv6NANOv15-v1/NANOAOD",
                ]
            },
            "EGamma": {
                "EGamma_Run2024C": [
                    "/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                    "/EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                ],
                "EGamma_Run2024D": [
                    "/EGamma0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                    "/EGamma1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                ],
                "EGamma_Run2024E": [
                    "/EGamma0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                    "/EGamma1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                ],
                "EGamma_Run2024F": [
                    "/EGamma0/Run2024F-MINIv6NANOv15-v1/NANOAOD",
                    "/EGamma1/Run2024F-MINIv6NANOv15-v1/NANOAOD",
                ],
                "EGamma_Run2024G": [
                    "/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                    "/EGamma1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                ],
                "EGamma_Run2024H": [
                    "/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                    "/EGamma1/Run2024G-MINIv6NANOv15-v1/NANOAOD",
                ],
                "EGamma_Run2024I": [
                    "/EGamma0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
                    "/EGamma1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
                ]
            },
            # "TTGamma": { #TODO - not produced
            #     "TTG-1Jets_PTG-10to100": "",
            #     "TTG-1Jets_PTG-100to200": "",
            #     "TTG-1Jets_PTG-200": "",
            # },
            "GJets": { 
                "GJ_PTG-100to200": "/GJ_Bin-PTG-100to200_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "GJ_PTG-200to400": "/GJ_Bin-PTG-200to400_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "GJ_PTG-400to600": "/GJ_Bin-PTG-400to600_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "GJ_PTG-600": "/GJ_Bin-PTG-600_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "Hbb": {
                "GluGluHto2B_M-125": [
                    "/GluGluH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                    "/GluGluH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2_ext1-v2/NANOAODSIM"
                ],
                "GluGluHto2B_PT-200_M-125": "/GluGluH-Hto2B_Bin-PT-200_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "VBFHto2B_M-125": [
                    "/VBFH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                    "/VBFH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2_ext1-v2/NANOAODSIM"
                ],
                # "VBFHto2B_M-125_dipoleRecoilOn": "", #TODO - not produced
                "WminusH_Hto2B_Wto2Q_M-125": "/WminusH-Wto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WminusH_Hto2B_WtoLNu_M-125": "/WminusH-WtoLNu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WplusH_Hto2B_Wto2Q_M-125": "/WplusH-Wto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WplusH_Hto2B_WtoLNu_M-125": "/WplusH-WtoLNu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2L_M-125": "/ZH-Zto2L-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2Nu_M-125": "/ZH-Zto2Nu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZH_Hto2B_Zto2Q_M-125": "/ZH-Zto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2L_M-125": "/GluGluZH-Zto2L-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2Nu_M-125": "/GluGluZH-Zto2Nu-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ggZH_Hto2B_Zto2Q_M-125": "/GluGluZH-Zto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ttHto2B_M-125": "/TTH-Hto2B_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "Hcc": {
                "GluGluHto2C_M-125": [
                    "/GluGluH-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                    "/GluGluH-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2_ext1-v2/NANOAODSIM"
                ],
                "GluGluHto2C_PT-200_M-125": "/GluGluH-Hto2C_Bin-PT-200_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "VBFHto2C_M-125": [
                    "/VBFH-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                    "/VBFH-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2_ext1-v2/NANOAODSIM"
                ],
                "WminusH_Hto2C_Wto2Q_M-125": "/WminusH-Wto2Q-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WminusH_Hto2C_WtoLNu_M-125": "/WminusH-WtoLNu-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WplusH_Hto2C_Wto2Q_M-125": "/WplusH-Wto2Q-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WplusH_Hto2C_WtoLNu_M-125": "/WplusH-WtoLNu-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZH_Hto2C_Zto2L_M-125": "/ZH-Zto2L-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZH_Hto2C_Zto2Nu_M-125": "/ZH-Zto2Nu-Hto2C_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # "ZH_Hto2C_Zto2Q_M-125": "", #TODO - not produced
                "ggZH_Hto2C_Zto2L_M-125": "/GluGluZH-Zto2L-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ggZH_Hto2C_Zto2Nu_M-125": "/GluGluZH-Zto2Q-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ggZH_Hto2C_Zto2Q_M-125": "/GluGluZH-Zto2Nu-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ttHto2C_M-125": "/TTH-Hto2C_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "QCD": { #TODO check 100-200
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_Bin-HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
            },
            "TT": {
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "SingleTop": {
                # "TBbarQ_t-channel_4FS": "", #TODO - not produced
                # "TbarBQ_t-channel_4FS": "",
                "TWminusto4Q": "/TWminusto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TWminustoLNu2Q": "/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TbarBQ_t-channel_4FS": "/TbarBQto2Q-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TbarWplusto4Q": "/TbarWplusto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TbarWplustoLNu2Q": "/TbarWplustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",


                "TbarWplusto2L2Nu": "/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TWminusto2L2Nu": "/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "Diboson_extra": { 
                "WWto4Q": "/WWto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # "WWto4Q_1Jets-4FS": "", #TODO - not produced
                "WZto4Q-1Jets-4FS": "/WZto4Q-1Jets-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZZto4Q_1Jets-4FS": "/ZZto4Q-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM", 
                # missing ZZto4Q
            },
            "VJets_had_NLO": { # Not split by 1J v 2J - new xs
                "Wto2Q-2Jets_Bin-PTQQ-100": "/Wto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "Wto2Q-2Jets_Bin-PTQQ-200": "/Wto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-2Jets_Bin-PTQQ-400": "/Wto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "Wto2Q-2Jets_Bin-PTQQ-600": "/Wto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-2Jets_Bin-PTQQ-100": "/Zto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-2Jets_Bin-PTQQ-200": "/Zto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-2Jets_Bin-PTQQ-400": "/Zto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-2Jets_Bin-PTQQ-600": "/Zto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "VJets_lep_NLO": {
                # WJetsToLNu #TODO - not produced
                # # "WtoLNu-2Jets": "",
                # "WtoLNu-2Jets_0J": "",
                # "WtoLNu-2Jets_1J": "",
                # "WtoLNu-2Jets_2J": "",
                # # DYToLL
                # "DYto2L-2Jets_MLL-10to50": "",
                # # "DYto2L-2Jets_MLL-50": [
                # #    "",
                # #    "",
                # # ],
                # "DYto2L-2Jets_MLL-50_0J": "",
                # "DYto2L-2Jets_MLL-50_1J": "",
                # "DYto2L-2Jets_MLL-50_2J": "",
                **{
                    f"DYto2L-2Jets_MLL-50_PTLL-{qbin}_1J": f"/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-{qbin}_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
                    for qbin in ["100to200", "200to400", "400to600", "600"]
                },
                **{
                    f"DYto2L-2Jets_MLL-50_PTLL-{qbin}_2J": f"/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-{qbin}_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
                    for qbin in ["100to200", "200to400", "400to600", "600"]
                },
            },
            "VJets_had_LO": {
                # Wto2Q - New Binning
                "Wto2Q-3Jets_Bin-HT-100to400": "/Wto2Q-3Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-400to800": "/Wto2Q-3Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-800to1500": "/Wto2Q-3Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-1500to2500": "/Wto2Q-3Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-2500": "/Wto2Q-3Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # Zto2Q - New Binning
                "Zto2Q-4Jets_Bin-HT-100to400": "/Zto2Q-4Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-400to800": "/Zto2Q-4Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-800to1500": "/Zto2Q-4Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-1500to2500": "/Zto2Q-4Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-2500": "/Zto2Q-4Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            },
            "VJets_lep_LO": { 
                # WJetsToLNu
                # WtoLNu-4Jets
                "WtoLNu-4Jets_Bin-1J": "/WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-4Jets_Bin-2J": "/WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-4Jets_Bin-3J": "/WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-4Jets_Bin-4J": "/WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # WtoLNu-2Jets
                "WtoLNu-2Jets_Bin-1J-PTLNu-40to100": "/WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-100to200": "/WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-200to400": "/WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-400to600": "/WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-600": "/WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-40to100": "/WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-100to200": "/WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-200to400": "/WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-400to600": "/WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-600": "/WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # DYToLL # TODO - not produced
                # "DYto2L-4Jets_MLL-10to50": "",
                # "DYto2L-4Jets_MLL-50": "",
                # **{
                #     f"DYto2L-4Jets_MLL-120_HT-{qbin}": f""
                #     for qbin in ["100to400","400to800","800to1500","1500to2500","2500"]
                # },
                # "DYto2L-4Jets_MLL-50_1J": "",
                # "DYto2L-4Jets_MLL-50_2J": "",
                # "DYto2L-4Jets_MLL-50_3J": "",
                # "DYto2L-4Jets_MLL-50_4J": "",
                # **{
                #    f"DYto2L-4Jets_MLL-50_PTLL-{qbin}": f""
                #    for qbin in ["100to200", "200to400", "400to600", "600"]
                # },
            },
            # "VGamma": { #TODO - not produced
            #     "WGtoLNuG-1Jets_PTG-100to200": "",
            #     "WGtoLNuG-1Jets_PTG-200to400": "",
            #     "WGtoLNuG-1Jets_PTG-400to600": "",
            #     "WGtoLNuG-1Jets_PTG-600": "",
            #     "ZGto2NuG-1Jets_PTG-400to600": "",
            #     **{
            #         f"ZGto2NuG-1Jets_PTG-{qbin}": f""
            #         for qbin in ["100to200", "200to400", "600"]
            #     },
            #     **{
            #         f"ZGto2QG-1Jets_PTG-{qbin}": f""
            #         for qbin in ["100to200", "200"]  # there is also a 10 bin
            #     },
            #     **{
            #         f"WGto2QG-1Jets_PTG{qbin}": f""
            #         for qbin in ["100to200", "200"]
            #     },
            # },
            "EWKV": {
                "VBFZto2Q": "/VBFZto2Q_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # "VBFZto2L": "", #TODO - not produced
                # "VBFZto2Nu": "",
                "VBFWto2Q": "/VBFWto2Q_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # "VBFWtoLNu": "",
            },
        },
        "2025" : {
            "JetMET": {
                "JetMET_Run2025C": [
                    "/JetMET0/Run2025C-PromptReco-v2/NANOAOD",
                    "/JetMET1/Run2025C-PromptReco-v2/NANOAOD",
                ],
                "JetMET_Run2025D": [
                    "/JetMET0/Run2025D-PromptReco-v1/NANOAOD",
                    "/JetMET1/Run2025D-PromptReco-v1/NANOAOD",
                ],
                "JetMET_Run2025E": [
                    "/JetMET0/Run2025E-PromptReco-v1/NANOAOD",
                    "/JetMET1/Run2025E-PromptReco-v1/NANOAOD",
                ],
                "JetMET_Run2025F": [
                    "/JetMET0/Run2025F-PromptReco-v2/NANOAOD",
                    "/JetMET1/Run2025F-PromptReco-v2/NANOAOD",
                ],
                "JetMET_Run2025G": [
                    "/JetMET0/Run2025G-PromptReco-v1/NANOAOD",
                    "/JetMET1/Run2025G-PromptReco-v1/NANOAOD",
                ],
            },
            "EGamma": {
                "EGamma_Run2025C": [
                    "/EGamma0/Run2025C-PromptReco-v2/NANOAOD",
                    "/EGamma1/Run2025C-PromptReco-v2/NANOAOD",
                    "/EGamma2/Run2025C-PromptReco-v2/NANOAOD",
                    "/EGamma3/Run2025C-PromptReco-v2/NANOAOD",
                ],
                "EGamma_Run2025D": [
                    "/EGamma0/Run2025D-PromptReco-v1/NANOAOD",
                    "/EGamma1/Run2025D-PromptReco-v1/NANOAOD",
                    "/EGamma2/Run2025D-PromptReco-v1/NANOAOD",
                    "/EGamma3/Run2025D-PromptReco-v1/NANOAOD",
                ],
                "EGamma_Run2025E": [
                    "/EGamma0/Run2025E-PromptReco-v1/NANOAOD",
                    "/EGamma1/Run2025E-PromptReco-v1/NANOAOD",
                    "/EGamma2/Run2025E-PromptReco-v1/NANOAOD",
                    "/EGamma3/Run2025E-PromptReco-v1/NANOAOD",
                ],
                "EGamma_Run2025F": [
                    "/EGamma0/Run2025F-PromptReco-v2/NANOAOD",
                    "/EGamma1/Run2025F-PromptReco-v2/NANOAOD",
                    "/EGamma2/Run2025F-PromptReco-v2/NANOAOD",
                    "/EGamma3/Run2025F-PromptReco-v2/NANOAOD",
                ],
                "EGamma_Run2025G": [
                    "/EGamma0/Run2025G-PromptReco-v1/NANOAOD",
                    "/EGamma1/Run2025G-PromptReco-v1/NANOAOD",
                    "/EGamma2/Run2025G-PromptReco-v1/NANOAOD",
                    "/EGamma3/Run2025G-PromptReco-v1/NANOAOD",
                ],
            },
            "Muon": {
                "Muon_Run2025C": [
                    "/Muon0/Run2025C-PromptReco-v2/NANOAOD",
                    "/Muon1/Run2025C-PromptReco-v2/NANOAOD",
                ],
                "Muon_Run2025D": [
                    "/Muon0/Run2025D-PromptReco-v1/NANOAOD",
                    "/Muon1/Run2025D-PromptReco-v1/NANOAOD",
                ],
                "Muon_Run2025E": [
                    "/Muon0/Run2025E-PromptReco-v1/NANOAOD",
                    "/Muon1/Run2025E-PromptReco-v1/NANOAOD",
                ],
                "Muon_Run2025F": [
                    "/Muon0/Run2025F-PromptReco-v2/NANOAOD",
                    "/Muon1/Run2025F-PromptReco-v2/NANOAOD",
                ],
                "Muon_Run2025G": [
                    "/Muon0/Run2025G-PromptReco-v1/NANOAOD",
                    "/Muon1/Run2025G-PromptReco-v1/NANOAOD",
                ],
            },
            
        }
    }
