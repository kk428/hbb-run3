"""
Cross Sections for 13.6 TeV
"""

from __future__ import annotations

BR_WQQ = 0.676
BR_WLNU = 0.324
BR_ZQQ = 0.69911
BR_ZLNU = 0.27107
BR_ZLL = 0.02982

# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR at mH=125.40
BR_HBB = 5.760e-01
BR_HCC = 2.860e-02

xsecs = {}

# GJets: Obtained from XSDB
xsecs["GJ_PTG-20to100"] = 195300
xsecs["GJ_PTG-100to200"] = 1396
xsecs["GJ_PTG-200to400"] = 88.52
xsecs["GJ_PTG-400to600"] = 3.783
xsecs["GJ_PTG-600"] = 0.5755

# TTGamma: Obtained from XSDB
xsecs["TTG-1Jets_PTG-10to100"] = 4.216
xsecs["TTG-1Jets_PTG-100to200"] = 0.4114
xsecs["TTG-1Jets_PTG-200"] = 0.1284


# QCD
# QCD-HT (obtained by Cristina manually with genXsecAnalyzer)
xsecs["QCD_HT-40to70"] = 311600000.0
xsecs["QCD_HT-70to100"] = 58520000.0
xsecs["QCD_HT-100to200"] = 25220000.0
xsecs["QCD_HT-200to400"] = 1963000.0
xsecs["QCD_HT-400to600"] = 94870.0
xsecs["QCD_HT-600to800"] = 13420.0
xsecs["QCD_HT-800to1000"] = 2992.0
xsecs["QCD_HT-1000to1200"] = 879.1
xsecs["QCD_HT-1200to1500"] = 384.5
xsecs["QCD_HT-1500to2000"] = 125.5
xsecs["QCD_HT-2000"] = 25.78

# Top
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
# cross check these?
# https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=TOP-22-012
xsecs["TTto4Q"] = 923.6 * 0.667 * 0.667  # = 410.89  (762.1)
xsecs["TTto2L2Nu"] = 923.6 * 0.333 * 0.333  # = 102.41 (96.9)
xsecs["TTtoLNu2Q"] = 923.6 * 2 * (0.667 * 0.333)  # = 410.28 (404.0)

# Diboson
xsecs["WW"] = 116.8  #  173.4 (116.8 at NNLO)
xsecs["WZ"] = 54.3
xsecs["ZZ"] = 16.7

# Diboson extra
xsecs["ZZto2L2Q"] = 2.36
xsecs["ZZto2Nu2Q"] = 4.48
xsecs["ZZto4L"] = 0.170
xsecs["ZZto2L2Nu"] = 0.674
xsecs["WZtoLNu2Q"] = 12.368
xsecs["WZto2L2Q"] = 3.696
xsecs["WZto3LNu"] = 1.786
xsecs["WWto4Q"] = 78.79
xsecs["WWtoLNu2Q"] = 76.16
xsecs["WWto2L2Nu"] = 18.41
# not in XSDB
xsecs["WWto4Q_1Jets-4FS"] = 0
xsecs["WZto4Q-1Jets-4FS"] = 0

# SingleTop
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopNNLORef#Single_top_quark_tW_channel_cros
xsecs["TWminusto4Q"] = 87.9 / 2 * 0.667 * BR_WQQ
xsecs["TWminusto2L2Nu"] = 87.9 / 2 * 0.333 * BR_WLNU
xsecs["TWminustoLNu2Q"] = 87.9 / 2 * (0.667 * BR_WQQ + 0.333 * BR_WLNU)
xsecs["TbarWplusto4Q"] = 87.9 / 2 * 0.667 * BR_WQQ
xsecs["TbarWplustoLNu2Q"] = 87.9 / 2 * 0.333 * BR_WLNU
xsecs["TbarWplusto2L2Nu"] = 87.9 / 2 * (0.667 * BR_WQQ + 0.333 * BR_WLNU)
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopNNLORef#Single_top_t_channel
xsecs["TbarBQ_t-channel_4FS"] = 87.2
xsecs["TBbarQ_t-channel_4FS"] = 145.0

# Higgs
xsecs["GluGluHto2B_M-125"] = 51.53 * BR_HBB  # 29.93
xsecs["GluGluHto2C_M-125"] = 51.53 * BR_HCC  # 29.93
# SX: took XSDB NLO number (0.5246) and multiplied it by the NNLO/NLO ratio for inclusive ggH from 13 TeV
xsecs["GluGluHto2B_PT-200_M-125"] = 0.5246 * (43.92 / 27.8) * BR_HBB
xsecs["GluGluHto2C_PT-200_M-125"] = 0.5247 * (43.92 / 27.8) * BR_HCC
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap
xsecs["VBFHto2B_M-125_dipoleRecoilOn"] = 4.078 * BR_HBB  # 2.34
xsecs["VBFHto2C_M-125"] = 4.078 * BR_HCC  # 2.34
xsecs["WminusH_Hto2B_Wto2Q_M-125"] = 0.8889 * BR_WQQ * BR_HBB  # 0.349
xsecs["WminusH_Hto2C_Wto2Q_M-125"] = 0.8889 * BR_WQQ * BR_HCC
xsecs["WminusH_Hto2B_WtoLNu_M-125"] = 0.8889 * BR_WLNU * BR_HBB  # 0.167
xsecs["WminusH_Hto2C_WtoLNu_M-125"] = 0.8889 * BR_WLNU * BR_HCC
xsecs["WplusH_Hto2B_Wto2Q_M-125"] = 0.5677 * BR_WQQ * BR_HBB  # 0.222
xsecs["WplusH_Hto2C_Wto2Q_M-125"] = 0.5677 * BR_WQQ * BR_HCC
xsecs["WplusH_Hto2B_WtoLNu_M-125"] = 0.5677 * BR_WLNU * BR_HBB  # 0.106
xsecs["WplusH_Hto2C_WtoLNu_M-125"] = 0.5677 * BR_WLNU * BR_HBB
xsecs["ZH_Hto2B_Zto2L_M-125"] = 0.8079 * BR_ZLL * BR_HBB
xsecs["ZH_Hto2C_Zto2L_M-125"] = 0.8079 * BR_ZLL * BR_HCC
xsecs["ZH_Hto2B_Zto2Q_M-125"] = 0.8079 * BR_ZQQ * BR_HBB
xsecs["ZH_Hto2C_Zto2Q_M-125"] = 0.8079 * BR_ZQQ * BR_HCC
xsecs["ZH_Hto2C_Zto2Nu_M-125"] = 0.8079 * BR_ZLNU * BR_HCC
xsecs["ggZH_Hto2B_Zto2L_M-125"] = 0.1360 * BR_ZLL * BR_HBB
xsecs["ggZH_Hto2C_Zto2L_M-125"] = 0.1360 * BR_ZLL * BR_HCC
xsecs["ggZH_Hto2B_Zto2Nu_M-125"] = 0.1360 * BR_ZLNU * BR_HBB
xsecs["ggZH_Hto2C_Zto2Nu_M-125"] = 0.1360 * BR_ZLNU * BR_HCC
xsecs["ggZH_Hto2B_Zto2Q_M-125"] = 0.1360 * BR_ZQQ * BR_HBB
xsecs["ggZH_Hto2C_Zto2Q_M-125"] = 0.1360 * BR_ZQQ * BR_HCC
xsecs["ttHto2B_M-125"] = 0.5700 * BR_HBB
xsecs["ttHto2C_M-125"] = 0.5700 * BR_HCC

# V+Jets
xsecs["Wto2Q-3Jets_HT-200to400"] = 2723.0
xsecs["Wto2Q-3Jets_HT-400to600"] = 299.8
xsecs["Wto2Q-3Jets_HT-600to800"] = 63.9
xsecs["Wto2Q-3Jets_HT-800"] = 31.9
xsecs["Zto2Q-4Jets_HT-200to400"] = 1082.0
xsecs["Zto2Q-4Jets_HT-400to600"] = 124.1
xsecs["Zto2Q-4Jets_HT-600to800"] = 27.28
xsecs["Zto2Q-4Jets_HT-800"] = 14.57

#LO v15 binning - From Zichun running genZSecAnalyzer
xsecs["Wto2Q-3Jets_Bin-HT-100to400"] = 16120
xsecs["Wto2Q-3Jets_Bin-HT-400to800"] = 354.2
xsecs["Wto2Q-3Jets_Bin-HT-800to1500"] = 29.6
xsecs["Wto2Q-3Jets_Bin-HT-1500to2500"] = 1.852
xsecs["Wto2Q-3Jets_Bin-HT-2500"] = 0.1177
xsecs["Zto2Q-4Jets_Bin-HT-100to400"] = 6328
xsecs["Zto2Q-4Jets_Bin-HT-400to800"] = 145.1
xsecs["Zto2Q-4Jets_Bin-HT-800to1500"] = 12.93
xsecs["Zto2Q-4Jets_Bin-HT-1500to2500"] = 0.8496
xsecs["Zto2Q-4Jets_Bin-HT-2500"] = 0.05672

# These are bugged for 2022-2023 eras
xsecs["Wto2Q-2Jets_PTQQ-100to200_1J"] = 1517.0 / 2.0
xsecs["Wto2Q-2Jets_PTQQ-100to200_2J"] = 1757.0 / 2.0
xsecs["Wto2Q-2Jets_PTQQ-200to400_1J"] = 103.6 / 2.0
xsecs["Wto2Q-2Jets_PTQQ-200to400_2J"] = 227.1 / 2.0
xsecs["Wto2Q-2Jets_PTQQ-400to600_1J"] = 3.496 / 2.0
xsecs["Wto2Q-2Jets_PTQQ-400to600_2J"] = 12.75 / 2.0
xsecs["Wto2Q-2Jets_PTQQ-600_1J"] = 0.4221 / 2.0
xsecs["Wto2Q-2Jets_PTQQ-600_2J"] = 2.128 / 2.0

xsecs["Zto2Q-2Jets_PTQQ-100to200_1J"] = 302.0
xsecs["Zto2Q-2Jets_PTQQ-100to200_2J"] = 343.9
xsecs["Zto2Q-2Jets_PTQQ-200to400_1J"] = 21.64
xsecs["Zto2Q-2Jets_PTQQ-200to400_2J"] = 48.36
xsecs["Zto2Q-2Jets_PTQQ-400to600_1J"] = 0.7376
xsecs["Zto2Q-2Jets_PTQQ-400to600_2J"] = 2.683
xsecs["Zto2Q-2Jets_PTQQ-600_1J"] = 0.08717
xsecs["Zto2Q-2Jets_PTQQ-600_2J"] = 0.4459

xsecs["Zto2Nu-2Jets_PTNuNu-100to200_1J"] = 87.89
xsecs["Zto2Nu-2Jets_PTNuNu-100to200_2J"] = 101.4
xsecs["Zto2Nu-2Jets_PTNuNu-200to400_1J"] = 6.314
xsecs["Zto2Nu-2Jets_PTNuNu-200to400_2J"] = 13.81
xsecs["Zto2Nu-2Jets_PTNuNu-400to600_1J"] = 0.2154
xsecs["Zto2Nu-2Jets_PTNuNu-400to600_2J"] = 0.833
xsecs["Zto2Nu-2Jets_PTNuNu-600_1J"] = 0.02587
xsecs["Zto2Nu-2Jets_PTNuNu-600_2J"] = 0.1574

xsecs["Zto2Nu-4Jets_HT-100to200"] = 273.7
xsecs["Zto2Nu-4Jets_HT-200to400"] = 75.96
xsecs["Zto2Nu-4Jets_HT-400to800"] = 13.19
xsecs["Zto2Nu-4Jets_HT-800to1500"] = 1.364

#NLO v15 - From Zichun running genZSecAnalyzer
xsecs["Wto2Q-2Jets_Bin-PTQQ-100"] = 1751.0
xsecs["Wto2Q-2Jets_Bin-PTQQ-200"] = 164.3
xsecs["Wto2Q-2Jets_Bin-PTQQ-400"] = 9.205
xsecs["Wto2Q-2Jets_Bin-PTQQ-600"] = 2.23
xsecs["Zto2Q-2Jets_Bin-PTQQ-100"] = 695.0
xsecs["Zto2Q-2Jets_Bin-PTQQ-200"] = 71.56
xsecs["Zto2Q-2Jets_Bin-PTQQ-400"] = 3.811
xsecs["Zto2Q-2Jets_Bin-PTQQ-600"] = 0.5086

# NLO
xsecs["WtoLNu-2Jets"] = 64481.58
xsecs["WtoLNu-2Jets_0J"] = 55760.0
xsecs["WtoLNu-2Jets_1J"] = 9529.0
xsecs["WtoLNu-2Jets_2J"] = 3532.0

xsecs["DYto2L-2Jets_MLL-50-PTLL-40to100_1J"] = 475.3
xsecs["DYto2L-2Jets_MLL-50-PTLL-100to200_1J"] = 45.42
xsecs["DYto2L-2Jets_MLL-50-PTLL-200to400_1J"] = 3.382
xsecs["DYto2L-2Jets_MLL-50-PTLL-400to600_1J"] = 0.1162
xsecs["DYto2L-2Jets_MLL-50-PTLL-600_1J"] = 0.01392

xsecs["DYto2L-2Jets_MLL-50-PTLL-40to100_2J"] = 179.3
xsecs["DYto2L-2Jets_MLL-50-PTLL-100to200_2J"] = 51.68
xsecs["DYto2L-2Jets_MLL-50-PTLL-200to400_2J"] = 7.159
xsecs["DYto2L-2Jets_MLL-50-PTLL-400to600_2J"] = 0.4157
xsecs["DYto2L-2Jets_MLL-50-PTLL-600_2J"] = 0.07019

# LO
xsecs["WtoLNu-4Jets"] = 55390.0
xsecs["WtoLNu-4Jets_1J"] = 9625.0
xsecs["WtoLNu-4Jets_2J"] = 3161.0
xsecs["WtoLNu-4Jets_3J"] = 1468.0

# WtoLNu-4Jets 2024 samples
xsecs["WtoLNu-4Jets_Bin-1J"] = 9141
xsecs["WtoLNu-4Jets_Bin-2J"] = 2931
xsecs["WtoLNu-4Jets_Bin-3J"] = 864.6
xsecs["WtoLNu-4Jets_Bin-4J"] = 417.8
# WtoLNu-2Jets 2024 samples
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-40to100"] = 4211
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-100to200"] = 342.3
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-200to400"] = 21.84
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-400to600"] = 0.6845
xsecs["WtoLNu-2Jets_Bin-1J-PTLNu-600"] = 0.07753
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-40to100"] = 1581
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-100to200"] = 411.1
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-200to400"] = 53.59
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-400to600"] = 3.099
xsecs["WtoLNu-2Jets_Bin-2J-PTLNu-600"] = 0.5259

xsecs["DYto2L-4Jets_MLL-50"] = 5467.0
xsecs["DYto2L-4Jets_MLL-10to50"] = 0
xsecs["DYto2L-4Jets_MLL-50_1J"] = 0
xsecs["DYto2L-4Jets_MLL-50_2J"] = 0
xsecs["DYto2L-4Jets_MLL-50_3J"] = 0
xsecs["DYto2L-4Jets_MLL-50_4J"] = 0

xsecs["DYto2L-2Jets_MLL-10to50"] = 0
xsecs["DYto2L-2Jets_MLL-50"] = 6688.0
xsecs["DYto2L-2Jets_MLL-50_0J"] = 5378.0
xsecs["DYto2L-2Jets_MLL-50_1J"] = 1017.0
xsecs["DYto2L-2Jets_MLL-50_2J"] = 385.5

# V + gamma
xsecs["WGtoLNuG-1Jets_PTG-10to100"] = 662.2
xsecs["WGtoLNuG-1Jets_PTG-100to200"] = 2.221
xsecs["WGtoLNuG-1Jets_PTG-200to400"] = 0.2908
xsecs["WGtoLNuG-1Jets_PTG-400to600"] = 0.02231
xsecs["WGtoLNuG-1Jets_PTG-600"] = 0.004907

xsecs["WGto2QG-1Jets_PTG-100to200"] = 3.993
xsecs["WGto2QG-1Jets_PTG-200"] = 0.6326

xsecs["ZGto2NuG-1Jets_PTG-10to100"] = 39.93
xsecs["ZGto2NuG-1Jets_PTG-100to200"] = 0.5652
xsecs["ZGto2NuG-1Jets_PTG-200to400"] = 0.07535
xsecs["ZGto2NuG-1Jets_PTG-400to600"] = 0.005547
xsecs["ZGto2NuG-1Jets_PTG-600"] = 0.001177
xsecs["ZGto2QG-1Jets_PTG-100to200"] = 1.949
xsecs["ZGto2QG-1Jets_PTG-200"] = 0.282

# EWK V
xsecs["VBFWto2Q"] = 95.3
xsecs["VBFWtoLNu"] = 41.02
xsecs["VBFZto2Q"] = 13.67
xsecs["VBFZto2L"] = 7.659
xsecs["VBFZto2Nu"] = 4.12
