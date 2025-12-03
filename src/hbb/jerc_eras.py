run_map = {
    ("Run2022C", "Run2022D"): "Run2022CD",
    ("Run2022E",): "Run2022E",
    ("Run2022F",): "Run2022F",
    ("Run2022G",): "Run2022G",
    ("Run2023Cv1", "Run2023Cv2", "Run2023Cv3"): "Run2023Cv123",
    ("Run2023Cv4",): "Run2023Cv4",
    ("Run2024C",): "Run2024C",
    ("Run2024D",): "Run2024D",
    ("Run2024E",): "Run2024E",
    ("Run2024F",): "Run2024F",
    ("Run2024G",): "Run2024G",
    ("Run2024H",): "Run2024H",
    ("Run2024I",): "Run2024I",
}

fatjet_jerc_keys = {
    "2022" : "AK8PFPuppi",
    "2022EE" :"AK8PFPuppi",
    "2023" : "AK8PFPuppi",
    "2023BPix" : "AK8PFPuppi",
    "2024" : "AK4PFPuppi", 
}

jet_jerc_keys = {
    "2022" : "AK4PFPuppi",
    "2022EE" :"AK4PFPuppi",
    "2023" : "AK4PFPuppi",
    "2023BPix" : "AK4PFPuppi",
    "2024" : "AK4PFPuppi"
}

jec_mc = {
    "L2Relative" : ".txt.gz",
    "UncertaintySources": ".junc.txt.gz",
    "Uncertainty" : ".junc.txt.gz"
}

jec_data = {
    "L2Relative" : ".txt.gz",
    "L2L3Residual" : ".txt.gz"
}

jer_mc = {
    "PtResolution" : ".jr.txt.gz",
    "SF" : ".jersf.txt.gz"
}

jec_eras = {
    "2022_mc": "Summer22_22Sep2023_V3_MC",
    "2022EE_mc": "Summer22EE_22Sep2023_V3_MC",
    "2023_mc": "Summer23Prompt23_V3_MC",
    "2023BPix_mc": "Summer23BPixPrompt23_V3_MC",
    "2024_mc": "Summer24Prompt24_V1_MC",
    "2022_Run2022CD": "Summer22_22Sep2023_RunCD_V3_DATA",
    "2022EE_Run2022E": "Summer22EE_22Sep2023_RunE_V3_DATA",
    "2022EE_Run2022F": "Summer22EE_22Sep2023_RunF_V3_DATA",
    "2022EE_Run2022G": "Summer22EE_22Sep2023_RunG_V3_DATA",
    "2023_Run2023Cv123": "Summer23Prompt23RunCv123_V3_DATA",
    "2023_Run2023Cv4": "Summer23Prompt23_RunCv4_V3_DATA",
    "2023BPix_Run2023D": "Summer23BPixPrompt23_RunD_V3_DATA",
    "2024_Run2024C": "Summer24Prompt24_RunCnib1_V1_DATA",
    "2024_Run2024D": "Summer24Prompt24_RunDnib1_V1_DATA",
    "2024_Run2024E": "Summer24Prompt24_RunEnib1_V1_DATA",
    "2024_Run2024F": "Summer24Prompt24_RunFnib1_V1_DATA",
    "2024_Run2024G": "Summer24Prompt24_RunGnib1_V1_DATA",
    "2024_Run2024H": "Summer24Prompt24_RunHnib1_V1_DATA",
    "2024_Run2024I": "Summer24Prompt24_RunInib1_V1_DATA",
}

jer_eras = {
    "2022_mc": "Summer2222Sep2023_JRV1_MC",
    "2022EE_mc": "Summer22EE22Sep2023_JRV1_MC",
    "2023_mc": "Summer23Prompt23RunCv1234_JRV1_MC",
    "2023BPix_mc": "Summer23BPixPrompt23RunD_JRV1_MC",
    "2024_mc": "Summer23BPixPrompt23RunD_JRV1_MC",
}

jerc_variations = {
    "JES": "JES_jes",
    "JER": "JER",
    "UES": "MET_UnclusteredEnergy",
}