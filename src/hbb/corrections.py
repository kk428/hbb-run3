"""
Collection of utilities for corrections and systematics in processors.

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import awkward as ak
import correctionlib
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.nanoaod import JetArray

ak.behavior.update(vector.behavior)
package_path = str(pathlib.Path(__file__).parent.parent.resolve())

# Important Run3 start of Run
FirstRun_2022C = 355794
FirstRun_2022D = 357487
LastRun_2022D = 359021
FirstRun_2022E = 359022
LastRun_2022F = 362180

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "photon": ["EGM", "photon.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "fatjet_jec": ["JME", "fatJet_jerc.json.gz"],
    "jet_jec": ["JME", "jet_jerc.json.gz"],
    "jetveto": ["JME", "jetvetomaps.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
    "jetid" : ["JME", "jetid.json.gz"],
}

years = {
    "2022": "2022_Summer22",
    "2022EE": "2022_Summer22EE",
    "2023": "2023_Summer23",
    "2023BPix": "2023_Summer23BPix",
    "2024": "2024_Summer24",
}


def ak_clip(arr: ak.Array, min_value: float, max_value: float):
    """
    Clip the values of an awkward array using where
    """
    return ak.where(arr < min_value, min_value, ak.where(arr > max_value, max_value, arr))


def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = years[year]

    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"


def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask

    path = Path(f"{package_path}/hbb/data/{filename}")
    return LumiMask(path)


lumiMasks = {
    "2022": build_lumimask("Cert_Collisions2022_355100_362760_Golden.json"),
    "2022EE": build_lumimask("Cert_Collisions2022_355100_362760_Golden.json"),
    "2023": build_lumimask("Cert_Collisions2023_366442_370790_Golden.json"),
    "2023BPix": build_lumimask("Cert_Collisions2023_366442_370790_Golden.json"),
    "2024": []
}


def add_pileup_weight(weights: Weights, year: str, nPU):
    # clip nPU from 0 to 150
    nPU = ak_clip(nPU, 0, 150)

    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3
    values = {}

    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))
    corr = {
        "2018": "Collisions18_UltraLegacy_goldenJSON",
        "2022": "Collisions2022_355100_357900_eraBCD_GoldenJson",
        "2022EE": "Collisions2022_359022_362760_eraEFG_GoldenJson",
        "2023": "Collisions2023_366403_369802_eraBC_GoldenJson",
        "2023BPix": "Collisions2023_369803_370790_eraD_GoldenJson",
        "2024": ""
    }[year]
    # evaluate and clip up to 4 to avoid large weights
    values["nominal"] = ak_clip(cset[corr].evaluate(nPU, "nominal"), 0, 4)
    values["up"] = ak_clip(cset[corr].evaluate(nPU, "up"), 0, 4)
    values["down"] = ak_clip(cset[corr].evaluate(nPU, "down"), 0, 4)

    weights.add("pileup", values["nominal"], values["up"], values["down"])


# Jet Veto Maps
# the JERC group recommends ALL analyses use these maps, as the JECs are derived excluding these zones.
# apply to both Data and MC
# https://cms-talk.web.cern.ch/t/jet-veto-maps-for-run3-data/18444?u=anmalara
# https://cms-talk.web.cern.ch/t/jes-for-2022-re-reco-cde-and-prompt-fg/32873
def get_jetveto_event(jets: JetArray, year: str):
    """
    Get event selection that rejects events with jets in the veto map
    """

    # correction: Non-zero value for (eta, phi) indicates that the region is vetoed
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetveto", year))
    j, nj = ak.flatten(jets), ak.num(jets)

    def get_veto(j, nj, csetstr):
        j_phi = ak_clip(j.phi, -3.1415, 3.1415)
        j_eta = ak_clip(j.eta, -4.7, 4.7)
        veto = cset[csetstr].evaluate("jetvetomap", j_eta, j_phi)
        return ak.unflatten(veto, nj)

    corr_str = {
        "2022": "Summer22_23Sep2023_RunCD_V1",
        "2022EE": "Summer22EE_23Sep2023_RunEFG_V1",
        "2023": "Summer23Prompt23_RunC_V1",
        "2023BPix": "Summer23BPixPrompt23_RunD_V1",
        "2024": ""
    }[year]

    jet_veto = get_veto(j, nj, corr_str) > 0

    event_sel = ~(ak.any((jets.pt > 15) & jet_veto, axis=1))
    return event_sel

def correct_jetid(jets, jet_type: str, year: str):
    """
    Apply jetid correction for v14+
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
    https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/199ba071f68176a815615651f8a5ae939ef0793e/examples/jetidExample.py
    """
    evaluator = correctionlib.CorrectionSet.from_file(get_pog_json("jetid", year))

    if jet_type == "AK8":
        name_tight = "AK8PUPPI_Tight"
        name_tightlv = "AK8PUPPI_TightLeptonVeto"
    elif jet_type == "AK4":
        name_tight = "AK4PUPPI_Tight"
        name_tightlv = "AK4PUPPI_TightLeptonVeto"

    flat_j = ak.flatten(jets)

    def get_jetid(j, nj, ev_str):
 
        eta    = ak_clip(j.eta, -4.7, 4.7)
        chHEF  = ak_clip(j.chHEF, 0., 1.)
        neHEF  = ak_clip(j.neHEF, 0., 1.)
        chEmEF = ak_clip(j.chEmEF, 0., 1.)
        neEmEF = ak_clip(j.neEmEF, 0., 1.)
        muEF   = ak_clip(j.muEF, 0., 1.)
        chMult = ak_clip(j.chMultiplicity, 0., 100.)
        neMult = ak_clip(j.neMultiplicity, 0., 100.)
        nConst = chMult + neMult

        jetid = evaluator[ev_str].evaluate(eta, chHEF, neHEF, chEmEF, neEmEF, muEF, chMult, neMult, nConst)
        return ak.unflatten(jetid, nj)

    jets["jetidtight"] = ak.values_astype(get_jetid(flat_j, ak.num(jets), name_tight), bool)
    jets["jetidtightlepveto"] = ak.values_astype(get_jetid(flat_j, ak.num(jets), name_tightlv), bool)

    return jets
