from __future__ import annotations

import awkward as ak
import numpy as np
from coffea.nanoevents.methods.nanoaod import (
    ElectronArray,
    FatJetArray,
    JetArray,
    MuonArray,
    PhotonArray,
)

from hbb.corrections import correct_jetid


def trig_match_sel(
    events, objects, trig_objects, year, trigger, filterbit, ptcut, HLTs, trig_dR=0.2
):
    """
    Returns selection for objects which are trigger matched to the specified trigger.
    """
    trigger = HLTs.hlts_by_type(year, trigger, hlt_prefix=False)[0]  # picking first trigger in list
    trig_fired = events.HLT[trigger]
    # print(f"{trigger} rate: {ak.mean(trig_fired)}")

    filterbit = 2**filterbit

    pass_trig = (trig_objects.filterBits & filterbit) == filterbit
    trig_obj = trig_objects[pass_trig]
    trig_obj_matched = ak.any(objects.metric_table(trig_obj) < trig_dR, axis=2)
    trig_obj_sel = trig_fired & trig_obj_matched & (objects.pt > ptcut)
    return trig_obj_sel


def good_photons(photons: PhotonArray):

    sel = (photons.pt > 15) & (photons.isScEtaEB | photons.isScEtaEE) & (photons.mvaID_WP80)

    return photons[sel]


def tight_photons(photons: PhotonArray):

    sel = (photons.pt > 30) & (photons.isScEtaEB | photons.isScEtaEE) & (photons.cutBased == 3)

    return photons[sel]


def good_muons(muons: MuonArray, pt_type):
    sel = (
        (getattr(muons, pt_type) > 10)
        & (np.abs(muons.eta) < 2.4)
        & (muons.looseId)
        & (muons.pfRelIso04_all < 0.15)
        & (
            ((abs(muons.eta) < 1.479) & (abs(muons.dz) < 0.1) & (abs(muons.dxy) < 0.05))
            | ((abs(muons.eta) >= 1.479) & (abs(muons.dz) < 0.2) & (abs(muons.dxy) < 0.1))
        )
    )
    return muons[sel]


def good_electrons(electrons: ElectronArray):
    sel = (
        (electrons.pt > 10)
        & (abs(electrons.eta) < 2.5)
        & (electrons.pfRelIso03_all < 0.15)
        & (electrons.mvaNoIso_WP90)
        & (
            ((abs(electrons.eta) < 1.479) & (abs(electrons.dz) < 0.1) & (abs(electrons.dxy) < 0.05))
            | (
                (abs(electrons.eta) >= 1.479)
                & (abs(electrons.dz) < 0.2)
                & (abs(electrons.dxy) < 0.1)
            )
        )
    )
    return electrons[sel]


# def set_ak4jets(jets: JetArray, year):
#     jets = correct_jetid(jets, "AK4", year) 
#     return jets

def set_ak4jets(jets: JetArray, isRealData: bool, year: str, nano_version: str, event_rho):
    """
    Jet ID fix for NanoAOD v12 copying
    # https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/117#note_8880716
    """

    if "v12" in nano_version:

        jetidtightbit = (jets.jetId & 2) == 2
        jetidtight = (
            ((np.abs(jets.eta) <= 2.7) & jetidtightbit)
            | (
                ((np.abs(jets.eta) > 2.7) & (np.abs(jets.eta) <= 3.0))
                & jetidtightbit
                & (jets.neHEF >= 0.99)
            )
            | ((np.abs(jets.eta) > 3.0) & jetidtightbit & (jets.neEmEF < 0.4))
        )

        jetidtightlepveto = (
            (np.abs(jets.eta) <= 2.7) & jetidtight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8)
        ) | ((np.abs(jets.eta) > 2.7) & jetidtight)

        jets["jetidtight"] = jetidtight
        jets["jetidtightlepveto"] = jetidtightlepveto
    else:
        jets = correct_jetid(jets, "AK4", year)

    # TODO: Add PNet pt regression

    # jerc variables
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    if not isRealData:  # only for jer
        jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)

    return jets


# ak4 jet definition
def good_ak4jets(jets: JetArray):
    # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
    # PuID might only be needed for forward region (WIP)

    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # sel = (jets.pt > 30) &(abs(jets.eta) < 5.0) & (jets.jetidtight) & (jets.jetidtightlepveto)
    sel = (
        (jets.pt > 30)
        & (jets.jetidtight)
        & (jets.jetidtightlepveto)
        & (abs(jets.eta) < 5.0)
        & ~((jets.pt <= 50) & (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0))
    )

    return jets[sel]


# def set_ak8jets(fatjets: FatJetArray, year):

#     fatjets["msd"] = fatjets.msoftdrop
#     fatjets["qcdrho"] = 2 * np.log(fatjets.msd / fatjets.pt)
#     fatjets["pnetmass"] = fatjets.particleNet_massCorr * fatjets.mass
#     fatjets["pnetXbbXcc"] = (fatjets.particleNet_XbbVsQCD + fatjets.particleNet_XccVsQCD) / (
#         fatjets.particleNet_XbbVsQCD + fatjets.particleNet_XccVsQCD + fatjets.particleNet_QCD
#     )

#     fatjets = correct_jetid(fatjets, "AK8", year) 
#     return fatjets

def set_ak8jets(fatjets: FatJetArray, isRealData: bool, year: str, nano_version: str, event_rho):

    if "v12" in nano_version:
        fatjets["jetidtight"] = fatjets.isTight
    else:
        fatjets = correct_jetid(fatjets, "AK8", year)

        fatjets["ParTPQCD"] = fatjets.globalParT3_QCD
        fatjets["ParTPXbb"] = fatjets.globalParT3_Xbb
        fatjets["ParTPXcc"] = fatjets.globalParT3_Xcc
        fatjets["ParTPXcs"] = fatjets.globalParT3_Xcs
        fatjets["ParTPXqq"] = fatjets.globalParT3_Xqq

        fatjets["ParTPXbbVsQCD"] = fatjets.globalParT3_Xbb / (
            fatjets.globalParT3_Xbb + fatjets.globalParT3_QCD
        )
        fatjets["ParTPXccVsQCD"] = fatjets.globalParT3_Xcc / (
            fatjets.globalParT3_Xcc + fatjets.globalParT3_QCD
        )
        fatjets["ParTPXbbXcc"] = (fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc) / (
            fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc + fatjets.globalParT3_QCD
        )

        # ParT masses were trained with the masses WITHOUT the jet mass correction, so we have to undo the correction here
        fatjets["ParTmassGeneric"] = (
            fatjets.globalParT3_massCorrGeneric * (1 - fatjets.rawFactor) * fatjets.mass
        )
        fatjets["ParTmassX2p"] = (
            fatjets.globalParT3_massCorrX2p * (1 - fatjets.rawFactor) * fatjets.mass
        )
    # jerc variables
    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
    fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass
    fatjets["event_rho"] = ak.broadcast_arrays(event_rho, fatjets.pt)[0]
    if not isRealData:  # only for jer
        fatjets["pt_gen"] = ak.values_astype(ak.fill_none(fatjets.matched_gen.pt, 0), np.float32)
    return fatjets


# ak8 jet definition
def good_ak8jets(fatjets: FatJetArray):
    # sel = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.jetidtight
    sel = fatjets.jetidtight & (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5)
    return fatjets[sel]
