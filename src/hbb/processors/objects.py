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

from hbb.corrections import (
    correct_jetid,
)


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


def good_muons(muons: MuonArray):
    sel = (
        (muons.pt > 10)
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


def set_ak4jets(jets: JetArray, year):
    jets = correct_jetid(jets, "AK4", year) 
    return jets


# ak4 jet definition
def good_ak4jets(jets: JetArray):
    # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
    # PuID might only be needed for forward region (WIP)

    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    sel = (jets.pt > 30) &(abs(jets.eta) < 5.0) & (jets.jetidtight) & (jets.jetidtightlepveto)

    return jets[sel]


def set_ak8jets(fatjets: FatJetArray, year):
    fatjets["msd"] = fatjets.msoftdrop
    fatjets["qcdrho"] = 2 * np.log(fatjets.msd / fatjets.pt)
    fatjets["pnetmass"] = fatjets.particleNet_massCorr * (1 - fatjets.rawFactor) * fatjets.mass

    if "globalParT_Xcs" in fatjets.fields:
        fatjets["ParTPQCD1HF"] = fatjets.globalParT_QCD1HF
        fatjets["ParTPQCD2HF"] = fatjets.globalParT_QCD2HF
        fatjets["ParTPQCD0HF"] = fatjets.globalParT_QCD0HF
        fatjets["ParTPXbb"] = fatjets.globalParT_Xbb
        fatjets["ParTPXcc"] = fatjets.globalParT_Xcc
        fatjets["ParTPXcs"] = fatjets.globalParT_Xcs
        fatjets["ParTPXgg"] = fatjets.globalParT_Xgg
        fatjets["ParTPXqq"] = fatjets.globalParT_Xqq
        # ParT masses were trained with the masses WITHOUT the jet mass correction, so we have to undo the correction here
        fatjets["ParTmassRes"] = fatjets.globalParT_massRes * (1 - fatjets.rawFactor) * fatjets.mass
        fatjets["ParTmassVis"] = fatjets.globalParT_massVis * (1 - fatjets.rawFactor) * fatjets.mass

    fatjets = correct_jetid(fatjets, "AK8", year) 

    return fatjets


# ak8 jet definition
def good_ak8jets(fatjets: FatJetArray):
    sel = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.jetidtight
    return fatjets[sel]
