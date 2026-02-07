from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import numpy as np
import xgboost as xgb
from coffea.analysis_tools import PackedSelection, Weights
from coffea.ml_tools import xgboost_wrapper
from hist.dask import Hist

from hbb.corrections import (
    add_btag_weights,
    add_muon_weights,
    add_pdf_weight,
    add_photon_weights,
    add_pileup_weight,
    add_ps_weight,
    add_scalevar_3pt,
    add_scalevar_7pt,
    apply_jerc,
    correct_met,
    correct_muons,
    get_jetveto_event,
    lumiMasks,
    mupt_variations,
)
from hbb.jerc_eras import jerc_variations, run_map
from hbb.processors.SkimmerABC import SkimmerABC
from hbb.taggers import b_taggers

from .GenSelection import (
    bosonFlavor,
    gen_selection_Hbb,
    gen_selection_V,
    gen_selection_Vg,
    getBosons,
)
from .objects import (
    good_ak4jets,
    good_ak8jets,
    good_electrons,
    good_muons,
    good_photons,
    set_ak4jets,
    set_ak8jets,
    tight_photons,
)

logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "Hto2B": gen_selection_Hbb,
    "Wto2Q-": gen_selection_V,
    "Zto2Q-": gen_selection_V,
    "ZGto2QG-": gen_selection_Vg,
}


def get_BDT_model(BDT_file: str):
    bdt_features = [
        "nFatJet",
        "nJet",
        "FatJet0_phi",
        "FatJet0_eta",
        "FatJet0_n2b1",
        "FatJet0_n3b1",
        "FatJet1_pt",
        "FatJet1_phi",
        "FatJet1_eta",
        "FatJet1_msd",
        "FatJet1_pnetMass",
        "FatJet1_pnetTXbb",
        "FatJet1_pnetTXcc",
        "FatJet1_pnetTXqq",
        "FatJet1_pnetTXgg",
        "VBFPair_mjj",
        "VBFPair_deta",
        "Photon0_pt",
        "Jet0_pt",
        "Jet0_eta",
        "Jet0_phi",
        "Jet0_mass",
        "Jet0_btagPNetB",
        "Jet0_btagPNetCvB",
        "Jet0_btagPNetCvL",
        "Jet0_btagPNetQvG",
        "Jet1_pt",
        "Jet1_eta",
        "Jet1_phi",
        "Jet1_mass",
        "Jet1_btagPNetB",
        "Jet1_btagPNetCvB",
        "Jet1_btagPNetCvL",
        "Jet1_btagPNetQvG",
        "Jet2_pt",
        "Jet2_eta",
        "Jet2_phi",
        "Jet2_mass",
        "Jet2_btagPNetB",
        "Jet2_btagPNetCvB",
        "Jet2_btagPNetCvL",
        "Jet2_btagPNetQvG",
        "Jet3_pt",
        "Jet3_eta",
        "Jet3_phi",
        "Jet3_mass",
        "Jet3_btagPNetB",
        "Jet4_btagPNetCvB",
        "Jet4_btagPNetCvL",
        "Jet4_btagPNetQvG",
        "JetClosestFatJet0_pt",
        "JetClosestFatJet0_eta",
        "JetClosestFatJet0_phi",
        "JetClosestFatJet0_mass",
    ]

    class xgboost_model(xgboost_wrapper):
        # Define how to prepare awkward arrays for BDT evaluation
        def prepare_awkward(self, events):
            features = []
            for name in bdt_features:
                feat = events[name]
                feat = ak.fill_none(feat, -999.0)
                features.append(feat[:, np.newaxis])
            ret = ak.concatenate(features, axis=1)
            return [], dict(data=ret)

    booster = xgb.Booster()
    booster.load_model(Path.cwd() / BDT_file)
    booster.feature_names = None  # Disable feature name checking
    model = xgboost_model(booster)
    return model


class categorizer(SkimmerABC):
    def __init__(
        self,
        year="2022",
        nano_version="v12",
        xsecs: dict = None,
        systematics=False,
        save_skim=False,
        skim_outpath="",
        evaluate_BDT=True,
        btag_eff=False,
        save_skim_nosysts=False,
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        self._year = year
        self._nano_version = nano_version
        self._systematics = systematics
        self._skip_syst = save_skim_nosysts
        self._save_skim = save_skim
        if self._skip_syst:
            self._save_skim = True
        self._skim_outpath = skim_outpath
        self._evaluate_BDT = evaluate_BDT
        self._btag_eff = btag_eff
        self._btagger, self._btag_wp = "btagPNetB", "M"
        if year == "2024":
            self._btagger = "btagUParTAK4B"
        self._btag_cut = b_taggers[self._year]["AK4"][self._btagger][self._btag_wp]
        self._mupt_type = "ptcorr"
        if self._evaluate_BDT:
            self.bdt_model = get_BDT_model("src/hbb/data/MultiClassBDT_23Oct25.ubj")

        with Path("src/hbb/muon_triggers.json").open() as f:
            self._muontriggers = json.load(f)

        with Path("src/hbb/egamma_triggers.json").open() as f:
            self._egammatriggers = json.load(f)

        with Path("src/hbb/triggers.json").open() as f:
            self._triggers = json.load(f)

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with Path("src/hbb/metfilters.json").open() as f:
            self._met_filters = json.load(f)

        self.make_output = lambda: {
            "sumw": {},
            "cutflow": Hist.new.StrCat([], growth=True, name="region", label="Region")
            .StrCat([], growth=True, name="dataset", label="Dataset")
            .Reg(15, 0, 15, name="cut", label="Cut index")
            .Variable([0, 1, 2, 3, 4], name="genflavor", label="Gen. jet flavor")
            .Weight(),
            "btagWeight": Hist.new.Reg(50, 0, 3, name="val", label="BTag correction").Weight(),
            "skim": {},
        }

        # btag efficiency plots - binning according to:
        # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#b-tagging-efficiencies-in-simulation
        self.make_btag_output = lambda: (
            Hist.new.StrCat([], growth=True, name="tagger", label="Tagger")
            .Reg(2, 0, 2, name="passWP", label="passWP")
            .Variable([0, 4, 5], name="flavor", label="Jet hadronFlavour")
            .Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", label="Jet pt")
            .Reg(4, 0, 2.5, name="abseta", label="Jet abseta")
            .Weight()
        )

    def process(self, events):

        # process only nominal case
        if self._skip_syst or not self._save_skim or not hasattr(events, "genWeight"):
            return {"nominal": self.process_shift(events, "nominal")}

        """
        Add `Up and Down` to total list of energy variations
        Muon Energy: `MuonPTScale and MuonPTResolution` defined in mupt_variations
        Jet Energy: `JES, JER, UES` defined in jerc_variations
        """
        total_variations = (
            ["nominal"]
            + [f"{var}_{dir}" for var in jerc_variations for dir in ["Up", "Down"]]
            + [f"{var}_{dir}" for var in mupt_variations for dir in ["Up", "Down"]]
        )

        """
        run processor for each shift defined in total_variations
        return output as dict {variation: output}
        """
        return {var: self.process_shift(events, var) for var in total_variations}

    def add_weights(
        self, weights, events, dataset, btag_jets, muons=None, photons=None
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights.add("genweight", events.genWeight)

        btag_SF = ak.ones_like(events.run)
        if not self._skip_syst:
            add_pileup_weight(weights, self._year, events.Pileup.nPU)
            add_ps_weight(weights, events.PSWeight)
            if not self._btag_eff:
                btag_SF = add_btag_weights(
                    weights, btag_jets, self._btagger, self._btag_wp, self._year
                )

            # Easier to save nominal weights for rest of MC with all of the syst names for grabbing columns in post-processing
            # Need to fix
            # flag_syst = ("Hto2B" in dataset) or ("Hto2C" in dataset) or ("VBFZto" in dataset)
            # add_pdf_weight(weights, getattr(events, "LHEPdfWeight", None) if flag_syst else None)
            # add_scalevar_7pt(
            #     weights, getattr(events, "LHEScaleWeight", None) if flag_syst else None
            # )
            # add_scalevar_3pt(
            #     weights, getattr(events, "LHEScaleWeight", None) if flag_syst else None
            # )

            if muons is not None:
                add_muon_weights(weights, self._year, muons, self._mupt_type)
            if photons is not None:
                add_photon_weights(weights, self._year, photons)

        logger.debug("weights", extra=weights._weights.keys())
        # logger.debug(f"Weight statistics: {weights.weightStatistics!r}")

        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.weight()

        # systematics
        for systematic in weights.variations:
            weights_dict[systematic] = weights.weight(modifier=systematic)

        ###################### Normalization (Step 1) ######################
        # strip the year from the dataset name
        dataset_no_year = dataset.replace(f"{self._year}_", "")
        weight_norm = self.get_dataset_norm(self._year, dataset_no_year)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()

        return weights_dict, totals_dict, btag_SF

    def process_shift(self, events, shift_name):

        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        output = self.make_output() if not self._btag_eff else self.make_btag_output()
        weights = Weights(None, storeIndividual=True)
        weights_mu = Weights(None, storeIndividual=True)
        weights_gamma = Weights(None, storeIndividual=True)
        if shift_name == "nominal" and not isRealData and not self._btag_eff:
            output["sumw"][dataset] = ak.sum(events.genWeight)

        trigger = ak.values_astype(ak.zeros_like(events.run), bool)
        for t in self._triggers[self._year]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        selection.add("trigger", trigger)
        del trigger

        if isRealData:
            selection.add("lumimask", lumiMasks[self._year[:4]](events.run, events.luminosityBlock))
        else:
            selection.add("lumimask", ak.values_astype(ak.ones_like(events.run), bool))

        trigger = ak.values_astype(ak.zeros_like(events.run), bool)
        for t in self._muontriggers[self._year]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        selection.add("muontrigger", trigger)
        del trigger

        trigger = ak.values_astype(ak.zeros_like(events.run), bool)
        for t in self._egammatriggers[self._year]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        selection.add("egammatrigger", trigger)
        del trigger

        metfilter = ak.values_astype(ak.ones_like(events.run), bool)
        for flag in self._met_filters[self._year]["data" if isRealData else "mc"]:
            if flag in events.Flag.fields:
                metfilter = metfilter & events.Flag[flag]
        selection.add("metfilter", metfilter)
        del metfilter

        mc_run = "mc"
        if isRealData:
            for keys, value in run_map.items():
                if any(k in dataset for k in keys):
                    mc_run = value
                    break
        jec_key = f"{self._year}_{mc_run}"

        fatjets = set_ak8jets(
            events.FatJet, isRealData, self._year, self._nano_version, events.Rho.fixedGridRhoFastjetAll
        )
        jets = set_ak4jets(
            events.Jet, isRealData, self._year, self._nano_version, events.Rho.fixedGridRhoFastjetAll
        )

        if self._nano_version == "v14_private" or self._nano_version == "v15":
            # subjets in PFNano reprocessing break the fatjet jercs for whatever reason
            keep_fields = [
                f
                for f in fatjets.fields
                if (
                    ("nConstituents" not in f)
                    and ("IdxG" not in f)
                    and ("Idx1G" not in f)
                    and ("Idx2G" not in f)
                )
            ]

            fatjets = ak.zip(
                {f: fatjets[f] for f in keep_fields}, with_name="FatJet", behavior=fatjets.behavior
            )
        met = events.PuppiMET
        # Apply jerc corrections to jets, fatjets, and met collections
        if not self._skip_syst:
            jets = apply_jerc(jets, "AK4", self._year, jec_key)
            fatjets = apply_jerc(fatjets, "AK8", self._year, jec_key)
            met = correct_met(met, jets)  # PuppiMET Recommended for Run3

        # Select jets, fatjets, and met collections according to jerc variation shift
        if shift_name != "nominal" and "Muon" not in shift_name:
            var, direction = shift_name.split("_")
            attr = jerc_variations[var]
            if var in ("JES", "JER"):
                jets = getattr(getattr(jets, attr), direction.lower())
                fatjets = getattr(getattr(fatjets, attr), direction.lower())
                met = getattr(getattr(met, attr), direction.lower())
            elif var == "UES":
                met = getattr(getattr(met, attr), direction.lower())

        goodfatjets = good_ak8jets(fatjets)
        goodjets = good_ak4jets(jets)

        cut_jetveto = get_jetveto_event(jets, self._year)
        selection.add("ak4jetveto", cut_jetveto)

        selection.add("2FJ", ak.num(goodfatjets, axis=1) == 2)
        selection.add("not2FJ", ak.num(goodfatjets, axis=1) != 2)

        if "v12" in self._nano_version:
            xbbfatjets = goodfatjets[ak.argsort(goodfatjets.pnetXbbXcc, axis=1, ascending=False)]
        else:
            xbbfatjets = goodfatjets[ak.argsort(goodfatjets.ParTPXbbXcc, axis=1, ascending=False)]

        candidatejet = ak.firsts(xbbfatjets[:, 0:1])
        subleadingjet = ak.firsts(xbbfatjets[:, 1:2])

        selection.add(
            "minjetkin",
            (candidatejet.pt >= 300) & (candidatejet.pt < 1200) & (candidatejet.msd >= 40.0)
            # & (candidatejet.msd < 201.0)
            & (abs(candidatejet.eta) < 2.5),
        )

        selection.add(
            "minjetkin_zgamma",
            (candidatejet.pt >= 200)  # Loosened pt cut
            & (candidatejet.pt < 1200)
            & (candidatejet.msd >= 0.0)  # Loosened msd cut
            & (candidatejet.msd < 201.0)
            & (abs(candidatejet.eta) < 2.5),
        )

        selection.add("particleNetXbbpass", (candidatejet.particleNet_XbbVsQCD >= 0.5))

        # only consider 4 AK4 jets leading in pT to be consistent with old framework
        jets = goodjets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        dR = jets.delta_r(candidatejet)
        ak4_opphem_ak8 = jets[dphi > np.pi / 2]
        ak4_outside_ak8 = jets[dR > 0.8]

        # ak4 closest to ak8
        ak4_closest_ak8 = ak.firsts(
            ak4_outside_ak8[ak.argmin(ak4_outside_ak8.delta_r(candidatejet), axis=1, keepdims=True)]
        )

        selection.add(
            "antiak4btagMediumOppHem",
            ak.max(getattr(ak4_opphem_ak8, self._btagger), axis=1, mask_identity=False)
            < self._btag_cut,
        )
        selection.add(
            "antiak4btagMedium",
            ak.max(getattr(ak4_outside_ak8, self._btagger), axis=1, mask_identity=False)
            < self._btag_cut,
        )
        selection.add(
            "ak4btagMedium08",
            ak.max(getattr(ak4_outside_ak8, self._btagger), axis=1, mask_identity=False)
            > self._btag_cut,
        )

        selection.add("lowmet", met.pt < 140.0)

        # VBF specific variables
        jet1_away = ak.firsts(ak4_outside_ak8[:, 0:1])
        jet2_away = ak.firsts(ak4_outside_ak8[:, 1:2])
        jet3_away = ak.firsts(ak4_outside_ak8[:, 2:3])
        jet4_away = ak.firsts(ak4_outside_ak8[:, 3:4])

        vbf_deta = abs(jet1_away.eta - jet2_away.eta)
        vbf_mjj = (jet1_away + jet2_away).mass
        vbf_deta = ak.fill_none(vbf_deta, -1)
        vbf_mjj = ak.fill_none(vbf_mjj, -1)

        isvbf = (vbf_deta > 3.5) & (vbf_mjj > 1000)
        isvbf = ak.fill_none(isvbf, False)
        isnotvbf = ak.fill_none(~isvbf, True)

        selection.add("isvbf", isvbf)
        selection.add("notvbf", isnotvbf)

        muons = correct_muons(events.Muon, events, self._year, isRealData)
        if shift_name != "nominal" and "Muon" in shift_name:
            var, direction = shift_name.split("_")
            self._mupt_type = f"{mupt_variations[var]}_{direction.lower()}"

        goodmuon = good_muons(muons, self._mupt_type)
        nmuons = ak.num(goodmuon, axis=1)
        leadingmuon = ak.firsts(goodmuon)
        ttbarmuon = ak.firsts(goodmuon[getattr(goodmuon, self._mupt_type) > 55.0])
        # low pt muons break sf (lower bound 15GeV)

        goodelectron = good_electrons(events.Electron)
        nelectrons = ak.num(goodelectron, axis=1)

        selection.add("noleptons", (nmuons == 0) & (nelectrons == 0))
        selection.add("onemuon", (nmuons == 1) & (nelectrons == 0))
        selection.add(
            "muonkin", (getattr(leadingmuon, self._mupt_type) > 55.0) & (abs(leadingmuon.eta) < 2.1)
        )
        selection.add("muonDphiAK8", abs(leadingmuon.delta_phi(candidatejet)) > 2 * np.pi / 3)

        goodphotons = good_photons(events.Photon)
        nphotons = ak.num(goodphotons, axis=1)

        ntightphotons = ak.num(tight_photons(events.Photon), axis=1)
        vgammaphoton = ak.firsts(tight_photons(events.Photon))

        selection.add("onephoton", (nphotons == 1))
        selection.add("atleastonephoton", (ntightphotons >= 1))
        selection.add("passphotonveto", (nphotons == 0))

        if self._evaluate_BDT:
            # Construct BDT input
            bdt_ak_array = {
                "nFatJet": ak.num(goodfatjets, axis=1),
                "nJet": ak.num(goodjets, axis=1),
                "FatJet0_phi": candidatejet.phi,
                "FatJet0_eta": candidatejet.eta,
                "FatJet0_n2b1": candidatejet.n2b1,
                "FatJet0_n3b1": candidatejet.n3b1,
                "FatJet1_pt": subleadingjet.pt,
                "FatJet1_phi": subleadingjet.phi,
                "FatJet1_eta": subleadingjet.eta,
                "FatJet1_msd": subleadingjet.msd,
                "FatJet1_pnetMass": subleadingjet.pnetmass,
                "FatJet1_pnetTXbb": subleadingjet.particleNet_XbbVsQCD,
                "FatJet1_pnetTXcc": subleadingjet.particleNet_XccVsQCD,
                "FatJet1_pnetTXqq": subleadingjet.particleNet_XqqVsQCD,
                "FatJet1_pnetTXgg": subleadingjet.particleNet_XggVsQCD,
                "VBFPair_mjj": vbf_mjj,
                "VBFPair_deta": vbf_deta,
                "Photon0_pt": vgammaphoton.pt,
                "Jet0_pt": jet1_away.pt,
                "Jet0_eta": jet1_away.eta,
                "Jet0_phi": jet1_away.phi,
                "Jet0_mass": jet1_away.mass,
                "Jet0_btagPNetB": jet1_away.btagPNetB,
                "Jet0_btagPNetCvB": jet1_away.btagPNetCvB,
                "Jet0_btagPNetCvL": jet1_away.btagPNetCvL,
                "Jet0_btagPNetQvG": jet1_away.btagPNetQvG,
                "Jet1_pt": jet2_away.pt,
                "Jet1_eta": jet2_away.eta,
                "Jet1_phi": jet2_away.phi,
                "Jet1_mass": jet2_away.mass,
                "Jet1_btagPNetB": jet2_away.btagPNetB,
                "Jet1_btagPNetCvB": jet2_away.btagPNetCvB,
                "Jet1_btagPNetCvL": jet2_away.btagPNetCvL,
                "Jet1_btagPNetQvG": jet2_away.btagPNetQvG,
                "Jet2_pt": jet3_away.pt,
                "Jet2_eta": jet3_away.eta,
                "Jet2_phi": jet3_away.phi,
                "Jet2_mass": jet3_away.mass,
                "Jet2_btagPNetB": jet3_away.btagPNetB,
                "Jet2_btagPNetCvB": jet3_away.btagPNetCvB,
                "Jet2_btagPNetCvL": jet3_away.btagPNetCvL,
                "Jet2_btagPNetQvG": jet3_away.btagPNetQvG,
                "Jet3_pt": jet4_away.pt,
                "Jet3_eta": jet4_away.eta,
                "Jet3_phi": jet4_away.phi,
                "Jet3_mass": jet4_away.mass,
                "Jet3_btagPNetB": jet4_away.btagPNetB,
                "Jet4_btagPNetCvB": jet4_away.btagPNetCvB,
                "Jet4_btagPNetCvL": jet4_away.btagPNetCvL,
                "Jet4_btagPNetQvG": jet4_away.btagPNetQvG,
                "JetClosestFatJet0_pt": ak4_closest_ak8.pt,
                "JetClosestFatJet0_eta": ak4_closest_ak8.eta,
                "JetClosestFatJet0_phi": ak4_closest_ak8.phi,
                "JetClosestFatJet0_mass": ak4_closest_ak8.mass,
            }
            bdt_input = ak.zip(bdt_ak_array, depth_limit=1)

            # Evaluate BDT
            bdt_model = self.bdt_model
            bdt_scores = bdt_model(bdt_input)

            # assign scores to selections
            selection.add("BDTisVBF", (bdt_scores == 0))
            selection.add("BDTisVH", (bdt_scores == 1))
            selection.add("BDTisggF", (bdt_scores == 2))

        gen_variables = {}
        btag_SF = ak.ones_like(events.run)
        if isRealData:
            genflavor = ak.zeros_like(candidatejet.pt)
            genBosonPt = ak.zeros_like(candidatejet.pt)
        else:
            # signal regions
            weights_dict, totals_temp, btag_SF = self.add_weights(
                weights, events, dataset, ak4_opphem_ak8
            )
            # muon region
            weights_dict_mu, totals_temp_mu, btag_SF_mu = self.add_weights(
                weights_mu, events, dataset, ak4_outside_ak8, muons=ttbarmuon
            )
            # gamma region
            weights_dict_gamma, totals_temp_gamma, btag_SF_gamma = self.add_weights(
                weights_gamma, events, dataset, ak4_outside_ak8, photons=vgammaphoton
            )

            for d, gen_func in gen_selection_dict.items():
                if d in dataset:
                    # match goodfatjets
                    gen_variables = gen_func(events, goodfatjets)

            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            match_mask = (abs(candidatejet.pt - matchedBoson.pt) / matchedBoson.pt < 0.5) & (
                abs(candidatejet.msd - matchedBoson.mass) / matchedBoson.mass < 0.3
            )
            selmatchedBoson = ak.mask(matchedBoson, match_mask)
            genflavor = bosonFlavor(selmatchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)

        # softdrop mass, 0 for genflavor == 0
        msd_matched = candidatejet.msd * (genflavor > 0) + candidatejet.msd * (genflavor == 0)

        regions = {
            "signal-all": [
                "trigger",
                "lumimask",
                "metfilter",
                # "ak4jetveto",
                # "minjetkin",
                # "antiak4btagMediumOppHem",
                # "lowmet",
                # "noleptons",
            ],
            # "signal-ggf": [
            #     "trigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "antiak4btagMediumOppHem",
            #     "lowmet",
            #     "noleptons",
            #     "notvbf",
            #     "not2FJ",
            # ],
            # "signal-vh": [
            #     "trigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "antiak4btagMediumOppHem",
            #     "lowmet",
            #     "noleptons",
            #     "notvbf",
            #     "2FJ",
            # ],
            # "signal-vbf": [
            #     "trigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "antiak4btagMediumOppHem",
            #     "lowmet",
            #     "noleptons",
            #     "isvbf",
            # ],
            # "control-tt": [
            #     "muontrigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "ak4btagMedium08",
            #     "onemuon",
            #     "muonkin",
            #     "muonDphiAK8",
            # ],
            # "control-zgamma": [
            #     "egammatrigger",
            #     "lumimask",
            #     "metfilter",
            #     "minjetkin_zgamma",
            #     "atleastonephoton",
            #     "antiak4btagMedium",
            # ],
        }
        if self._evaluate_BDT:
            regions.update(
                {
                    "signal-ggf-BDT": [
                        "trigger",
                        "lumimask",
                        "metfilter",
                        "ak4jetveto",
                        "minjetkin",
                        "antiak4btagMediumOppHem",
                        "lowmet",
                        "noleptons",
                        # "notvbf",
                        # "not2FJ",
                        "BDTisggF",
                    ],
                    "signal-vh-BDT": [
                        "trigger",
                        "lumimask",
                        "metfilter",
                        "ak4jetveto",
                        "minjetkin",
                        "antiak4btagMediumOppHem",
                        "lowmet",
                        "noleptons",
                        # "notvbf",
                        # "2FJ",
                        "BDTisVH",
                    ],
                    "signal-vbf-BDT": [
                        "trigger",
                        "lumimask",
                        "metfilter",
                        "ak4jetveto",
                        "minjetkin",
                        "antiak4btagMediumOppHem",
                        "lowmet",
                        "noleptons",
                        # "isvbf",
                        "BDTisVBF",
                    ],
                }
            )

        btag_eff_cuts = [
            "trigger",
            "lumimask",
            "metfilter",
            "ak4jetveto",
            "minjetkin",
            "lowmet",
            "noleptons",
        ]

        tic = time.time()

        nominal_weight = ak.ones_like(events.run) if isRealData else weights_dict["weight"]
        gen_weight = ak.ones_like(events.run) if isRealData else events.genWeight

        egamma_trigger_booleans = {}
        for t in self._egammatriggers[self._year]:
            if t in events.HLT.fields:
                egamma_trigger_booleans[t] = events.HLT[t]
            else:
                egamma_trigger_booleans[t] = ak.values_astype(ak.zeros_like(events.run), bool)

        if self._btag_eff:
            cut = selection.all(*btag_eff_cuts)
            flat_gj = ak.flatten(goodjets)

            output.fill(
                tagger=self._btagger,
                abseta=self.normalize(abs(flat_gj.eta), cut),
                pt=self.normalize(flat_gj.pt, cut),
                flavor=self.normalize(flat_gj.hadronFlavour, cut),
                passWP=self.normalize(getattr(flat_gj, self._btagger) > self._btag_cut, cut),
            )
            return output

        output_array = None
        if self._save_skim:

            output_array = {
                "GenBoson_pt": genBosonPt,
                "GenFlavor": genflavor,
                "nFatJet": ak.num(goodfatjets, axis=1),
                "nJet": ak.num(goodjets, axis=1),
                "FatJet0_pt": candidatejet.pt,
                "FatJet0_phi": candidatejet.phi,
                "FatJet0_eta": candidatejet.eta,
                "FatJet0_msd": candidatejet.msd,
                "FatJet0_msdmatched": msd_matched,
                "FatJet0_n2b1": candidatejet.n2b1,
                "FatJet0_n3b1": candidatejet.n3b1,
                # "FatJet0_pnetMass": candidatejet.pnetmass,
                # "FatJet0_pnetTXbb": candidatejet.particleNet_XbbVsQCD,
                # "FatJet0_pnetTXcc": candidatejet.particleNet_XccVsQCD,
                # "FatJet0_pnetTXqq": candidatejet.particleNet_XqqVsQCD,
                # "FatJet0_pnetTXgg": candidatejet.particleNet_XggVsQCD,
                # "FatJet0_pnetTQCD": candidatejet.particleNet_QCD,
                # "FatJet0_pnetXbbXcc": candidatejet.pnetXbbXcc,
                "FatJet1_pt": subleadingjet.pt,
                "FatJet1_phi": subleadingjet.phi,
                "FatJet1_eta": subleadingjet.eta,
                "FatJet1_msd": subleadingjet.msd,
                # "FatJet1_pnetMass": subleadingjet.pnetmass,
                # "FatJet1_pnetTXbb": subleadingjet.particleNet_XbbVsQCD,
                # "FatJet1_pnetTXcc": subleadingjet.particleNet_XccVsQCD,
                # "FatJet1_pnetTXqq": subleadingjet.particleNet_XqqVsQCD,
                # "FatJet1_pnetTXgg": subleadingjet.particleNet_XggVsQCD,
                "VBFPair_mjj": vbf_mjj,
                "VBFPair_deta": vbf_deta,
                "Photon0_pt": vgammaphoton.pt,
                "Photon0_phi": vgammaphoton.phi,
                "Photon0_eta": vgammaphoton.eta,
                "MET": met.pt,
                "weight": nominal_weight,
                "genWeight": gen_weight,
                # **gen_variables,
                # **egamma_trigger_booleans,
            }
            if self._evaluate_BDT:
                output_array.update({"BDT_score": bdt_scores})

            # reduced output array for energy variation shift
            energy_var_array = {
                "GenBoson_pt": genBosonPt,
                "GenFlavor": genflavor,
                "FatJet0_pt": candidatejet.pt,
                "FatJet0_msd": candidatejet.msd,
                "FatJet0_msdmatched": msd_matched,
                "FatJet0_pnetTXbb": candidatejet.particleNet_XbbVsQCD,
                "FatJet0_pnetTXcc": candidatejet.particleNet_XccVsQCD,
                "FatJet0_pnetXbbXcc": candidatejet.pnetXbbXcc,
                "VBFPair_mjj": vbf_mjj,
                "weight": nominal_weight,
                "genWeight": gen_weight,
            }

            if "v12" not in self._nano_version:
                parT_array = {
                    "FatJet0_ParTPQCD": candidatejet.ParTPQCD,
                    "FatJet0_ParTPXbb": candidatejet.ParTPXbb,
                    "FatJet0_ParTPXcc": candidatejet.ParTPXcc,
                    "FatJet0_ParTPXqq": candidatejet.ParTPXqq,
                    "FatJet0_ParTPXcs": candidatejet.ParTPXcs,
                    "FatJet0_ParTPXbbVsQCD": candidatejet.ParTPXbbVsQCD,
                    "FatJet0_ParTPXccVsQCD": candidatejet.ParTPXccVsQCD,
                    "FatJet0_ParTPXbbXcc": candidatejet.ParTPXbbXcc,
                    "FatJet0_ParTmassGeneric": candidatejet.ParTmassGeneric,
                    "FatJet0_ParTmassX2p": candidatejet.ParTmassX2p,
                    "FatJet1_ParTPQCD": subleadingjet.ParTPQCD,
                    "FatJet1_ParTPXbb": subleadingjet.ParTPXbb,
                    "FatJet1_ParTPXcc": subleadingjet.ParTPXcc,
                    "FatJet1_ParTPXqq": subleadingjet.ParTPXqq,
                    "FatJet1_ParTPXcs": subleadingjet.ParTPXcs,
                    "FatJet1_ParTPXbbVsQCD": subleadingjet.ParTPXbbVsQCD,
                    "FatJet1_ParTPXccVsQCD": subleadingjet.ParTPXccVsQCD,
                    "FatJet1_ParTPXbbXcc": subleadingjet.ParTPXbbXcc,
                    "FatJet1_ParTmassGeneric": subleadingjet.ParTmassGeneric,
                    "FatJet1_ParTmassX2p": subleadingjet.ParTmassX2p,
                }
                output_array = {**output_array, **parT_array}

                energy_var_array_parT = {
                    "FatJet0_ParTPXbbVsQCD": candidatejet.ParTPXbbVsQCD,
                    "FatJet0_ParTPXccVsQCD": candidatejet.ParTPXccVsQCD,
                    "FatJet0_ParTPXbbXcc": candidatejet.ParTPXbbXcc,
                }
                energy_var_array = {**energy_var_array, **energy_var_array_parT}

            # extra variables for big array
            output_array_extra = {
                # AK4 Jets away from FatJet0
                "Jet0_pt": jet1_away.pt,
                "Jet0_eta": jet1_away.eta,
                "Jet0_phi": jet1_away.phi,
                "Jet0_mass": jet1_away.mass,
                # "Jet0_btagPNetB": jet1_away.btagPNetB,
                # "Jet0_btagPNetCvB": jet1_away.btagPNetCvB,
                # "Jet0_btagPNetCvL": jet1_away.btagPNetCvL,
                # "Jet0_btagPNetQvG": jet1_away.btagPNetQvG,
                "Jet1_pt": jet2_away.pt,
                "Jet1_eta": jet2_away.eta,
                "Jet1_phi": jet2_away.phi,
                "Jet1_mass": jet2_away.mass,
                # "Jet1_btagPNetB": jet2_away.btagPNetB,
                # "Jet1_btagPNetCvB": jet2_away.btagPNetCvB,
                # "Jet1_btagPNetCvL": jet2_away.btagPNetCvL,
                # "Jet1_btagPNetQvG": jet2_away.btagPNetQvG,
                "Jet2_pt": jet3_away.pt,
                "Jet2_eta": jet3_away.eta,
                "Jet2_phi": jet3_away.phi,
                "Jet2_mass": jet3_away.mass,
                # "Jet2_btagPNetB": jet3_away.btagPNetB,
                # "Jet2_btagPNetCvB": jet3_away.btagPNetCvB,
                # "Jet2_btagPNetCvL": jet3_away.btagPNetCvL,
                # "Jet2_btagPNetQvG": jet3_away.btagPNetQvG,
                # "Jet3_pt": jet4_away.pt,
                # "Jet3_eta": jet4_away.eta,
                # "Jet3_phi": jet4_away.phi,
                # "Jet3_mass": jet4_away.mass,
                # "Jet3_btagPNetB": jet4_away.btagPNetB,
                # "Jet4_btagPNetCvB": jet4_away.btagPNetCvB,
                # "Jet4_btagPNetCvL": jet4_away.btagPNetCvL,
                # "Jet4_btagPNetQvG": jet4_away.btagPNetQvG,
                # AK4 Jet away but closest to FatJet0
                "JetClosestFatJet0_pt": ak4_closest_ak8.pt,
                "JetClosestFatJet0_eta": ak4_closest_ak8.eta,
                "JetClosestFatJet0_phi": ak4_closest_ak8.phi,
                "JetClosestFatJet0_mass": ak4_closest_ak8.mass,
            }

        def skim(region, output_array):
            selections = regions[region]
            cut = selection.all(*selections)

            # to debug...
            # print(output_array.compute())
            # print(output_array[cut].compute())

            if "root:" in self._skim_outpath:
                skim_path = f"{self._skim_outpath}/{shift_name.replace('_', '')}/{self._year}/{dataset}/{region}"
            else:
                skim_path = (
                    Path(self._skim_outpath)
                    / shift_name.replace("_", "")
                    / self._year
                    / dataset
                    / region
                )
                skim_path.mkdir(parents=True, exist_ok=True)
            print("Saving skim to: ", skim_path)

            # possible TODO: add systematic weights?
            output["skim"][region] = dak.to_parquet(
                output_array[cut],
                str(skim_path),
                compute=False,
            )

            if shift_name == "nominal":

                # Fill cutflow hist
                allcuts = set()
                cut = selection.all(*allcuts)
                output["cutflow"].fill(
                    dataset=dataset,
                    region=region,
                    genflavor=self.normalize(genflavor, None),
                    cut=0,
                    weight=nominal_weight,
                )
                for i, cut in enumerate(selections):
                    allcuts.add(cut)
                    cumulative_cut = selection.all(*allcuts)
                    output["cutflow"].fill(
                        dataset=dataset,
                        region=region,
                        genflavor=self.normalize(genflavor, cumulative_cut),
                        cut=i + 1,
                        weight=nominal_weight[cumulative_cut],
                    )

                # Fill btag SF hist
                cut = selection.all(*selections)
                output["btagWeight"].fill(val=self.normalize(btag_SF, cut))

        if self._save_skim:
            if shift_name == "nominal":
                for region in regions:
                    if region == "signal-all":
                        skim(region, ak.zip({**output_array, **output_array_extra}, depth_limit=1))
                    else:
                        if isRealData:
                            skim(region, ak.zip(output_array, depth_limit=1))
                        else:
                            if "signal" in region:
                                skim(
                                    region, ak.zip({**output_array, **weights_dict}, depth_limit=1)
                                )
                            elif region == "control-tt":
                                output_array["weight"] = (
                                    ak.ones_like(events.run)
                                    if isRealData
                                    else weights_dict_mu["weight"]
                                )
                                skim(
                                    region,
                                    ak.zip({**output_array, **weights_dict_mu}, depth_limit=1),
                                )
                            elif region == "control-zgamma":
                                output_array["weight"] = (
                                    ak.ones_like(events.run)
                                    if isRealData
                                    else weights_dict_gamma["weight"]
                                )
                                skim(
                                    region,
                                    ak.zip({**output_array, **weights_dict_gamma}, depth_limit=1),
                                )

            else:  # energy variation shift case
                for region in regions:
                    if region != "signal-all":
                        if isRealData:
                            skim(region, ak.zip(energy_var_array, depth_limit=1))
                        else:
                            if "signal" in region:
                                skim(
                                    region,
                                    ak.zip({**energy_var_array, **weights_dict}, depth_limit=1),
                                )
                            elif region == "control-tt":
                                output_array["weight"] = (
                                    ak.ones_like(events.run)
                                    if isRealData
                                    else weights_dict_mu["weight"]
                                )
                                skim(
                                    region,
                                    ak.zip({**energy_var_array, **weights_dict_mu}, depth_limit=1),
                                )
                            elif region == "control-zgamma":
                                output_array["weight"] = (
                                    ak.ones_like(events.run)
                                    if isRealData
                                    else weights_dict_gamma["weight"]
                                )
                                skim(
                                    region,
                                    ak.zip(
                                        {**energy_var_array, **weights_dict_gamma}, depth_limit=1
                                    ),
                                )

        toc = time.time()
        output["filltime"] = toc - tic
        print(f"Time to fill histograms: {toc - tic:.2f} seconds")
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        pass
