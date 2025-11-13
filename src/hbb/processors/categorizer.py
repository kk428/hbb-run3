from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import numpy as np
import pandas as pd
from coffea.analysis_tools import PackedSelection, Weights
from coffea.ml_tools import xgboost_wrapper
from dask_awkward import to_dataframe
from hist.dask import Hist

from hbb.corrections import (
    add_pileup_weight,
    get_jetveto_event,
    lumiMasks,
)
from hbb.processors.SkimmerABC import SkimmerABC

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


class categorizer(SkimmerABC):
    def __init__(
        self,
        year="2022",
        xsecs: dict = None,
        systematics=False,
        save_skim=False,
        skim_outpath="",
        evaluate_BDT=True,
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        self._year = year
        self._systematics = systematics
        self._save_skim = save_skim
        self._skim_outpath = skim_outpath
        self._evaluate_BDT = evaluate_BDT

        with Path("src/hbb/muon_triggers.json").open() as f:
            self._muontriggers = json.load(f)

        with Path("src/hbb/egamma_triggers.json").open() as f:
            self._egammatriggers = json.load(f)

        with Path("src/hbb/triggers.json").open() as f:
            self._triggers = json.load(f)

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with Path("src/hbb/metfilters.json").open() as f:
            self._met_filters = json.load(f)

        with Path("src/hbb/taggers.json").open() as f:
            self._b_taggers = json.load(f)

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

    def process(self, events):
        # fatjets = events.FatJet
        # jets = events.Jet
        # met = events.MET
        # shifts = [({"Jet": jets, "FatJet": fatjets, "MET": met}, None)]
        # TODO return processor.accumulate(self.process_shift(update(events, collections), name) for collections, name in shifts)
        return self.process_shift(events, None)

    def add_weights(
        self,
        weights,
        events,
        dataset,
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights.add("genweight", events.genWeight)

        add_pileup_weight(weights, self._year, events.Pileup.nPU)

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

        return weights_dict, totals_dict

    def process_shift(self, events, shift_name):

        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        output = self.make_output()
        weights = Weights(None, storeIndividual=True)
        if shift_name is None and not isRealData:
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

        fatjets = set_ak8jets(events.FatJet)
        goodfatjets = good_ak8jets(fatjets)
        jets = set_ak4jets(events.Jet)
        goodjets = good_ak4jets(jets)

        cut_jetveto = get_jetveto_event(jets, self._year)
        selection.add("ak4jetveto", cut_jetveto)

        selection.add("2FJ", ak.num(goodfatjets, axis=1) == 2)
        selection.add("not2FJ", ak.num(goodfatjets, axis=1) != 2)

        xbbfatjets = goodfatjets[ak.argsort(goodfatjets.pnetXbbXcc, axis=1, ascending=False)]

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

        btag_cut = self._b_taggers[self._year]["AK4"]["Jet_btagPNetB"]["M"]
        selection.add(
            "antiak4btagMediumOppHem",
            ak.max(ak4_opphem_ak8.btagPNetB, axis=1, mask_identity=False) < btag_cut,
        )
        selection.add(
            "antiak4btagMedium",
            ak.max(ak4_outside_ak8.btagPNetB, axis=1, mask_identity=False) < btag_cut,
        )
        selection.add(
            "ak4btagMedium08",
            ak.max(ak4_outside_ak8.btagPNetB, axis=1, mask_identity=False) > btag_cut,
        )

        met = events.MET
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

        goodmuon = good_muons(events.Muon)
        nmuons = ak.num(goodmuon, axis=1)
        leadingmuon = ak.firsts(goodmuon)

        goodelectron = good_electrons(events.Electron)
        nelectrons = ak.num(goodelectron, axis=1)

        selection.add("noleptons", (nmuons == 0) & (nelectrons == 0))
        selection.add("onemuon", (nmuons == 1) & (nelectrons == 0))
        selection.add("muonkin", (leadingmuon.pt > 55.0) & (abs(leadingmuon.eta) < 2.1))
        selection.add("muonDphiAK8", abs(leadingmuon.delta_phi(candidatejet)) > 2 * np.pi / 3)

        goodphotons = good_photons(events.Photon)
        nphotons = ak.num(goodphotons, axis=1)
        leadingphoton = ak.firsts(goodphotons)

        selection.add("onephoton", (nphotons == 1))
        selection.add("atleastonephoton", (nphotons >= 1))
        selection.add("passphotonveto", (nphotons == 0))

        gen_variables = {}
        if isRealData:
            genflavor = ak.zeros_like(candidatejet.pt)
            genBosonPt = ak.zeros_like(candidatejet.pt)
        else:
            weights_dict, totals_temp = self.add_weights(
                weights,
                events,
                dataset,
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
                "ak4jetveto",
                "minjetkin",
                "antiak4btagMediumOppHem",
                "lowmet",
                "noleptons",
            ],
            "signal-ggf": [
                "trigger",
                "lumimask",
                "metfilter",
                "ak4jetveto",
                "minjetkin",
                "antiak4btagMediumOppHem",
                "lowmet",
                "noleptons",
                "notvbf",
                "not2FJ",
            ],
            "signal-vh": [
                "trigger",
                "lumimask",
                "metfilter",
                "ak4jetveto",
                "minjetkin",
                "antiak4btagMediumOppHem",
                "lowmet",
                "noleptons",
                "notvbf",
                "2FJ",
            ],
            "signal-vbf": [
                "trigger",
                "lumimask",
                "metfilter",
                "ak4jetveto",
                "minjetkin",
                "antiak4btagMediumOppHem",
                "lowmet",
                "noleptons",
                "isvbf",
            ],
            "control-tt": [
                "muontrigger",
                "lumimask",
                "metfilter",
                "ak4jetveto",
                "minjetkin",
                "ak4btagMedium08",
                "onemuon",
                "muonkin",
                "muonDphiAK8",
            ],
            "control-zgamma": [
                "egammatrigger",
                "lumimask",
                "metfilter",
                "minjetkin_zgamma",
                "atleastonephoton",
                "antiak4btagMedium",
            ],
        }

        def normalize(val, cut):
            if cut is None:
                ar = ak.fill_none(val, np.nan)
                return ar
            else:
                ar = ak.fill_none(val[cut], np.nan)
                return ar

        tic = time.time()

        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]

        nominal_weight = ak.ones_like(events.run) if isRealData else weights_dict["weight"]
        gen_weight = ak.ones_like(events.run) if isRealData else events.genWeight

        output_array = None
        if self._save_skim:
            # define "flat" output array
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
                "FatJet0_pnetMass": candidatejet.pnetmass,
                "FatJet0_pnetTXbb": candidatejet.particleNet_XbbVsQCD,
                "FatJet0_pnetTXcc": candidatejet.particleNet_XccVsQCD,
                "FatJet0_pnetTXqq": candidatejet.particleNet_XqqVsQCD,
                "FatJet0_pnetTXgg": candidatejet.particleNet_XggVsQCD,
                "FatJet0_pnetTQCD": candidatejet.particleNet_QCD,
                "FatJet0_pnetXbbXcc": candidatejet.pnetXbbXcc,
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
                "Photon0_pt": leadingphoton.pt,
                "MET": met,
                "weight": nominal_weight,
                "genWeight": gen_weight,
                **gen_variables,
            }

            # extra variables for big array
            output_array_extra = {
                # AK4 Jets away from FatJet0
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
                skim_path = f"{self._skim_outpath}/{self._year}/{dataset}/parquet/{region}"
            else:
                skim_path = Path(self._skim_outpath) / self._year / dataset / "parquet" / region
                skim_path.mkdir(parents=True, exist_ok=True)
            print("Saving skim to: ", skim_path)

            if self._evaluate_BDT and region == "signal-all":
                # Load pre-trained BDT model
                # model = xgb.XGBClassifier()
                # model.load_model('src/hbb/MultiClassBDT_23Oct25.json')
                # model = xgboost_wrapper('src/hbb/MultiBDT_coffea_3Nov25.json')

                # Prepare features for BDT evaluation
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
                        ret = ak.concatenate(
                            [events[name][:, np.newaxis] for name in bdt_features], axis=1
                        )
                        return [], {"data": ret}

                model = xgboost_model(Path.cwd() / "MultiClassBDT_23Oct25.ubj")

                # Prepare the BDT input array
                bdt_dask_arrays = {
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
                    "Photon0_pt": leadingphoton.pt,
                    # AK4 Jets away from FatJet0
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
                    # AK4 Jet away but closest to FatJet0
                    "JetClosestFatJet0_pt": ak4_closest_ak8.pt,
                    "JetClosestFatJet0_eta": ak4_closest_ak8.eta,
                    "JetClosestFatJet0_phi": ak4_closest_ak8.phi,
                    "JetClosestFatJet0_mass": ak4_closest_ak8.mass,
                }
                bdt_dask_arrays = {
                    key: to_dataframe(value) for key, value in bdt_dask_arrays.items()
                }
                bdt_input = pd.DataFrame()
                for _key, value in bdt_dask_arrays.items():
                    bdt_input = pd.concat([bdt_input, pd.DataFrame(value)], axis=1)
                bdt_input.columns = bdt_features

                # Evaluate BDT scores and append to output array
                bdt_scores = model(bdt_input)
                output_array["BDT_score"] = dak.from_awkward(
                    ak.from_numpy(bdt_scores), npartitions=1
                )

            # possible TODO: add systematic weights?
            output["skim"][region] = dak.to_parquet(
                output_array[cut],
                str(skim_path),
                compute=False,
            )

            allcuts = set()
            cut = selection.all(*allcuts)
            output["cutflow"].fill(
                dataset=dataset,
                region=region,
                genflavor=normalize(genflavor, None),
                cut=0,
                weight=nominal_weight,
            )
            for i, cut in enumerate(selections):
                allcuts.add(cut)
                cut = selection.all(*allcuts)  # noqa: PLW2901
                output["cutflow"].fill(
                    dataset=dataset,
                    region=region,
                    genflavor=normalize(genflavor, cut),
                    cut=i + 1,
                    weight=nominal_weight[cut],
                )

        for region in regions:
            if self._save_skim:
                print(region)
                if region == "signal-all":
                    skim(region, ak.zip({**output_array, **output_array_extra}, depth_limit=1))
                else:
                    skim(region, ak.zip(output_array, depth_limit=1))
            for systematic in systematics:
                if isRealData and systematic is not None:
                    continue

        toc = time.time()
        output["filltime"] = toc - tic
        print(f"Time to fill histograms: {toc - tic:.2f} seconds")
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        pass
