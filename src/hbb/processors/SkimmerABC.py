"""
Skimmer Base Class - common functions for all skimmers.
Author(s): Raghav Kansal
"""

from __future__ import annotations

import logging
from abc import abstractmethod

from coffea import processor
import awkward as ak
import numpy as np

from hbb.common_vars import LUMI

logging.basicConfig(level=logging.INFO)


class SkimmerABC(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
    """

    XSECS = None

    def get_dataset_norm(self, year, dataset):
        """
        Cross section * luminosity normalization for a given dataset and year.
        This still needs to be normalized with the acceptance of the pre-selection in post-processing.
        (Done in postprocessing/utils.py:load_samples())
        """
        if dataset in self.XSECS:
            xsec = self.XSECS[dataset]
            weight_norm = xsec * LUMI[year]
            logging.info(f"XSEC: {xsec}, LUMI: {LUMI[year]}")
        else:
            logging.warning(f"Dataset name: {dataset} not found in xsecs.py")
            logging.warning("Weight not normalized to cross section")
            weight_norm = 1

        print("weight_norm", weight_norm)

        return weight_norm

    def normalize(self, val, cut):
        """
        Fills dak.array nones with nan and applies selection cut
        Used for filling hist.Hist, 
        where the events with nones in the object collections throw errors on fill during compute
        """
        if cut is None:
            ar = ak.fill_none(val, np.nan)
            return ar
        else:
            ar = ak.fill_none(val[cut], np.nan)
            return ar

    @abstractmethod
    def add_weights(self) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
