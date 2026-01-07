from __future__ import annotations

DATA_SAMPLES = ["JetMET", "Muon", "ParkingHH", "ParkingSingleMuon", "ParkingVBF"]

LUMI = {
    "2022": 7980.5,
    "2022EE": 26671.6,
    "2023": 18084.4,
    "2023BPix": 9692.1,
    "2022-2023": 62428.6,
    "2024": 0
}

norm_preserving_weights = ["genweight", "pileup", "ISRPartonShower", "FSRPartonShower"]

data_key = "data"
