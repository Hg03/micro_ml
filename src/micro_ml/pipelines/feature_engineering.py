from micro_ml.scripts.load import split_data
from micro_ml.scripts.transform import build_features


class FeatureEngineering:
    def __init__(self, cfg):
        self.cfg = cfg.feature

    def fire(self):
        if self.cfg.pipeline == "enable":
            print("Feature Engineering Pipeline Selected...")
            split_data(cfg=self.cfg)
            build_features(cfg=self.cfg)
        else:
            print("Feature Engineering Pipeline Ignored...")
