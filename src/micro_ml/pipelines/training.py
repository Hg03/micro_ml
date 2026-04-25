from micro_ml.scripts.train import build_model, make_predictions
from micro_ml.scripts.evaluate import get_metrics


class Training:
    def __init__(self, cfg):
        self.cfg = cfg.training

    def fire(self):
        if self.cfg.pipeline == "enable":
            print("Training Pipeline Selected...")
            build_model(cfg=self.cfg)
            make_predictions(cfg=self.cfg)
            get_metrics(cfg=self.cfg)
        else:
            print("Training Pipeline Ignored...")
