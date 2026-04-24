from micro_ml.scripts.train import build_model


class Training:
    def __init__(self, cfg):
        self.cfg = cfg.training

    def fire(self):
        if self.cfg.pipeline == "enable":
            print("Training Pipeline Selected...")
            build_model(cfg=self.cfg)
        else:
            print("Training Pipeline Ignored...")
