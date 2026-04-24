from micro_ml.pipelines.feature_engineering import FeatureEngineering
from micro_ml.pipelines.training import Training
from omegaconf import DictConfig
import hydra


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    FeatureEngineering(cfg=cfg).fire()
    Training(cfg).fire()


if __name__ == "__main__":
    main()
