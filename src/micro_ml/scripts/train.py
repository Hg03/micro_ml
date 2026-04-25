from sklearn.ensemble import HistGradientBoostingClassifier
from micro_ml.scripts.utils import load_assets, save_assets
from huggingface_hub import HfApi
from omegaconf import DictConfig
from dotenv import load_dotenv
import polars as pl
import os

load_dotenv()


def build_model(cfg: DictConfig) -> None:
    X_trainT, y_trainT = load_assets(
        paths=[
            cfg.paths.X_trainT,
            cfg.paths.y_trainT,
        ]
    )
    sklearn_model = HistGradientBoostingClassifier()
    sklearn_model.fit(X=X_trainT, y=y_trainT.select(cfg.columns.target).to_series())
    save_assets({cfg.paths.sklearn_model: sklearn_model})


def make_predictions(cfg: DictConfig) -> None:
    X_trainT, X_testT, sklearn_model = load_assets(
        paths=[cfg.paths.X_trainT, cfg.paths.X_testT, cfg.paths.sklearn_model]
    )
    train_predictions = sklearn_model.predict(X=X_trainT)
    test_predictions = sklearn_model.predict(X=X_testT)
    save_assets(
        {
            cfg.paths.train_predictions: pl.DataFrame(
                {cfg.columns.target: train_predictions}
            ),
            cfg.paths.test_predictions: pl.DataFrame(
                {cfg.columns.target: test_predictions}
            ),
        }
    )


def upload_to_huggingface(cfg: DictConfig):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        repo_id=cfg.huggingface.repo_id,
        repo_type=cfg.huggingface.repo_type,
        path_or_fileobj=cfg.paths.sklearn_model,
        path_in_repo=cfg.huggingface.model_name,
    )
    api.upload_file(
        repo_id=cfg.huggingface.repo_id,
        repo_type=cfg.huggingface.repo_type,
        path_or_fileobj=cfg.paths.preprocessing_model,
        path_in_repo=cfg.huggingface.preprocessor_name,
    )
    print(f"Artifacts uploaded to {cfg.huggingface.repo_id}")
