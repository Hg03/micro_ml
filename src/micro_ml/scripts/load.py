from micro_ml.scripts.utils import load_assets, save_assets
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import polars as pl


def fix_target(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
    df = df.with_columns(
        pl.col(cfg.columns.target).replace(cfg.columns.target_encodings)
    )
    return df.drop_nulls(subset=[cfg.columns.target])


def split_data(cfg: DictConfig) -> None:
    df = fix_target(load_assets(paths=[cfg.paths.raw]), cfg)
    FeatureFrame = df.select(cfg.columns.features)
    TargetFrame = df.select(cfg.columns.target)
    X_train, X_test, y_train, y_test = train_test_split(
        FeatureFrame, TargetFrame, test_size=cfg.test_size
    )
    save_assets(
        {
            cfg.paths.encoded: df,
            cfg.paths.X_train: X_train,
            cfg.paths.X_test: X_test,
            cfg.paths.y_train: y_train,
            cfg.paths.y_test: y_test,
        }
    )
