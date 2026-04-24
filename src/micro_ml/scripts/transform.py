from micro_ml.scripts.utils import load_assets, save_assets
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from omegaconf import DictConfig


def build_features(cfg: DictConfig) -> None:
    X_train, X_test, y_train, y_test = load_assets(
        [cfg.paths.X_train, cfg.paths.X_test, cfg.paths.y_train, cfg.paths.y_test]
    )
    imputer = SimpleImputer(strategy=cfg.strategy.impute)
    preprocessor = Pipeline(steps=[("imputer", imputer)]).set_output(transform="polars")
    X_trainT = preprocessor.fit_transform(X_train)
    X_testT = preprocessor.transform(X_test)
    save_assets(
        {
            cfg.paths.X_trainT: X_trainT,
            cfg.paths.X_testT: X_testT,
            cfg.paths.y_trainT: y_train,
            cfg.paths.y_testT: y_test,
            cfg.paths.preprocessing_model: preprocessor,
        }
    )
