from sklearn.ensemble import HistGradientBoostingClassifier
from micro_ml.scripts.utils import load_assets, save_assets
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> None:
    X_trainT, X_testT, y_trainT, y_testT = load_assets(
        paths=[
            cfg.paths.X_trainT,
            cfg.paths.X_testT,
            cfg.paths.y_trainT,
            cfg.paths.y_testT,
        ]
    )
    sklearn_model = HistGradientBoostingClassifier()
    sklearn_model.fit(X=X_trainT, y=y_trainT.select(cfg.columns.target).to_series())
    # train_predictions = sklearn_model.predict(X=X_trainT)
    # test_predictions = sklearn_model.predict(X=X_testT)
    save_assets({cfg.paths.sklearn_model: sklearn_model})
