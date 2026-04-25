from sklearn.metrics import precision_score, recall_score
from micro_ml.scripts.utils import load_assets
from omegaconf import DictConfig


def get_metrics(cfg: DictConfig) -> None:
    X_train, X_test, y_train, y_test, train_predictions, test_predictions = load_assets(
        paths=[
            cfg.paths.X_trainT,
            cfg.paths.X_testT,
            cfg.paths.y_trainT,
            cfg.paths.y_testT,
            cfg.paths.train_predictions,
            cfg.paths.test_predictions,
        ]
    )
    precision = precision_score(y_train, train_predictions, average="weighted")
    recall = recall_score(y_test, test_predictions, average="weighted")
    metrics_ = {"Precision Score": precision, "Recall Score": recall}
    print(metrics_)
