import json
import pprint
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from plotters import plot_classification_metrics, plot_regression_metrics

SEED = 42
PDB_COL = "af2_pdb"


def make_feature_names(categ_cols, numer_cols, df) -> list[str]:
    names = []
    for c in categ_cols:
        dim = len(df[c].iloc[0])
        names.extend([f"{c}_{i}" for i in range(dim)])
    names.extend(numer_cols)
    return names


def get_kfold_splits(df: pd.DataFrame, split_by_protein: bool, n_splits: int):
    splitter = (
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        if split_by_protein
        else KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    )
    groups = df[PDB_COL] if split_by_protein else None
    for train_idx, test_idx in splitter.split(df, groups=groups):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        yield train_df, test_df


def make_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categ_cols: list[str],
    numer_cols: list[str],
    target_col: str,
    af2_pred_col: str,
):
    def make_feature_matrix(df: pd.DataFrame):
        ohe = np.hstack([np.stack(df[c].values) for c in categ_cols])
        num = df[numer_cols].to_numpy()

        X = np.hstack((ohe, num))
        y = df[target_col].to_numpy()
        y_af2 = df[af2_pred_col].to_numpy()
        return X, y, y_af2

    X_train, y_train, y_af2_train = make_feature_matrix(train_df)
    X_test, y_test, y_af2_test = make_feature_matrix(test_df)

    print(f"{X_train.shape=}, {X_test.shape=}, train:test proteins = {train_df[PDB_COL].nunique()}:{test_df[PDB_COL].nunique()}")

    return X_train, X_test, y_train, y_test, y_af2_train, y_af2_test


def calculate_regression_metrics(y_test, y_pred, y_af2):
    metrics: dict[str, float] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": pow(mean_squared_error(y_test, y_pred), 0.5),
        "R^2": r2_score(y_test, y_pred),
        "AF2_MAE": mean_absolute_error(y_test, y_af2),
        "AF2_RMSE": pow(mean_squared_error(y_test, y_af2), 0.5),
        "AF2_R^2": r2_score(y_test, y_af2),
    }
    return metrics


def calculate_classification_metrics(y_test, y_pred, y_af2) -> dict[str, float]:
    af2_report = classification_report(y_test, y_af2, digits=5, output_dict=True)
    model_report = classification_report(y_test, y_pred, digits=5, output_dict=True)
    metrics = {
        "f1-score": model_report["macro avg"]["f1-score"],
        "af2 f1-score": af2_report["macro avg"]["f1-score"],
        "precision": model_report["macro avg"]["precision"],
        "af2 precision": af2_report["macro avg"]["precision"],
        "recall": model_report["macro avg"]["recall"],
        "af2 recall": af2_report["macro avg"]["recall"],
    }
    return metrics


def aggregate_kfold_metrics(kfold_metrics: list[dict[str, float]]):
    metric_names = kfold_metrics[0].keys()
    aggregated = {}
    for metric in metric_names:
        values = np.array([fold[metric] for fold in kfold_metrics])
        aggregated[f"{metric}_mean"] = float(values.mean())
        aggregated[f"{metric}_std"] = float(values.std(ddof=1))  # sample std
    return aggregated


def save_output(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: RegressorMixin | ClassifierMixin,
    y_test: Sequence[float],
    y_af2_test: Sequence[float],
    y_pred: Sequence[float],
    metrics: dict[str, float],
    cfg: dict[str, Any],
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)

    test_df["y_pred"] = y_pred

    train_df.to_json(outdir / "train_df.json", orient="records")
    test_df.to_json(outdir / "test_df.json", orient="records")

    with open(outdir / "model.pkl", "wb") as f:
        print("Skipping model saving.")
        # pickle.dump(model, f, protocol=5)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(outdir / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=4)


def train_model(
    model_cls: type[RegressorMixin | ClassifierMixin],
    model_init_params: dict[str, Any],
    feature_df: pd.DataFrame,
    categ_cols: list[str],
    numer_cols: list[str],
    target_col: str,
    af2_pred_col: str,
    n_splits: int,
    split_by_protein: bool,
    outdir: Path,
    is_debug: bool = False,
):
    cfg = {k: str(v) for k, v in locals().items()}
    feature_names = make_feature_names(categ_cols, numer_cols, feature_df)
    kfold_metrics = []

    if is_debug:
        print("DEBUG MODE ENABLED")
        feature_df = feature_df.head(1_000)

    if is_classifier(model_cls()):
        calculate_metrics = calculate_classification_metrics
        plot_graphs = plot_classification_metrics
    elif is_regressor(model_cls()):
        calculate_metrics = calculate_regression_metrics
        plot_graphs = plot_regression_metrics
    else:
        raise ValueError("sklearn regressors and classifiers only")

    for i, (train_df, test_df) in enumerate(get_kfold_splits(feature_df, split_by_protein, n_splits)):
        tic = time.perf_counter()

        X_train, X_test, y_train, y_test, y_af2_train, y_af2_test = make_features(
            train_df, test_df, categ_cols, numer_cols, target_col, af2_pred_col
        )

        model = model_cls(**model_init_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        feature_importances = model.feature_importances_

        metrics = calculate_metrics(y_test, y_pred, y_af2_test)
        save_output(train_df, test_df, model, y_test, y_af2_test, y_pred, metrics, cfg, outdir / f"fold_{i}")
        kfold_metrics.append(metrics)

        pprint.pp(metrics)
        plot_graphs(y_test, y_pred, y_af2_test, feature_importances, feature_names)

        toc = time.perf_counter()
        print(f"Finished Split {i + 1} in {toc - tic: 0.4f}", "=" * 25, "\n", sep="\n")

    print("Training Complete")
    pprint.pp(aggregate_kfold_metrics(kfold_metrics))
