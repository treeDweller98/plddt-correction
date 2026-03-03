import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import ConfusionMatrixDisplay


def __plot_hexbin_regression(ax, y_true, y_pred, xlabel, ylabel, title, gridsize=175, cmap="inferno", add_colorbar=True):
    h = ax.hexbin(y_true, y_pred, gridsize=gridsize, cmap=cmap, norm=LogNorm())
    if add_colorbar:
        plt.colorbar(h, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    return h


def __plot_confmat_classification(ax, y_true, y_pred, labels, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax)


def __plot_feature_importance(ax, feature_names, feature_importances):
    def aggregate_feature_importance(feature_names, feature_importances):
        def group_feature_name(name: str) -> str:
            if name.startswith("af2_max_flag"):
                return "af2_max_flag"
            if name.startswith("residue_"):
                return "residue"
            if name.startswith("nbr_sasa"):
                return "nbr_sasa"
            if name.startswith("nbr_depth"):
                return "nbr_depth"
            if name.startswith("nbr_s"):
                return "nbr"
            return name

        df = pd.DataFrame({"name": feature_names, "importance": feature_importances})
        df["group"] = df["name"].apply(group_feature_name)
        grouped = df.groupby("group")["importance"].sum().sort_values(ascending=True)
        return grouped

    grouped_importance = aggregate_feature_importance(feature_names, feature_importances)
    grouped_importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Total Gini importance")
    ax.set_title("Feature importance by group")


def plot_regression_metrics(y_true, y_pred, y_af2, feature_importances, feature_names):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # LDDT Plots
    __plot_hexbin_regression(ax1, y_true, y_af2, "True LDDT", "AF2 pLDDT", "True vs AF2 pLDDT")
    __plot_hexbin_regression(ax2, y_true, y_pred, "True LDDT", "Model LDDT", "True vs Model Prediction")

    # Feature importance
    __plot_feature_importance(ax3, feature_names, feature_importances)

    fig.tight_layout()
    plt.show()


def plot_classification_metrics(y_true, y_pred, y_af2, feature_importances, feature_names):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Confusion Matrices
    labels = ["helix", "beta", "coil"]
    __plot_confmat_classification(ax1, y_true, y_af2, labels, "True Label", "AF2 Pred", "NMR SS flag vs AF2 Pred")
    __plot_confmat_classification(ax2, y_true, y_pred, labels, "True Label", "Model Pred", "NMR SS flag vs Model Pred")

    # Feature importance
    __plot_feature_importance(ax3, feature_names, feature_importances)

    fig.tight_layout()
    plt.show()
