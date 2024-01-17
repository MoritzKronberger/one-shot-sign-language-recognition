"""Visualization functionalities."""

from keras.callbacks import History
from sklearn import metrics
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, NullLocator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import colorcet as cc

# Use Times New Roman for IEEE
# Reference: https://stackoverflow.com/a/40734893
plt.rcParams["font.family"] = "Times New Roman"


def visualize_history(
    history: History,
    metrics: list[str],
    loss_name: str,
    filepath: str | None,
    figsize: tuple[int, int] = (5, 3),
    log_yscale: bool = False,
) -> None:
    """Visualize model history."""
    num_rows = len(metrics)
    colors = [
        (0, 0.12, 0.35),
        (1, 0.01, 0.31),
    ]

    plt.figure(figsize=figsize)

    for i, metric in enumerate(metrics):
        metric_name = (loss_name if metric == "loss"
                       else metric)
        plt.subplot(num_rows, 1, i+1)
        plt.plot(
            history.history[metric],
            label=f"{metric_name} (training)",
            color=colors[0]
        )
        plt.plot(
            history.history[f"val_{metric}"],
            label=f"{metric_name} (validation)",
            color=colors[1]
        )
        plt.legend()
        plt.xlabel("Epoch")
        if log_yscale:
            # Set log scale
            ax = plt.gca()
            ax.set_yscale('log')
            # Remove scientific label formatting for consistency
            # https://stackoverflow.com/a/29188910
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
    plt.close()


def visualize_confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    pretty_labels: npt.ArrayLike,
    filepath: str | None,
    figsize: tuple[int, int] = (5, 5)
) -> None:
    """Visualize confusion matrix."""
    plt.figure(figsize=figsize)

    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(
        pd.DataFrame(  # type: ignore
            conf_matrix,
            index=pretty_labels,
            columns=pretty_labels
        ),
        annot=True,
        cmap="Blues",
        fmt='d',
        cbar=False
    )

    plt.tight_layout()
    plt.ylabel("Predicted")
    plt.xlabel("Ground Truth")

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
    plt.close()


def visualize_embeddings(
    embeddings_2d: npt.NDArray[np.float32],
    embedding_labels: npt.ArrayLike,
    label_map: dict[str, int],
    filepath: str | None,
    figsize: tuple[int, int] = (7, 4)
) -> None:
    """Visualize 2-dimensional embeddings in scatter plot."""
    # Create a custom color map for each label (= letter)
    # References:
    # - https://stackoverflow.com/a/69688664
    # - https://stackoverflow.com/a/37902765
    # - https://stackoverflow.com/a/32771206
    num_classes = len(label_map)
    sns_palette = sns.color_palette(cc.glasbey, n_colors=num_classes)
    custom_cmap = ListedColormap(sns_palette.as_hex(), N=num_classes)
    norm = plt.Normalize(0, num_classes-1)  # type: ignore

    # Visualize embeddings in scatter plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        lw=0,
        s=8,
        c=embedding_labels,
        cmap=custom_cmap,
        norm=norm
    )

    # Add custom legend
    unique_embedding_labels = np.unique(embedding_labels)
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=str_lbl.upper(),
            markerfacecolor=custom_cmap(int_lbl),
            markersize=12
        )
        for str_lbl, int_lbl
        in label_map.items()
        if int_lbl in unique_embedding_labels
    ]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.08, 1),
        borderaxespad=0
    )

    # Hide ticks (since numbers are meaningless)
    ax.set_xticks([])
    ax.set_yticks([])

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
    plt.close()
