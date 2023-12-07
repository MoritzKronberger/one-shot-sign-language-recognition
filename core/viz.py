"""Visualization functionalities."""

from keras.callbacks import History
from sklearn import metrics
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import colorcet as cc


def visualize_history(
    history: History,
    metrics: list[str],
    figsize: tuple[int, int] = (12, 8)
):
    """Visualize model history."""
    num_cols = len(metrics)

    plt.figure(figsize=figsize)

    for i, metric in enumerate(metrics):
        plt.subplot(2, num_cols, i+1)
        plt.plot(history.history[metric], label=f"train {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")
        plt.legend()

    plt.title("Training History")

    plt.show()
    plt.close()


def visualize_confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    pretty_labels: npt.ArrayLike
):
    """Visualize confusion matrix."""
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(
        pd.DataFrame(
            conf_matrix,
            index=pretty_labels,
            columns=pretty_labels
        ),
        annot=True,
        cmap="YlGnBu",
        fmt='d'
    )
    plt.tight_layout()
    plt.title("Confusion Matrix", y=1.1)
    plt.ylabel("Predicted")
    plt.xlabel("Ground Truth")
    plt.show()
    plt.close()


def visualize_embeddings(
    embeddings_2d: npt.NDArray[np.float32],
    embedding_labels: npt.ArrayLike,
    label_map: dict[str, int],
    figsize: tuple[int, int] = (12, 8)
):
    """Visualize 2-dimensional embeddings in scatter plot."""
    # Create a custom color map for each label (= letter)
    # References:
    # - https://stackoverflow.com/a/69688664
    # - https://stackoverflow.com/a/37902765
    # - https://stackoverflow.com/a/32771206
    num_classes = len(label_map)
    sns_palette = sns.color_palette(cc.glasbey, n_colors=num_classes)
    custom_cmap = ListedColormap(sns_palette.as_hex(), N=num_classes)
    norm = plt.Normalize(0, num_classes-1)

    # Visualize embeddings in scatter plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        lw=0,
        s=20,
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
            markersize=15
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

    plt.show()
    plt.close()
