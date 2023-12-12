"""Abstract loading of "German Sign Language (DGS) Alphabet" dataset.

Kaggle:
    https://www.kaggle.com/datasets/moritzkronberger/german-sign-language
"""

from dataclasses import dataclass
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import kaggle
import pandas as pd
import numpy as np
import numpy.typing as npt

NumericLabelMap = dict[str, int]


@dataclass
class CategoricalNumericDataset:
    """Categorical dataset with numeric integer labels."""
    x_train: npt.NDArray[np.float32]
    y_train: npt.NDArray[int]
    x_test: npt.NDArray[np.float32]
    y_test: npt.NDArray[int]
    label_map: NumericLabelMap


@dataclass
class CategoricalOneHotDataset:
    """Categorical dataset with one hot labels."""
    x_train: npt.NDArray[np.float32]
    y_train: npt.NDArray[int]
    x_test: npt.NDArray[np.float32]
    y_test: npt.NDArray[int]
    label_map: NumericLabelMap


@dataclass
class PairDataset:
    """Pair dataset with distance labels."""
    x_train_1: npt.NDArray[np.float32]
    x_train_2: npt.NDArray[np.float32]
    y_train: npt.NDArray[np.float32]
    x_test_1: npt.NDArray[np.float32]
    x_test_2: npt.NDArray[np.float32]
    y_test: npt.NDArray[np.float32]


class DGSAlphabet:
    """Wrapper for "German Sign Language (DGS) Alphabet" dataset.

    Kaggle:
        https://www.kaggle.com/datasets/moritzkronberger/german-sign-language
    """

    kaggle_dataset_id: str
    local_dataset_dir: str
    dataset_filename: str
    dataset_filetype: str

    def __init__(
        self,
        kaggle_dataset_id: str = "moritzkronberger/german-sign-language",
        local_dataset_dir: str = "./dataset",
        dataset_filename="german_sign_language",
        dataset_filetype="csv"
    ):
        """Create new dataset wrapper."""
        self.kaggle_dataset_id = kaggle_dataset_id
        self.local_dataset_dir = local_dataset_dir
        self.dataset_filename = dataset_filename
        self.dataset_filetype = dataset_filetype

    def __kaggle_download(self) -> None:
        """Download dataset from kaggle.

        Reference:
            https://www.kaggle.com/docs/api
        """
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            self.kaggle_dataset_id,
            path=self.local_dataset_dir,
            unzip=True
        )

    def __str_labels_to_int(
        self,
        labels: npt.NDArray[np.str_]
    ) -> tuple[npt.NDArray[int], NumericLabelMap]:
        """Convert string labels to integers using map."""
        unique_labels = np.unique(labels)
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        return np.array([label_map[lbl] for lbl in labels]), label_map

    def __make_pairs(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[int],
        margin: float = 1
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Pair every sample with a random matching and non-matching sample.

        New labels are distance between pairs:
            - Matching: 0
            - Non-matching: `margin`

        Reference:
            https://keras.io/examples/vision/siamese_contrastive
        """
        unique_labels = np.unique(y)
        sign_indices = {i: np.where(y == i)[0] for i in unique_labels}

        pairs: list[tuple[npt.NDArray[np.float32],
                          npt.NDArray[np.float32]]] = []
        labels: list[float] = []

        for idx1 in range(len(x)):
            # Add a matching example
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(sign_indices[label1])
            x2 = x[idx2]
            assert y[idx2] == y[idx1]

            pairs.append((x1, x2))
            labels.append(0.0)

            # Add a non-matching example
            label2 = random.choice(unique_labels)
            while label2 == label1:
                label2 = random.choice(unique_labels)

            idx2 = random.choice(sign_indices[label2])
            x2 = x[idx2]
            assert y[idx2] != y[idx1]

            pairs.append((x1, x2))
            labels.append(margin)

        return np.array(pairs), np.array(labels).astype("float32")

    def load_dataframe(self) -> pd.DataFrame:
        """Load dataset as pandas DatFrame."""
        filepath = f"{self.local_dataset_dir}/{self.dataset_filename}.{self.dataset_filetype}"
        # Download dataset if not exists
        if not os.path.isfile(filepath):
            self.__kaggle_download()
        return pd.read_csv(filepath)

    def load_categorical_numeric(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        landmark_dims: int = 3,
        exclusive_test_labels: list[str] | None = None,
        shuffle_exclusive_test_data: bool = False
    ) -> CategoricalNumericDataset:
        """Load dataset in categorical format.

        (Numeric integer labels)
        """
        df = self.load_dataframe().copy()

        # Regular train test split
        if exclusive_test_labels is None:
            # Convert to numeric labels
            y = df.label.to_numpy()
            y, label_map = self.__str_labels_to_int(y)

            # Reshape data to 3D landmarks
            x = df.drop(['label'], axis=1)
            x = x.to_numpy()
            x = x.reshape((x.shape[0], -1, landmark_dims))

            # Split into train and test data
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                shuffle=True,
                random_state=random_state
            )
        # Use specified labels for test data
        else:
            # Create mask for train labels (on string labels)
            test_mask = df.label.isin(exclusive_test_labels)

            # Convert to numeric labels
            y = df.label.array
            y, label_map = self.__str_labels_to_int(y)
            df["label"] = y

            # Get train data
            train_df = df[~test_mask]
            y_train = train_df.label.to_numpy()
            x_train = train_df.drop(['label'], axis=1)
            x_train = x_train.to_numpy()
            x_train = x_train.reshape((x_train.shape[0], -1, landmark_dims))

            # Get test data
            test_df = df[test_mask]
            y_test = test_df.label.to_numpy()
            x_test = test_df.drop(['label'], axis=1)
            x_test = x_test.to_numpy()
            x_test = x_test.reshape((x_test.shape[0], -1, landmark_dims))

            # Shuffle train data
            x_train, y_train = shuffle(x_train, y_train)

            # Only shuffle test data if specified
            if shuffle_exclusive_test_data:
                x_test, y_test = shuffle(x_test, y_test)

        return CategoricalNumericDataset(
            x_train,
            y_train,
            x_test,
            y_test,
            label_map
        )

    def load_categorical_one_hot(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        landmark_dims: int = 3,
        exclusive_test_labels: list[str] | None = None,
        shuffle_exclusive_test_data: bool = False
    ):
        """Load dataset in categorical format.

        (One hot labels)
        """
        numeric_dataset = self.load_categorical_numeric(
            test_size,
            random_state,
            landmark_dims,
            exclusive_test_labels,
            shuffle_exclusive_test_data
        )

        y_train_one_hot = to_categorical(numeric_dataset.y_train)
        y_test_one_hot = to_categorical(numeric_dataset.y_test)

        return CategoricalOneHotDataset(
            numeric_dataset.x_train,
            y_train_one_hot,
            numeric_dataset.x_test,
            y_test_one_hot,
            numeric_dataset.label_map
        )

    def load_pairs(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        landmark_dims: int = 3,
        margin: float = 1,
        exclusive_test_labels: list[str] | None = None,
        shuffle_exclusive_test_data: bool = False
    ):
        """Load dataset in pair format.

        (Distance labels)
        """
        numeric_dataset = self.load_categorical_numeric(
            test_size,
            random_state,
            landmark_dims,
            exclusive_test_labels,
            shuffle_exclusive_test_data
        )

        pairs_train, y_train = self.__make_pairs(
            numeric_dataset.x_train,
            numeric_dataset.y_train,
            margin
        )
        pairs_test, y_test = self.__make_pairs(
            numeric_dataset.x_test,
            numeric_dataset.y_test,
            margin
        )

        x_train_1 = pairs_train[:, 0]
        x_train_2 = pairs_train[:, 1]
        x_test_1 = pairs_test[:, 0]
        x_test_2 = pairs_test[:, 1]

        return PairDataset(
            x_train_1,
            x_train_2,
            y_train,
            x_test_1,
            x_test_2,
            y_test
        )
