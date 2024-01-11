"""Miscellaneous utility functions."""

import copy
import json
import os
from typing import Callable, Literal
from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
    Callback,
    History
)
from keras.models import Model
from keras.optimizers import Adam
from keras_tuner import HyperParameters, Tuner
from keras_tuner.src.engine.trial import Trial
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import numpy.typing as npt
from core.model import new_SNN_classifier, new_SNN_encoder, new_Siamese_Network


def new_default_callbacks(
    monitor: str = "val_loss",
    mode: str = "min",
    use_lr_schedule: bool = False,
    verbose: int = 1,
    early_stopping_patience: int = 10,
    early_stopping_start: int = 5,
    start_lr: float = 0.001,
    lr_reduce_factor: float = 0.5,
    lr_reduce_patience: int = 5,
    lr_schedule_1: int = 50,
    lr_schedule_2: int = 100,
) -> list[Callback]:
    """Create Keras train callbacks with sane defaults."""
    # Stop training if no improvement is made
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        restore_best_weights=True,
        patience=early_stopping_patience,
        verbose=verbose,
        start_from_epoch=early_stopping_start
    )

    if use_lr_schedule:
        # Reduce learning rate on schedule
        def __schedule(epoch: int) -> float:
            if epoch < lr_schedule_1:
                return start_lr
            elif epoch < lr_schedule_2:
                return start_lr * lr_reduce_factor
            else:
                return start_lr * lr_reduce_factor * lr_reduce_factor
        lr_callback = LearningRateScheduler(__schedule, verbose=verbose)
    else:
        # Reduce learning rate if no improvement is made
        lr_callback = ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=lr_reduce_factor,
            min_lr=0,
            patience=lr_reduce_patience,
            verbose=verbose
        )

    return [lr_callback, early_stopping]


def evaluate_n_way_accuracy(
    x_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[int],
    encoder: Model,
    k_prototype: int,
    iterations: int,
    random_state: int = 42,
):
    """Evaluate n-way classification accuracy.

    In this simplified case n is the same as the number of classes in `y_test`.

    `k_prototype` denotes the number of embeddings that will be averaged for each support.

    Reference:
        https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463
    """
    # Let model predict embeddings
    pred_embs = encoder.predict(x_test)

    # Match labels and embeddings in DataFrame
    df = pd.DataFrame({"label": y_test, "embedding": [y for y in pred_embs]})

    # Collect predictions and accuracies over iterations
    y_true: list[int] = []
    y_pred: list[int] = []
    acc_scores: list[float] = []

    for i in range(iterations):
        # Sample n samples per class for support prototypes
        prototype_df = df.groupby("label").sample(
            n=k_prototype,
            random_state=random_state*i
        )

        # Average n embeddings for each class -> support embeddings
        support_df = prototype_df.groupby(
            "label"
        )["embedding"].mean().reset_index()

        # Choose all other samples not used for support prototypes as queries
        query_df = df.drop(prototype_df.index)

        # Extract support labels
        support_labels = support_df["label"].tolist()

        def __dist(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]):
            """Euclidean distance."""
            return np.linalg.norm(a - b)

        # Collect predictions
        it_y_true: list[int] = []
        it_y_pred: list[int] = []

        for _, query_row in query_df.iterrows():
            # Get query embedding and label
            query_emb = np.array(query_row["embedding"])
            query_label = query_row["label"]

            # Calculate distance of query to every support
            dists: list[np.float32] = []
            for _, support_row in support_df.iterrows():
                support_emb = np.array(support_row["embedding"])
                dist = __dist(query_emb, support_emb)
                dists.append(dist)

            # Select suppurt with min distance as prediction
            pred_idx = np.argmin(np.array(dists))
            it_y_pred.append(support_labels[pred_idx])
            it_y_true.append(query_label)

        # Compute accuracy
        acc = accuracy_score(it_y_true, it_y_pred)

        # Collect iteration results
        y_true.extend(it_y_true)
        y_pred.extend(it_y_pred)
        acc_scores.append(acc)

    # Calculate accuracy metrics
    np_acc_scores = np.array(acc_scores)
    mean_acc = np.mean(np_acc_scores)
    min_acc = np.min(np_acc_scores)
    max_acc = np.max(np_acc_scores)
    std_acc = np.std(np_acc_scores)

    return (
        y_true,
        y_pred,
        mean_acc,
        min_acc,
        max_acc,
        std_acc
    )


def new_SNN_builder(
    mode: Literal["classifier", "siamese"],
    loss: Callable | str,
    metrics: list[str] | None = None,
    num_classes: int | None = None,
    distance: Callable | None = None
):
    """Create SNN builder for Keras Tuner.

    Reference:
        https://keras.io/guides/keras_tuner/getting_started/#tune-the-model-architecture
    """

    def __build_model(hp: HyperParameters, get_encoder: bool = False):
        # Encoder part
        snn_encoder = new_SNN_encoder(
            dense_count=hp.Int("dense_count", min_value=2,
                               max_value=10, step=2),
            dense_base=hp.Int("dense_base", min_value=8, max_value=64, step=8),
            dropout=hp.Boolean("encoder_dropout")
        )

        if mode == "classifier":
            # Full classifier
            assert num_classes is not None
            model = new_SNN_classifier(
                snn_encoder,
                num_classes=num_classes,
                dropout=hp.Boolean("decoder_dropout")
            )
        else:
            # Siamese NN
            model = new_Siamese_Network(
                snn_encoder,
                distance=distance,
                batch_normalization=hp.Boolean("siamese_bn"),
                sigmoid_output=hp.Boolean("siamese_dropout")
            )

        # Use Adam optimizer
        adam = Adam(
            learning_rate=hp.Float(
                "start_lr", min_value=0.00005, max_value=0.001, step=0.00005
            ),
            beta_1=hp.Float(
                "beta_1", min_value=0.7, max_value=0.9, step=0.1
            ),
            beta_2=hp.Float(
                "beta_2", min_value=0.99, max_value=0.999, step=0.001
            ),
            epsilon=hp.Float(
                "epsilon", min_value=0.001, max_value=0.01, step=0.001
            )
        )

        # Compile model
        model.compile(
            loss=loss,
            optimizer=adam,
            metrics=metrics
        )

        if not get_encoder:
            return model
        else:
            return snn_encoder, model

    return __build_model


class TunerHistory:
    """Keep track of tuner history.

    (Index histories by trial.)
    """

    __history: dict[tuple[str, str], History] = {}

    def get_or_init_trial_history(
        self,
        trial_id: tuple[str, str],
        model: Model
    ) -> History:
        """Get history for trial or initialize if not exists.

        (Must provide `Model` in case trial history must be initialized.
        -> Prevents error with Keras `History` callback.)
        """
        # Get history for exact trial
        trial_history = self.__history.get(trial_id)

        # Initialize trial history if exact history does not exist
        if trial_history is None:
            # Try finding a previous trial with same hyperparameter hash
            trial_hp_hash = trial_id[1]
            full_trial_ids = list(self.__history.keys())
            hp_hashes = [id[1] for id in full_trial_ids]
            # If exists base history of current trial off of previous trial
            try:
                prev_trial_idx = hp_hashes.index(trial_hp_hash)
                prev_trial_id = full_trial_ids[prev_trial_idx]
                trial_history = copy.copy(self.__history[prev_trial_id])
            # Else create new history
            except ValueError:
                trial_history = History()
                trial_history.on_train_begin()
            self.__history[trial_id] = trial_history

        # Set model for trial history
        # (Might also be uninitialized if history was loaded from JSON)
        if trial_history.model is None:
            trial_history.model = model

        return trial_history

    def get_history_by_trial_id(self, trial_id: str) -> History | None:
        """Get history using only Keras tuner trial id."""
        full_trial_ids = list(self.__history.keys())
        trial_ids = [id[0] for id in full_trial_ids]
        # Try finding matching trial
        try:
            trial_idx = trial_ids.index(trial_id)
            full_trial_id = full_trial_ids[trial_idx]
            return self.__history[full_trial_id]
        except ValueError:
            return None

    def to_json(self, filepath: str) -> None:
        """Serialize trial histories as JSON string."""
        # Collect trial histories into JSON-serializable dict
        history = {
            trial_id: {
                "config": trial_config,
                "history": {
                    # Keras tuner saves metric values as list of np.float32???
                    # Anyways, we need to convert them for JSON serialization
                    k: [float(n) for n in v] for k, v
                    in trial_history.history.items()
                },
                "epoch": trial_history.epoch,
            }
            for [trial_id, trial_config], trial_history
            in self.__history.items()
        }

        # Save to JSON file
        with open(filepath, 'w') as f:
            return json.dump(history, f)

    def init_from_json(self, filepath: str) -> None:
        """Set trial histories' state from JSON contents."""
        # Load trial histories from JSON file
        with open(filepath, 'r') as f:
            history = json.load(f)

        # Reconstruct `History` classes from serialized dict
        for trial_id, trial_history_dict in history.items():
            trial_history = History()
            trial_config = trial_history_dict["config"]
            trial_history.history = trial_history_dict["history"]
            trial_history.epoch = trial_history_dict["epoch"]
            self.__history[(trial_id, trial_config)] = trial_history


class TunerHistoryCallback(Callback):
    """Hack to collect training history for Keras Tuner trials.

    Usage:

    ```python
    # Instantiate for tuner
    tuner_history_cb = TunerHistoryCallback(tuner)

    # Use as search callback
    tuner.search(
        ...,
        callbacks=[tuner_history_cb]
    )

    # Get entire model history for trial
    best_model_history = tuner_history_cb.get_trial_history(
        best_trial.trial_id
    )
    ```

    History is persisted the same way as tuner checkpoints.

    Reference:
        https://stackoverflow.com/a/70410522
    """

    # Keep track of histories for all tries
    # Use class instead of dict, because tuner copies cb class for every trial
    # -> Keep state across trials
    __history: TunerHistory
    __trial: Trial | None
    __trial_id: tuple[str, str]
    __tuner: Tuner

    def __init__(self, tuner: Tuner):
        """Create new TunerHistoryCallback."""
        super().__init__()

        # Initialize tuner
        self.__tuner = tuner
        tuner.on_trial_begin = self.__on_trial_begin
        tuner.on_trial_end = self.__on_trial_end

        # Initialize history
        self.__history = TunerHistory()

        # Check if history file exists in tuner's project dir
        tuner_dir = self.__tuner.directory
        project_subdir = self.__tuner.project_name
        history_dir = f"{tuner_dir}/{project_subdir}"
        history_filepath = f"{history_dir}/history.json"

        # If exists, load history from JSON
        if os.path.isfile(history_filepath):
            print(f"Loading history from JSON {history_filepath}")
            self.__history.init_from_json(history_filepath)

    def __hp_hash(self, hp: dict[str, int | float | bool]) -> str:
        """Create hash for hyperparameter config."""
        # Remove all tuner-specific values (unique for every trial)
        hp = copy.copy(hp)
        tuner_keys = [
            key for key in hp.keys()
            if "tuner" in key
        ]
        for tuner_key in tuner_keys:
            del hp[tuner_key]

        return json.dumps(hp)

    def __full_trial_id(self, trial: Trial) -> tuple[str, str]:
        """Get trial id and hash of hyperparameters for trial."""
        trial_id = str(trial.trial_id)
        hp_values = trial.hyperparameters.values
        hp_hash = self.__hp_hash(hp_values)
        return (trial_id, hp_hash)

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end callback."""
        # Ensure current trial is set
        if self.__trial is None:
            raise Exception("No trial set during epoch callback.")

        # Get history for current trial
        trial_history = self.__history.get_or_init_trial_history(
            self.__trial_id,
            self.model
        )

        # Forward epoch end callback to trial history
        trial_history.on_epoch_end(epoch, logs)

    def __on_trial_begin(self, trial: Trial):
        """Sync. current trial.

        Overwrite tuner `on_trial_begin`:

        ```python
        tuner.on_trial_begin = tuner_history_cb.__on_trial_begin
        ```
        """
        self.__trial = trial
        self.__trial_id = self.__full_trial_id(trial)

    def __on_trial_end(self, trial: Trial):
        """Persist history as JSON."""
        # Get path to tuner's project directory
        tuner_dir = self.__tuner.directory
        project_subdir = self.__tuner.project_name
        history_dir = f"{tuner_dir}/{project_subdir}"

        # Save current history to JSON
        history_filepath = f"{history_dir}/history.json"
        print(f"Saving history to JSON {history_filepath}")
        self.__history.to_json(history_filepath)

        # Execute trial end as in `keras.tuner.BaseTuner.on_epoch_end`
        # Since original callback is overwritten by this method
        # Super hacky, but works
        # (No idea why BaseTuner would have callback that can't be overwritten)
        self.__tuner.oracle.end_trial(trial)
        self.__tuner.save()

    def get_trial_history(self, trial_id: str) -> History:
        """Get history for trial."""
        return self.__history.get_history_by_trial_id(str(trial_id))
