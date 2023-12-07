"""Miscellaneous utility functions."""

from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
    Callback
)
from keras.models import Model
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.typing as npt


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


def evaluate_classification(
    model: Model,
    x_test: npt.NDArray[np.float32],
    y_test: npt.NDArray[int],
    iterations: int = 10,
    random_seed: int = 42,
    batch_size: int = 32,
    verbose: int = 1
) -> tuple[list[int], list[int], list[float], float]:
    """Evaluate contrastive model on classification task."""
    # Sort dataset by numeric labels
    label_sort_idx = np.argsort(y_test)
    y_test = y_test[label_sort_idx]
    x_test = x_test[label_sort_idx]

    # Get unique labels (= classes)
    unique_labels = np.unique(y_test)
    num_classes = len(unique_labels)

    # Seed for reproducibility
    np.random.seed(random_seed)

    # Collect evaluation results for every iteration
    it_y_true: list[int] = []
    it_y_pred: list[int] = []
    it_acc: list[float] = []

    # Perform evaluation for every iteration
    for i in range(iterations):
        # Initialize array to store indices for reference samples/labels
        reference_indices = np.zeros(num_classes, dtype=int)

        # Iterate over classes and get a random index to use as reference
        for i, class_label in enumerate(unique_labels):
            indices_for_label = np.where(y_test == class_label)[0]
            random_index = np.random.choice(indices_for_label)
            reference_indices[i] = random_index

        # Get test labels
        test_labels = np.delete(y_test, reference_indices)

        # Get test and reference samples
        reference_samples = np.take(x_test, reference_indices, axis=0)
        test_samples = np.delete(x_test, reference_indices, axis=0)

        # Collect reference and test samples for test data
        x_test_1: list[npt.NDArray[np.float32]] = []
        x_test_2: list[npt.NDArray[np.float32]] = []

        for test_sample in test_samples:
            # Add reference samples for every test sample
            x_test_1.extend(reference_samples)
            # Repeat test sample for every reference sample
            x_test_2.extend([test_sample for _ in range(num_classes)])

        x_test_1_np = np.array(x_test_1)
        x_test_2_np = np.array(x_test_2)

        # Make predictions
        predictions = model.predict(
            [x_test_1_np, x_test_2_np],
            batch_size=batch_size,
            verbose=verbose
        )

        # Group predictions against references for every test sample
        predictions = predictions.reshape((-1, num_classes))
        assert predictions.shape[0] == len(test_samples)

        # Get index of lowest distance score for every test sample
        pred_label_idx = np.argmin(predictions, axis=1)

        # Get predicted label for every image
        y_pred = [unique_labels[lbl_idx] for lbl_idx in pred_label_idx]

        # Ground truth == test labels
        y_true = list(test_labels)

        # Get model accuracy
        acc = accuracy_score(y_true, y_pred)

        # Collect results for iteration
        it_y_true.extend(y_true)
        it_y_pred.extend(y_pred)
        it_acc.append(acc)

    # Calculate mean accuracy
    mean_acc = float(np.mean(it_acc))

    return it_y_true, it_y_pred, it_acc, mean_acc
