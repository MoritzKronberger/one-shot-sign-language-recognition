"""Miscellaneous utility functions."""

from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
    Callback
)


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
