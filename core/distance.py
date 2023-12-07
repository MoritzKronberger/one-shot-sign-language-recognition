"""Distance metrics for embedding vectors."""

import tensorflow as tf


def euclidean_distance(vects: list[tf.Tensor]):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance between vectors.

    Reference:
        https://keras.io/examples/vision/siamese_network
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(
        tf.math.square(x - y),
        axis=1,
        keepdims=True
    )
    return tf.math.sqrt(
        tf.math.maximum(sum_square, tf.keras.backend.epsilon())
    )
