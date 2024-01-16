"""Distance metrics for embedding vectors."""

import tensorflow as tf


def euclidean_distance(vects: list[tf.Tensor]) -> tf.Tensor:
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


def cosine_distance(vects: list[tf.Tensor]) -> tf.Tensor:
    """Find the Cosine distance between two vectors.

    Reference:
        https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/losses/triplet.py#L310-L344
    """
    x, y = vects
    x = tf.math.l2_normalize(x)
    y = tf.math.l2_normalize(y)

    angular_distance = 1 - tf.multiply(x, y)
    angular_distance = tf.maximum(angular_distance, 0.0)
    angular_distance = tf.math.reduce_sum(
        angular_distance,
        axis=1,
        keepdims=True
    )

    return angular_distance
