"""Create Keras models."""

from typing import Callable
from keras.layers import (
    Input,
    LayerNormalization,
    Flatten,
    Dense,
    AlphaDropout,
    Lambda,
    BatchNormalization
)
from keras.models import Model


def new_SNN_encoder(
    input_shape: list[int] = [21, 3],
    dense_count: int = 6,
    dense_base: int = 48,
    dropout: bool = True
) -> Model:
    """Create new self normalizing encoder network.

    Default config matches original model from Kaggle notebook:
        https://www.kaggle.com/code/moritzkronberger/snn-0-94-acc
    """
    # Model input
    inputs = Input(
        shape=input_shape,
        name='Landmark_Input'
    )

    # Layer normalization (only for inputs)
    layerNorm = LayerNormalization(name='Layer_Norm')(inputs)

    # Flatten 3D inputs
    flatten = Flatten(name='Flatten_Landmarks')(layerNorm)

    # Create conical, fully connected NN with n dense layers
    out = flatten

    for i in range(dense_count):
        units = (dense_count-i) * (dense_count-i) * dense_base
        dense = Dense(
            units,
            kernel_initializer="lecun_normal",
            bias_initializer="zeros",
            activation='selu',
            name=f'Dense_{i+1}'
        )
        a_dropout = AlphaDropout(0.05, name=f'Dropout_{i+1}')
        out = dense(out)
        if (dropout and i < dense_count-1):
            out = a_dropout(out)

    return Model(inputs=inputs, outputs=out, name=f"SNN_{dense_count}_encoder")


def new_SNN_classifier(
    encoder: Model,
    dropout: bool = True,
    num_classes: int = 24
):
    """Append classification decoder to SNN encoder.

    Default config matches original model from Kaggle notebook:
        https://www.kaggle.com/code/moritzkronberger/snn-0-94-acc
    """
    if dropout:
        decoder = AlphaDropout(0.05, name="Output_Dropout")(encoder.output)
        decoder = Dense(
            num_classes,
            activation="softmax",
            name="Output_Vector"
        )(decoder)
    else:
        decoder = Dense(
            num_classes,
            activation="softmax",
            name="Output_Vector"
        )(encoder.output)

    return Model(
        inputs=encoder.input,
        outputs=decoder,
        name=encoder.name.replace("encoder", "classifier")
    )


def new_Siamese_Network(
    encoder: Model,
    distance: Callable,
    batch_normalization: bool = True,
    sigmoid_output: bool = True
) -> Model:
    """Create Siamese Neural Network using encoder.

    Reference:
        https://keras.io/examples/vision/siamese_network
    """
    input_shape = encoder.input_shape[1:]

    input_1 = Input(input_shape)
    input_2 = Input(input_shape)

    tower_1 = encoder(input_1)
    tower_2 = encoder(input_2)

    siamese = Lambda(distance)([tower_1, tower_2])

    if batch_normalization:
        siamese = BatchNormalization()(siamese)
    if sigmoid_output:
        siamese = Dense(1, activation="sigmoid")(siamese)

    return Model(
        inputs=[input_1, input_2],
        outputs=siamese,
        name=encoder.name.replace("encoder", "siamese")
    )
