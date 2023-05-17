from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (BatchNormalization, Conv3D, Dense,
                                     GlobalAveragePooling3D,
                                     MaxPool3D)


def get_model(height=128, width=128, depth=64, channels=4, n_classes=2):
    """Build a 3D convolutional neural network model."""
    inputs = Input((height, width, depth, channels))

    x = Conv3D(
        filters=16, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(inputs)
    x = Conv3D(
        filters=16, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(
        filters=32, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(x)
    x = Conv3D(
        filters=32, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(
        filters=64, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(x)
    x = Conv3D(
        filters=64, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(
        filters=128, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(x)
    x = Conv3D(
        filters=128, kernel_size=3, strides=(1, 1, 1), padding='same', activation='relu'
    )(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)

    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    outputs = Dense(units=n_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='3D-CNN')
