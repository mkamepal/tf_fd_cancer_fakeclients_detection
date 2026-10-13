"""CIFAR-10 ResNet-18 with Group Normalization.

Input: 32x32 RGB images.
Output: probabilities for 10 classes.
"""

import tensorflow as tf


def create_keras_model():
    layers = tf.keras.layers

    def conv_gn(x, filters, kernel_size, stride=1, name="conv"):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name}_conv",
        )(x)

        return layers.GroupNormalization(
            groups=8,
            axis=-1,
            epsilon=1e-5,
            name=f"{name}_gn",
        )(x)

    def residual_block(x, filters, stride=1, name="block"):
        shortcut = x

        # Main path: Conv -> GN -> ReLU -> Conv -> GN.
        x = conv_gn(x, filters, 3, stride=stride, name=f"{name}_1")
        x = layers.ReLU(name=f"{name}_relu1")(x)
        x = conv_gn(x, filters, 3, name=f"{name}_2")

        # Match the shortcut's spatial dimensions and channel count.
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = conv_gn(
                shortcut,
                filters,
                kernel_size=1,
                stride=stride,
                name=f"{name}_shortcut",
            )

        x = layers.Add(name=f"{name}_add")([x, shortcut])
        return layers.ReLU(name=f"{name}_relu2")(x)

    inputs = layers.Input(shape=(32, 32, 3), name="image")

    # CIFAR stem: preserve 32x32 resolution; no initial max pooling.
    x = conv_gn(inputs, 64, 3, name="stem")
    x = layers.ReLU(name="stem_relu")(x)

    # Four stages, two residual blocks per stage: [2, 2, 2, 2].
    for stage, filters in enumerate((64, 128, 256, 512), start=1):
        stride = 1 if stage == 1 else 2

        x = residual_block(
            x,
            filters,
            stride=stride,
            name=f"stage{stage}_block1",
        )
        x = residual_block(
            x,
            filters,
            stride=1,
            name=f"stage{stage}_block2",
        )

    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    outputs = layers.Dense(
        10,
        activation="softmax",
        name="predictions",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cifar10_resnet18_gn",
    )
