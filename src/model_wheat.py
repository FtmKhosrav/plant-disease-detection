from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


def build_wheat_model(num_classes):

    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(224,224,3))

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
