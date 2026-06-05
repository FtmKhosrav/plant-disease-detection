from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models


def build_rice_model(num_classes):

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224,224,3))

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
