from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.h5'
SAMPLE_IMAGES_DIR = BASE_DIR / 'sample_images'


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build_model() -> Sequential:
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ])


def create_callbacks() -> list[tf.keras.callbacks.Callback]:
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    return [reduce_lr, early_stopping]


def save_sample_images(x_test: np.ndarray, count: int = 5) -> None:
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(min(count, len(x_test))):
        plt.imsave(SAMPLE_IMAGES_DIR / f'test_{i}.png', x_test[i].reshape(28, 28), cmap='gray')


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist_data()
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(x_train)

    model = build_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = create_callbacks()
    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Độ chính xác trên tập kiểm tra: {test_accuracy * 100:.2f}%')

    save_sample_images(x_test)
    model.save(MODEL_PATH)
    print(f'Mô hình đã được lưu thành công vào "{MODEL_PATH}"')


if __name__ == '__main__':
    main()