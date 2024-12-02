
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(128, 128, 1), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, model_save_path="models/cnn_model.h5"):
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=32, image_size=(128, 128), color_mode='grayscale')
    model = create_model()
    model.fit(train_ds, epochs=10)
    model.save(model_save_path)

# Exemplo de uso
train_model("data/processed")
