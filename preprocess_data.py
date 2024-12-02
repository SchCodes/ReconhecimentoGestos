
import cv2
import os
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

def augment_image(image):
    flipped = cv2.flip(image, 1)
    return [image, flipped]

def preprocess_and_augment(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        image = preprocess_image(file_path)
        augmented_images = augment_image(image)
        for i, img in enumerate(augmented_images):
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_{i}.jpg")
            cv2.imwrite(output_file, (img * 255).astype(np.uint8))

# Exemplo de uso
preprocess_and_augment("data/raw/positive", "data/processed/positive")
