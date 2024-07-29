import cv2
import os
import numpy as np
import sys

# Ensure the root directory is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

if __name__ == "__main__":
    folder = 'data/raw'
    images = load_images_from_folder(folder)
    preprocessed_images = [preprocess_image(img) for img in images]
    # Save preprocessed images
    for idx, img in enumerate(preprocessed_images):
        cv2.imwrite(f'data/processed/img_{idx}.jpg', img)
