import numpy as np
import cv2
import sys

# Ensure the root directory is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def resize_image(image, size):
    return cv2.resize(image, size)

def save_encodings(encodings, labels, enc_file, label_file):
    np.save(enc_file, encodings)
    np.save(label_file, labels)

def load_encodings(enc_file, label_file):
    encodings = np.load(enc_file)
    labels = np.load(label_file)
    return encodings, labels
