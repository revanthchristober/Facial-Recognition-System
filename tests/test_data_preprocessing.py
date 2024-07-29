import unittest
import cv2
import os
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import load_images_from_folder, preprocess_image

class TestDataPreprocessing(unittest.TestCase):

    def test_load_images_from_folder(self):
        images = load_images_from_folder('data/raw')
        self.assertTrue(len(images) > 0, "No images loaded from folder")

    def test_preprocess_image(self):
        image = cv2.imread('data/raw/sample_image.jpg')
        preprocessed_image = preprocess_image(image)
        self.assertEqual(preprocessed_image.shape, image.shape[:2], "Image preprocessing failed")

if __name__ == '__main__':
    unittest.main()
