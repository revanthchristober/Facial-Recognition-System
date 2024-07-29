import unittest
import cv2
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from face_detection import detect_faces
from feature_extraction import extract_features

class TestFeatureExtraction(unittest.TestCase):

    def test_extract_features(self):
        image = cv2.imread('data/processed/img_0.jpg')
        faces = detect_faces(image)
        for face in faces:
            features = extract_features(image, face)
            self.assertTrue(len(features) > 0, "No features extracted")

if __name__ == '__main__':
    unittest.main()
