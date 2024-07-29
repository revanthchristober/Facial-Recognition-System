import unittest
import cv2
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from face_detection import detect_faces

class TestFaceDetection(unittest.TestCase):

    def test_detect_faces(self):
        image = cv2.imread('data/processed/img_0.jpg')
        faces = detect_faces(image)
        self.assertTrue(len(faces) > 0, "No faces detected")

if __name__ == '__main__':
    unittest.main()
