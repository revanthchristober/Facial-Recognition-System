import unittest
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import face_recognition
from face_recognition import recognize_faces

class TestFaceRecognition(unittest.TestCase):

    def test_recognize_faces(self):
        known_image = face_recognition.load_image_file("data/processed/known_person.jpg")
        unknown_image = face_recognition.load_image_file("data/processed/unknown_person.jpg")

        known_face_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

        result = recognize_faces([known_face_encoding], unknown_face_encoding)
        self.assertIsNotNone(result, "Face recognition failed")

if __name__ == '__main__':
    unittest.main()
