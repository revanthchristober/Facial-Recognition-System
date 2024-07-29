import cv2
import dlib
import numpy as np
import sys

# Ensure the root directory is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_features(image, face):
    landmarks = predictor(image, face)
    points = []
    for i in range(0, 68):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    return points

if __name__ == "__main__":
    image = cv2.imread('data/processed/img_0.jpg')
    faces = detect_faces(image)
    for face in faces:
        features = extract_features(image, face)
        for (x, y) in features:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    cv2.imshow('Feature Extraction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
