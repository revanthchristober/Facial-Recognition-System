import cv2
import dlib
import sys

# Ensure the root directory is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

detector = dlib.get_frontal_face_detector()

def detect_faces(image):
    faces = detector(image)
    return faces

if __name__ == "__main__":
    image = cv2.imread('data/processed/img_0.jpg')
    faces = detect_faces(image)
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
