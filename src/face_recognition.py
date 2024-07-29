import face_recognition
import sys

# Ensure the root directory is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def recognize_faces(known_face_encodings, face_encoding):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return best_match_index
    return None

if __name__ == "__main__":
    known_image = face_recognition.load_image_file("data/processed/known_person.jpg")
    unknown_image = face_recognition.load_image_file("data/processed/unknown_person.jpg")

    known_face_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

    result = recognize_faces([known_face_encoding], unknown_face_encoding)
    print("Face recognized:", result is not None)
