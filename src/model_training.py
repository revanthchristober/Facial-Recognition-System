from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import joblib
import sys

# Ensure the root directory is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model(face_encodings, labels):
    X_train, X_test, y_train, y_test = train_test_split(face_encodings, labels, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, 'data/models/face_recognition_model.pkl')
    print(f"Model accuracy: {model.score(X_test, y_test) * 100:.2f}%")

if __name__ == "__main__":
    face_encodings = np.load('data/processed/face_encodings.npy')
    labels = np.load('data/processed/labels.npy')
    train_model(face_encodings, labels)
