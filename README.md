# Facial Recognition System

## Project Overview

This project aims to develop a facial recognition system capable of identifying and verifying faces in images or video streams. The system leverages advanced machine learning techniques and utilizes popular libraries such as OpenCV, Dlib, and `face_recognition`. The project is divided into several components, including data preprocessing, face detection, feature extraction, model training, and real-time integration.

## Project Structure

```
facial_recognition_system/
├── data/
│   ├── raw/                 # Raw data files (images)
│   ├── processed/           # Processed data (resized images, face encodings)
│   └── models/              # Trained models
├── src/
│   ├── data_preprocessing.py # Script for resizing images
│   ├── face_detection.py     # Script for detecting faces in images
│   ├── face_recognition.py   # Script for recognizing faces using encodings
│   ├── feature_extraction.py # Script for extracting face encodings
│   ├── model_training.py     # Script for training the face recognition model
│   ├── real_time_integration.py # Script for real-time face recognition
│   └── utils.py              # Utility functions
├── notebooks/
│   ├── EDA.ipynb             # Notebook for exploratory data analysis
│   └── Model_Training.ipynb  # Notebook for training the model
├── tests/
│   ├── test_data_preprocessing.py # Unit tests for data preprocessing
│   ├── test_face_detection.py     # Unit tests for face detection
│   ├── test_face_recognition.py   # Unit tests for face recognition
│   └── test_feature_extraction.py # Unit tests for feature extraction
├── requirements.txt           # List of project dependencies
├── README.md                  # Project documentation
└── setup.py                   # Setup script for packaging the project
```

## Getting Started

### Prerequisites

Ensure you have the following installed on your local machine:

- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/revanthchristober/Facial-Recognition-System.git
    cd Facial-Recognition-System
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Data Preprocessing**:
    - Resize images in the `data/raw` folder:
    ```sh
    python src/data_preprocessing.py
    ```

2. **Face Detection**:
    - Detect faces in the preprocessed images:
    ```sh
    python src/face_detection.py
    ```

3. **Feature Extraction**:
    - Extract face encodings from the images:
    ```sh
    python src/feature_extraction.py
    ```

4. **Model Training**:
    - Train the facial recognition model using the extracted face encodings:
    ```sh
    python src/model_training.py
    ```

5. **Real-Time Integration**:
    - Use the trained model to recognize faces in real-time:
    ```sh
    python src/real_time_integration.py
    ```

## Notebooks

- `EDA.ipynb`: Explore the dataset and perform data analysis.
- `Model_Training.ipynb`: Detailed steps for training the facial recognition model.

## Tests

- Unit tests are provided in the `tests` directory. Run them using:
    ```sh
    pytest tests/
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [scikit-learn](https://scikit-learn.org/stable/)
