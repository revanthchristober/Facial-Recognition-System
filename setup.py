from setuptools import setup, find_packages

setup(
    name='facial_recognition_system',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'dlib',
        'face_recognition',
        'scikit-learn',
        'joblib'
    ],
    entry_points={
        'console_scripts': [
            'data_preprocessing=src.data_preprocessing:main',
            'face_detection=src.face_detection:main',
            'face_recognition=src.face_recognition:main',
            'feature_extraction=src.feature_extraction:main',
            'model_training=src.model_training:main',
            'real_time_integration=src.real_time_integration:main'
        ],
    },
)
