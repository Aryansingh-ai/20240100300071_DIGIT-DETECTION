# 20240100300071_DIGIT-DETECTION
This project implements two approaches for recognizing images and digits.
# Image Classification & Digit Detection

## Project Description
This project implements two approaches for recognizing images and digits:
1. **Image Classification using Machine Learning (SVM & Scikit-learn)**: Classifies handwritten digits using Support Vector Machines (SVM) on the Digits dataset.
2. **Digit Detection using Deep Learning (CNN & TensorFlow/Keras)**: Detects handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Dataset
- **Digits Dataset (Scikit-learn)**: Automatically loaded using `datasets.load_digits()`.
  - Alternative Download: [Kaggle Digits Dataset](https://www.kaggle.com/datasets/sachinpatel21/sklearn-digits-dataset)
- **MNIST Dataset (Handwritten Digits for CNN)**: Automatically loaded using `keras.datasets.mnist.load_data()`.
  - Alternative Download: [Kaggle MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or [Official MNIST ZIP](http://yann.lecun.com/exdb/mnist/)

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/image-classification-digit-detection.git
   cd image-classification-digit-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### 1. Image Classification (SVM)
Run the following command to train and test an SVM classifier:
```sh
python image_classification.py
```

### 2. Digit Detection (CNN)
Run the following command to train and test a CNN model:
```sh
python digit_detection.py
```

## Project Structure
```
├── image_classification.py  # SVM model for image classification
├── digit_detection.py       # CNN model for digit recognition
├── datasets/                # Place for downloaded datasets (if needed)
├── README.md                # Project documentation
├── requirements.txt         # Required Python packages
```

## Requirements
- Python 3.x
- TensorFlow/Keras
- Scikit-learn
- NumPy
- Matplotlib

## Results
- **SVM Classifier**: Achieves high accuracy on the Digits dataset.
- **CNN Model**: Detects handwritten digits with >98% accuracy on MNIST.

## Contributing
Feel free to fork and submit pull requests to improve the project.

## License
MIT License

