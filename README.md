# Sarcasm Detector

This repository contains code for a sarcasm detector, which aims to classify headlines as sarcastic or non-sarcastic. The dataset used for training and evaluation consists of headlines along with their corresponding labels (sarcastic or not).

## Dataset

The dataset contains headlines with labels indicating whether each headline is sarcastic or not. The headlines will be tokenized and preprocessed to be fed into the model.

## Main.py

### Data Loading and Preprocessing

- The dataset is loaded from a JSON file using the `pandas` library.
- The headlines and labels are extracted from the dataset and split into training and testing sets using the specified `split_size`.
- The headlines are tokenized and converted to sequences using `Tokenizer` from `tensorflow.keras.preprocessing.text`.
- The sequences are then padded to a fixed length to ensure uniformity in input size.

### Model Architecture

- The sarcasm detector model is built using `tensorflow.keras.models.Sequential`.
- The model consists of an Embedding layer with a vocabulary size of 10,000 and an embedding dimension of 16. This layer converts the sequences of words into dense vectors.
- The GlobalAveragePooling1D layer averages the embeddings to obtain a fixed-length output.
- The model further contains two Dense layers with ReLU activation functions for feature extraction and a final Dense layer with a sigmoid activation function for binary classification (sarcastic or non-sarcastic).

### Training and Evaluation

- The model is trained using binary cross-entropy loss and the Adam optimizer for 30 epochs.
- The training and validation accuracy are monitored during training.

## Usage

To use this sarcasm detector, you can follow these steps:

1. Install the required libraries and dependencies by running:

   ```bash
   pip install pandas numpy tensorflow
   ```
Feel free to experiment with different hyperparameters, model architectures, or other NLP techniques to improve the sarcasm detection accuracy.

For any questions or suggestions, please contact [Francesco Alotto](mailto:franalotto94@gmail.com). Happy sarcasm detection with AI! 😄🤖