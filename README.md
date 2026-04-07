# ECE1508 Deep Learning Project — Sentiment Analysis

## 1. Overview

This project implements multiple approaches for sentiment classification on the Sentiment140 dataset, including:

- Logistic Regression with TF-IDF features
- TextCNN (Convolutional Neural Network)
- TextLSTM with attention mechanism

The goal is to compare traditional machine learning methods with deep learning models for text classification.

---

## 2. Project Structure

```
.
├── data/
│   ├── raw/                # Original dataset (downloaded from Kaggle)
│   ├── processed/          # Preprocessed data (v1 for logistic regression)
│   └── processed_v2/       # Preprocessed data (v2 for deep learning models)
│
├── preprocess.py           # Preprocessing pipeline (v1)
├── preprocess_v2.py        # Preprocessing pipeline (v2)
├── download_data.py        # Script to download dataset
│
├── vocab.py                # Vocabulary builder
├── cnn_dataset.py          # Dataset class
│
├── text_cnn.py             # CNN model
├── text_lstm.py            # LSTM model
│
├── train_logistic.py       # Logistic Regression training
├── train_cnn.py            # CNN training
├── train_lstm.py           # LSTM training
├── tune_lstm.py            # Hyperparameter tuning
│
├── requirements.txt
└── README.md
```

---

## 3. Setup Instructions

### 3.1 Install Dependencies

```
pip install -r requirements.txt
```

### 3.2 Download NLTK Resources

Run once:

```
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

## 4. Dataset Preparation

### Step 1: Download dataset

```
python download_data.py
```

Note:
- Requires Kaggle API setup
- Dataset: Sentiment140

---

### Step 2: Preprocess data

For Logistic Regression:

```
python preprocess.py
```

For Deep Learning models:

```
python preprocess_v2.py
```

---

## 5. Training Models

### 5.1 Logistic Regression

```
python train_logistic.py
```

### 5.2 CNN Model

```
python train_cnn.py
```

### 5.3 LSTM Model

```
python train_lstm.py
```

### 5.4 Hyperparameter Tuning (LSTM)

```
python tune_lstm.py
```

---

## 6. Reproducibility

- Fixed random seed: 2026
- Stratified train/validation splits
- Consistent preprocessing pipelines

---

## 7. Notes

- `processed/` is used for traditional ML (TF-IDF)
- `processed_v2/` is used for deep learning models
- Stopwords are kept in v2 to preserve semantic meaning
- Lemmatization is removed in v2 for better sequence modeling

---

## 8. Dependencies

See `requirements.txt`

---