# ECE1508 Deep Learning Project — Sentiment Analysis

## 1. Overview

This project implements multiple approaches for sentiment classification on the Sentiment140 dataset, including:

- Logistic Regression with TF-IDF features
- TextCNN (Convolutional Neural Network)
- TextLSTM with attention mechanism

The goal is to compare traditional machine learning methods with deep learning models for text classification.

---
## 2. Setup Instructions

### 2.1 Install Dependencies

```
pip install -r requirements.txt
```

### 2.2 Download NLTK Resources

Run once:

```
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

## 3. Dataset Preparation

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

## 4. Training Models

### 4.1 Logistic Regression

```
python train_logistic.py
```

### 4.2 CNN Model

```
python train_cnn.py
```

### 4.3 LSTM Model

```
python train_lstm.py
```

### 4.4 Hyperparameter Tuning (LSTM)

```
python tune_lstm.py
```

---

## 5. Reproducibility

- Fixed random seed: 2026
- Stratified train/validation splits
- Consistent preprocessing pipelines

---

## 6. Notes

- `processed/` is used for traditional ML (TF-IDF)
- `processed_v2/` is used for deep learning models
- Stopwords are kept in v2 to preserve semantic meaning
- Lemmatization is removed in v2 for better sequence modeling

---

## 7. Dependencies

See `requirements.txt`

---