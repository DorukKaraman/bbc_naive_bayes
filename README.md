# bbc_naive_bayes
A from-scratch implementation of a Multinomial Naive Bayes classifier for text classification using the BBC Sports dataset. Includes Laplace smoothing, training/evaluation scripts, and accuracy reporting — built only with Python and pandas.

This repository contains a simple implementation of a **Multinomial Naive Bayes Classifier** for text classification.  
The model is trained and evaluated on the [BBC Sports dataset](http://mlg.ucd.ie/datasets/bbc.html), which contains sports news articles categorized into:

- **Athletics (0)**
- **Cricket (1)**
- **Football (2)**
- **Rugby (3)**
- **Tennis (4)**

---

## 🚀 Features
- Implements **Naive Bayes from scratch** (no scikit-learn).
- Supports **Laplace smoothing** to handle unseen words.
- Evaluates classification **accuracy** on validation data.

---

## 📂 Dataset
The dataset should be placed under `dataset/` directory:
dataset/
├── bbcsports_train.csv
├── bbcsports_val.csv

- Each row represents a document.
- Columns are word counts (bag-of-words features).
- The **last column is the class label** (0–4).

---

## Example output:

=== Without Laplace Smoothing ===
Accuracy = 85.32 %
Correct: 225 | Incorrect: 39

=== With Laplace Smoothing ===
Accuracy = 88.17 %
Correct: 233 | Incorrect: 31


---

## 📝 Notes

- Without **Laplace smoothing**, unseen words in validation may cause **zero-probability issues**.  
- With **Laplace smoothing**, we add `+1` to each word count to avoid this.

---

## 📖 Learning Outcome

This project demonstrates how **Naive Bayes** works under the hood for document classification:

- Feature likelihoods are computed as **conditional probabilities**.  
- Document scores are computed as **log-probabilities** to avoid underflow.  
- The class with the **maximum score** is chosen as the prediction.  

