# Amazon Review Sentiment Analysis

Sentiment classification on Amazon customer reviews using classical ML, built to handle real-world class imbalance, not just chase accuracy on a clean dataset.

## What it does
Classifies customer reviews as positive or negative by cleaning review text, converting it to numerical features with TF-IDF, and training models to predict sentiment. Specifically tackles class imbalance (far more positive than negative reviews) using SMOTE.

## Tech Stack
- Python, Pandas, Scikit-learn
- TF-IDF Vectorization
- SMOTE (imbalanced-learn)
- Logistic Regression, Random Forest

## Pipeline
1. **Data Cleaning** - lowercase, remove punctuation/numbers, remove stopwords, stem words to root form
2. **Feature Extraction** - TF-IDF vectorization (top 5,000 features)
3. **Train/Test Split** - 80/20 split
4. **Class Imbalance Handling** - SMOTE applied only on training data to avoid data leakage
5. **Model Training** - Logistic Regression and Random Forest, compared
6. **Evaluation** - precision, recall, F1-score per class

## Dataset
Amazon product reviews dataset (20,000+ reviews), sourced from Kaggle.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook/
```

## Key Learning
Handling class imbalance mattered more than model choice here — without SMOTE, both models defaulted toward predicting the majority class and missed real negative reviews.
