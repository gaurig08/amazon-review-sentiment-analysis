# Amazon Review Sentiment Analysis
 
This project classifies Amazon reviews as Negative, Neutral, or Positive. It compares two models, Logistic Regression and Random Forest, trained on TF-IDF text features. Since most reviews are positive, I used SMOTE to fix the class imbalance and checked how much it actually helped.
 
## Tech Stack
 
- Python
- Pandas, scikit-learn
- NLTK (stopwords, Porter stemming)
- TF-IDF (unigrams + bigrams)
- SMOTE (imbalanced-learn)
- Logistic Regression, Random Forest
## Dataset
 
[Datafiniti "Consumer Reviews of Amazon Products"](https://data.world/datafiniti/consumer-reviews-of-amazon-products): 34,660 real Amazon reviews with star ratings and review text. Also available on Kaggle as `1429_1.csv`. After removing rows with missing text or rating, 34,626 reviews are left, well past the 20,000+ target.

 
| Sentiment | Count | % |
|---|---|---|
| Positive (4-5 stars) | 32,315 | 93.3% |
| Neutral (3 stars) | 1,499 | 4.3% |
| Negative (1-2 stars) | 812 | 2.3% |
 
## Pipeline
 
1. **Label**: turn star rating into sentiment (1-2 stars = Negative, 3 = Neutral, 4-5 = Positive)
2. **Clean and stem**: lowercase the text, remove URLs and non-letter characters, drop stopwords, stem each word with Porter stemmer
3. **Split**: 80/20 train/test split, stratified, done before any resampling
4. **Vectorize**: TF-IDF with unigrams and bigrams, top 8,000 features, fit only on the training set
5. **Baseline models**: train Logistic Regression and Random Forest on the raw, imbalanced training data
6. **SMOTE**: oversample the minority classes, but only in the training set (the test set is left untouched), then retrain both models
7. **Compare**: check Negative-class recall before and after SMOTE
8. **Hyperparameter tuning**: 3-fold cross-validation with GridSearchCV. SMOTE is applied inside each fold (using imblearn's Pipeline) so synthetic samples don't leak across folds. Tunes Logistic Regression's C and Random Forest's n_estimators/max_depth
9. **Error analysis**: look at actual misclassified reviews to find patterns
10. **Feature importance**: check which words each model actually relies on
## Results
 
| Model | Accuracy | Macro F1 | Negative recall |
|---|---|---|---|
| Logistic Regression (baseline) | 0.94 | 0.44 | 0.20 |
| Random Forest (baseline) | 0.93 | 0.36 | 0.07 |
| Logistic Regression + SMOTE (default C) | 0.87 | 0.52 | 0.52 |
| Random Forest + SMOTE (default params) | 0.93 | 0.40 | 0.12 |
| Logistic Regression + SMOTE, tuned (C=1, via 3-fold CV) | 0.87 | 0.52 | 0.52 |
| Random Forest + SMOTE, tuned (max_depth=30, n_estimators=200) | 0.91 | 0.52 | 0.33 |
 
**SMOTE's effect on Logistic Regression** (the stronger model here): Negative-class recall went from about 20% to about 52%. That's a 32-point jump, or roughly 162% relative.
 
**Hyperparameter tuning**: Logistic Regression's default C=1 was already the best setting. Random Forest, on the other hand, was overfitting with unlimited depth. Capping max_depth at 30 nearly tripled its Negative recall, from 12% to 33%, and brought its macro F1 up to match Logistic Regression.
 
**Error analysis**: about 12.9% of the test set was misclassified. Looking through these, most Negative reviews that got labeled Positive use mild or indirect complaints ("everything works fine, but the line bothers me," "biggest complaint is...") instead of strongly negative words. TF-IDF just counts word frequency, so it can't pick up on negation, sarcasm, or tone. A review that complains politely ends up looking similar to a positive one. A transformer model like BERT would probably handle this better, at the cost of a lot more compute.
 
**Feature importance**: the top words for each class make sense. Words like "disappoint," "terrible," "slow," and "return" push toward Negative. Words like "love," "great," "excellent," and "amazing" push toward Positive. That's a good sign the models are picking up on real sentiment, not some random pattern in the data.

 
## How to Run
 
```bash
pip install -r requirements.txt
jupyter notebook notebook/amazon_review_sentiment_analysis.ipynb
```
